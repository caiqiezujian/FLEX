"""Auditable audio-only HF Thinker backend, pinned to transformers 4.57.1.

The full checkpoint is loaded by the official Thinker class (its
base_model_prefix is ``thinker``). Frozen audio features are injected only at
processor-expanded input-audio positions. Generated tokens are never decoded
and re-encoded into actor history. No labels/reference text enter this module.
"""
from __future__ import annotations

import contextlib
import json
import math
import subprocess
from pathlib import Path

PROTOCOL_IDS = {"<|WAIT|>": 151676, "<|DIRECT|>": 151677, "<|FLEX|>": 151678}
AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
SYSTEM_PROMPT = (
    "你是一个同声传译专家，负责将用户音频中的语音同声翻译为{target_lang}。"
    "用户将按时间顺序逐块发送音频。每收到一块音频后，立即输出响应："
    "若该块暂无可翻译内容，输出<|WAIT|>；若仅能确认部分译文、语义尚不完整，输出<|FLEX|>并紧跟"
    "确信的译文（只输出有把握的部分，宁少勿错）；若译文语义已完整，输出<|DIRECT|>并紧跟新识别"
    "的译文。所有译文均为增量输出：只输出新增内容，不重复、不修改此前已输出的任何译文。"
)


def parse_response(text: str) -> tuple[str, str]:
    """Validate, but never rewrite sampled actions used for training."""
    tag = next((tag for tag in PROTOCOL_IDS if text.startswith(tag)), None)
    if tag is None:
        raise ValueError("response must start with one protocol token")
    delta = text[len(tag):]
    if "<|" in delta or "|>" in delta:
        raise ValueError("embedded protocol/control token in translation")
    if tag == "<|WAIT|>" and delta.strip():
        raise ValueError("WAIT must not contain translation")
    if tag != "<|WAIT|>" and not delta.strip():
        raise ValueError("FLEX/DIRECT must contain translation")
    return tag, delta


def public_trajectory(value):
    """Strip only private replay tensors; keep raw token/logprob audit fields."""
    if isinstance(value, dict):
        return {k: public_trajectory(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, list):
        return [public_trajectory(v) for v in value]
    return value


def validate_raw_sampling(options: dict) -> None:
    """The stored log-probability must describe the actual behavior policy.

    Top-p/top-k truncation or temperature scaling would change that policy.
    This first implementation deliberately samples the unmodified softmax.
    """
    top_k = options.get("top_k", 0)
    if (float(options.get("temperature", 1.0)) != 1.0
            or float(options.get("top_p", 1.0)) != 1.0
            or type(top_k) is not int or top_k not in (0, -1)):
        raise ValueError("HF GSPO requires raw-policy sampling: temperature=1, top_p=1, top_k=0 or -1")
    for name, default in (("max_new_tokens", 128), ("max_context_tokens", 4096)):
        value = options.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def validate_adapter_state(expected: dict, actual: dict) -> None:
    """Reject a partial/mismatched adapter before any weight is modified."""
    missing, unexpected = set(expected) - set(actual), set(actual) - set(expected)
    if missing or unexpected:
        raise ValueError(f"resume adapter keys mismatch: missing={sorted(missing)[:8]} "
                         f"unexpected={sorted(unexpected)[:8]}")
    if not expected or any("lora_" not in key for key in expected):
        raise ValueError("resume supports the attention-only LoRA adapter, not base-model tensors")
    for key, value in expected.items():
        if tuple(value.shape) != tuple(actual[key].shape):
            raise ValueError(f"resume adapter shape mismatch for {key}: "
                             f"{tuple(actual[key].shape)} != {tuple(value.shape)}")


class CurrentTaskAudioCache:
    """Lazy CPU audio-feature cache shared by candidates of only one task.

    A future interval is never decoded/encoded to populate the cache eagerly.
    Cache identity includes the file stat and exact chunk schedule, not just id.
    Task changes drop this owner's references; active replay objects can still
    safely retain the old immutable tensors until their backward pass finishes.
    """

    def __init__(self):
        self.identity = None
        self.items: dict[tuple[float, float, float], dict] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def chunk_key(chunk: dict) -> tuple[float, float, float]:
        key = tuple(float(chunk[name]) for name in ("start_seconds", "end_seconds", "input_seconds"))
        if not all(math.isfinite(value) for value in key) or not 0 <= key[0] < key[1]:
            raise ValueError("audio chunk interval must be finite and positive")
        if key[2] + 1e-9 < key[1] - key[0]:
            raise ValueError("input_seconds cannot truncate the true chunk")
        return key

    def begin(self, task: dict) -> None:
        path = Path(task["audio_path"]).resolve(strict=True)
        stat = path.stat()
        identity = (task["id"], str(path), stat.st_size, stat.st_mtime_ns,
                    tuple(self.chunk_key(chunk) for chunk in task["chunks"]))
        if identity != self.identity:
            self.clear()
            self.identity = identity

    def get(self, chunk: dict, loader):
        if self.identity is None:
            raise RuntimeError("begin(task) must precede audio cache use")
        key = self.chunk_key(chunk)
        if key not in self.identity[-1]:
            raise ValueError("audio interval is not part of the active task")
        if key not in self.items:
            # Cache only complete successful loads, never partial failed work.
            value = loader()
            self.items[key] = value
            self.misses += 1
        else:
            self.hits += 1
        return self.items[key]

    def clear(self):
        self.identity = None
        self.items.clear()
        self.hits = self.misses = 0


class HFThinkerBackend:
    def __init__(self, config: dict, device):
        import torch
        import transformers
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

        if transformers.__version__ != "4.57.1":
            raise RuntimeError("This backend is audited against transformers==4.57.1; use the dedicated environment")
        self.torch, self.device, self.config = torch, device, config
        self.rollout_config = config.get("rollout", {})
        validate_raw_sampling(self.rollout_config)
        self._progress_callback = config.get("_progress_callback")
        self._audio_cache = CurrentTaskAudioCache()
        options = config.get("model", {})
        self._progress("loading local Qwen3-Omni processor")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(config["model_path"], local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.enc = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        for text, expected in PROTOCOL_IDS.items():
            if self.enc(text) != [expected] or text not in self.tokenizer.all_special_tokens:
                raise ValueError(f"trained tokenizer mismatch: {text} must be the single special token {expected}")
        self.stop_ids = {self.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        if None in self.stop_ids or len(self.stop_ids) != 2:
            raise ValueError("missing distinct ChatML stopping tokens")
        self.audio_pad_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.user_head, self.assistant_head = self.enc("<|im_start|>user\n"), self.enc("<|im_start|>assistant\n")
        self.tail = self.enc("<|im_end|>\n")
        self._progress("loading local Qwen3-Omni Thinker BF16 weights")
        base, loading = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            config["model_path"], torch_dtype=torch.bfloat16,
            attn_implementation=options.get("attn_implementation", "sdpa"),
            device_map={"": str(device)}, local_files_only=True,
            low_cpu_mem_usage=True, output_loading_info=True,
        )
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise RuntimeError(f"Thinker checkpoint did not load completely: {loading}")
        unexpected = [k for k in loading.get("unexpected_keys", [])
                      if not k.startswith(("talker.", "code2wav.", "code_predictor."))]
        if unexpected:
            raise RuntimeError(f"unexpected non-Talker checkpoint tensors: {unexpected[:20]}")
        if max(PROTOCOL_IDS.values()) >= base.get_input_embeddings().weight.shape[0]:
            raise ValueError("checkpoint embeddings do not contain the trained protocol rows")
        base.requires_grad_(False)
        # Full names deliberately exclude audio/vision attention and MoE experts.
        targets = [name for name, module in base.named_modules()
                   if name.startswith("model.layers.") and ".self_attn." in name
                   and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}
                   and isinstance(module, torch.nn.Linear)]
        if not targets:
            raise RuntimeError("no Thinker language attention projections found")
        self.model = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=int(options.get("lora_rank", 8)),
            lora_alpha=int(options.get("lora_alpha", 32)), lora_dropout=0.0,
            target_modules=targets, bias="none", modules_to_save=None,
        ))
        if config.get("resume_adapter_path"):
            self._load_adapter(config["resume_adapter_path"])
        self.base = self.model.get_base_model()
        if options.get("gradient_checkpointing", True):
            self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and "lora_" not in name:
                raise RuntimeError(f"non-LoRA trainable parameter: {name}")
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Dropout) and module.p:
                raise RuntimeError(f"nonzero dropout would invalidate replay: {name}")
            for field in ("jitter_noise", "router_jitter_noise", "attention_dropout"):
                if isinstance(getattr(module, field, None), (float, int)) and getattr(module, field) != 0:
                    raise RuntimeError(f"nonzero {field} would invalidate replay: {name}")
        self.model.eval()
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.trainable_count = sum(p.numel() for p in self.trainable)
        self._native_audit_task = None
        self._progress(f"Thinker ready: {self.trainable_count:,} trainable LoRA parameters")

    def _progress(self, status: str):
        if self._progress_callback is not None:
            self._progress_callback(status)

    def _load_adapter(self, directory: str):
        """Load only a complete local safetensors adapter into the same layout."""
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file

        root = Path(directory).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("resume_adapter_path must be a local adapter directory")
        self._progress(f"loading local RL adapter: {root}")
        with (root / "adapter_config.json").open(encoding="utf-8") as handle:
            saved = json.load(handle)
        active = self.model.peft_config["default"]
        for name in ("r", "lora_alpha", "lora_dropout", "bias", "use_dora", "use_rslora",
                     "fan_in_fan_out", "rank_pattern", "alpha_pattern"):
            if saved.get(name, getattr(active, name)) != getattr(active, name):
                raise ValueError(f"resume adapter configuration differs: {name}")
        if saved.get("peft_type") != "LORA" or set(saved.get("target_modules", [])) != set(active.target_modules):
            raise ValueError("resume adapter type/target modules differ from this backend")
        # No Hub lookup and no unsafe pickle fallback, including on missing files.
        state = load_file(str(root / "adapter_model.safetensors"), device="cpu")
        validate_adapter_state(get_peft_model_state_dict(self.model), state)
        if any(not self.torch.isfinite(tensor).all().item() for tensor in state.values()):
            raise ValueError("resume adapter contains non-finite parameters")
        result = set_peft_model_state_dict(self.model, state)
        missing_adapter = [key for key in result.missing_keys if "lora_" in key]
        if result.unexpected_keys or missing_adapter:
            raise RuntimeError(f"resume adapter was not fully loaded: {result}")

    def _audio_chunk(self, task: dict, chunk: dict) -> dict:
        return self._audio_cache.get(chunk, lambda: self._encode_audio_chunk(task, chunk))

    def _encode_audio_chunk(self, task: dict, chunk: dict) -> dict:
        """Read/encode exactly one arrived interval, never the unseen suffix."""
        import numpy as np
        torch = self.torch
        duration = float(chunk["end_seconds"]) - float(chunk["start_seconds"])
        self._progress(f"id={task['id']} decoding/encoding arrived audio "
                       f"{float(chunk['start_seconds']):.2f}-{float(chunk['end_seconds']):.2f}s")
        # -ss after -i is accurate output seeking; ffmpeg handles arbitrary
        # source codecs/rates/channels. No physical chunk WAV is written.
        decoded = subprocess.run(
            [self.config.get("ffmpeg", "ffmpeg"), "-nostdin", "-v", "error", "-i", task["audio_path"],
             "-ss", str(chunk["start_seconds"]), "-t", str(duration), "-vn", "-f", "f32le",
             "-ac", "1", "-ar", "16000", "pipe:1"], capture_output=True, check=False,
            timeout=float(self.config.get("model", {}).get("audio_decode_timeout_seconds", 120)),
        )
        if decoded.returncode:
            raise RuntimeError("ffmpeg chunk decode failed: " + decoded.stderr.decode(errors="replace")[-2000:])
        audio = np.frombuffer(decoded.stdout, dtype="<f4").copy()
        expected = round(duration * 16000)
        # Container end-time estimates can differ by a tiny number of samples.
        # Never silently turn a materially missing audio interval into silence.
        if abs(len(audio) - expected) > 160 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"audio interval decode length mismatch: expected {expected}, got {len(audio)}")
        audio = np.pad(audio[:expected], (0, max(0, expected - len(audio))))
        target = round(float(chunk["input_seconds"]) * 16000)
        if target < len(audio):
            raise ValueError("input_seconds cannot truncate the true chunk")
        audio = np.pad(audio, (0, target - len(audio)))
        processed = self.processor(text=AUDIO_PLACEHOLDER, audio=[audio], sampling_rate=16000,
                                   return_tensors="pt", add_special_tokens=False)
        ids = processed["input_ids"][0].tolist()
        expected_non_audio = self.enc("<|audio_start|><|audio_end|>")
        if [x for x in ids if x != self.audio_pad_id] != expected_non_audio:
            raise RuntimeError("processor altered the frozen audio bracket layout")
        features = processed["input_features"].to(self.device, dtype=torch.bfloat16)
        mask = processed["feature_attention_mask"].to(self.device)
        self.base.audio_tower.eval()
        with torch.no_grad():
            encoded = self.base.get_audio_features(features, feature_attention_mask=mask)
        encoded = encoded.reshape(-1, encoded.shape[-1]).detach().cpu()
        if ids.count(self.audio_pad_id) != encoded.shape[0]:
            raise RuntimeError("processor expansion and audio encoder disagree")
        return {"ids": ids, "features": encoded, "feature_length": int(mask.sum()),
                "input_features": processed["input_features"].cpu(),
                "feature_attention_mask": processed["feature_attention_mask"].cpu()}

    def _prompt(self, history: list[int], audio_spans: list, feature_lengths: list[int]) -> dict:
        torch = self.torch
        ids = torch.tensor([history + self.assistant_head], device=self.device)
        positions, _ = self.base.get_rope_index(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            audio_seqlens=torch.tensor(feature_lengths, device=self.device), use_audio_in_video=False,
        )
        return {"prompt_ids": ids[0].tolist(), "position_ids": positions.cpu(), "audio_spans": list(audio_spans)}

    def _embeddings(self, replay: dict, actions: list[int] | None = None):
        torch = self.torch
        ids = replay["prompt_ids"] + (actions or [])
        with torch.no_grad():
            embeddings = self.base.get_input_embeddings()(torch.tensor([ids], device=self.device))
            for start, features in replay["audio_spans"]:
                embeddings[0, start:start + len(features)] = features.to(self.device, dtype=embeddings.dtype)
        return embeddings

    def _text_forward(self, embeddings, positions, total_length, cache=None, *, use_cache=None):
        """Same official Thinker text backbone for sampling and teacher forcing."""
        torch = self.torch
        start = total_length - embeddings.shape[1]
        return self.base.model(
            inputs_embeds=embeddings, position_ids=positions,
            attention_mask=torch.ones((1, total_length), dtype=torch.long, device=self.device),
            past_key_values=cache,
            use_cache=not torch.is_grad_enabled() if use_cache is None else use_cache,
            cache_position=torch.arange(start, total_length, device=self.device),
            output_router_logits=False, return_dict=True,
        )

    def sample_turn(self, replay: dict, generator):
        torch = self.torch
        ids, old_logprobs = [], []
        self.model.eval()
        with torch.no_grad():
            embeddings = self._embeddings(replay)
            positions = replay["position_ids"].to(self.device)
            total_length = embeddings.shape[1]
            next_position = int(positions.max()) + 1
            cache = None
            for _ in range(int(self.rollout_config.get("max_new_tokens", 128))):
                output = self._text_forward(embeddings, positions, total_length, cache)
                cache = output.past_key_values
                logits = self.base.lm_head(output.last_hidden_state[:, -1]).float()[0]
                logprobs = torch.log_softmax(logits, dim=-1)
                token = int(torch.multinomial(logprobs.exp(), 1, generator=generator))
                ids.append(token)
                old_logprobs.append(float(logprobs[token]))
                if token in self.stop_ids:
                    return ids, old_logprobs, "stop"
                embeddings = self.base.get_input_embeddings()(torch.tensor([[token]], device=self.device))
                positions = torch.full((3, 1, 1), next_position, dtype=torch.long, device=self.device)
                total_length += 1
                next_position += 1
        return ids, old_logprobs, "length"

    def assert_native_parity(self, replay: dict, audio_items: list[dict]) -> float:
        """Fail closed unless cached audio/backbone agrees with native Thinker.

        Run on every prefix of the first sampled task (including multi-audio
        prefixes). The temporary head pre-hook selects only the final position,
        avoiding an unnecessary [prompt length, vocabulary] logits allocation.
        Neither native position IDs nor native audio features are overridden.
        """
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            embeddings = self._embeddings(replay)
            output = self._text_forward(embeddings, replay["position_ids"].to(self.device),
                                        embeddings.shape[1], use_cache=False)
            custom = self.base.lm_head(output.last_hidden_state[:, -1]).float()
            del output, embeddings
            width = max(item["input_features"].shape[-1] for item in audio_items)
            features = torch.cat([torch.nn.functional.pad(item["input_features"], (0, width - item["input_features"].shape[-1])) for item in audio_items])
            masks = torch.cat([torch.nn.functional.pad(item["feature_attention_mask"], (0, width - item["feature_attention_mask"].shape[-1])) for item in audio_items])
            ids = torch.tensor([replay["prompt_ids"]], device=self.device)
            hook = self.base.lm_head.register_forward_pre_hook(lambda module, args: (args[0][:, -1:],))
            try:
                native = self.base(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    input_features=features.to(self.device, dtype=torch.bfloat16),
                    feature_attention_mask=masks.to(self.device),
                    cache_position=torch.arange(ids.shape[1], device=self.device),
                    use_cache=False, output_router_logits=False, return_dict=True,
                ).logits[:, -1].float()
            finally:
                hook.remove()
            error = float((torch.log_softmax(custom, -1) - torch.log_softmax(native, -1)).abs().max())
        tolerance = float(self.config.get("model", {}).get("native_parity_tolerance", 0.05))
        if not error <= tolerance:
            raise RuntimeError(f"cached-audio/backbone versus official Thinker parity failed: {error} > {tolerance}")
        return error

    def rollout(self, task: dict, group_index: int, seed: int) -> dict:
        torch = self.torch
        self._audio_cache.begin(task)
        language = {"en": "英文", "zh": "中文"}.get(task["target_lang"], task["target_lang"])
        prompt = self.config.get("system_prompt", SYSTEM_PROMPT).format(target_lang=language)
        history = self.enc("<|im_start|>system\n") + self.enc(prompt) + self.tail
        audio_spans, feature_lengths, deltas, audio_items = [], [], [], []
        if self._native_audit_task is None:
            self._native_audit_task = task["id"]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        trajectory = {"id": task["id"], "group_index": group_index, "seed": seed,
                      "turns": [], "translation": "", "valid": True, "error": None,
                      "complete_input": False, "final_wait": False}
        for index, chunk in enumerate(task["chunks"]):
            audio = self._audio_chunk(task, chunk)
            audio_start = len(history) + len(self.user_head) + audio["ids"].index(self.audio_pad_id)
            history += self.user_head + audio["ids"] + self.tail
            audio_spans.append((audio_start, audio["features"]))
            feature_lengths.append(audio["feature_length"])
            audio_items.append(audio)
            replay = self._prompt(history, audio_spans, feature_lengths)
            if len(replay["prompt_ids"]) + int(self.rollout_config.get("max_new_tokens", 128)) > int(self.rollout_config.get("max_context_tokens", 4096)):
                raise RuntimeError("context budget exhausted: infrastructure/data budget failure, not a policy reward; shorten complete episodes or increase the audited context budget")
            native_error = None
            if group_index == 0 and task["id"] == self._native_audit_task:
                self._progress(f"id={task['id']} native multi-audio parity audit "
                               f"chunk={index + 1}/{len(task['chunks'])}")
                native_error = self.assert_native_parity(replay, audio_items)
            self._progress(f"id={task['id']} candidate={group_index + 1} "
                           f"generating chunk={index + 1}/{len(task['chunks'])} "
                           f"audio={float(chunk['end_seconds']):.2f}s")
            tokens, probabilities, finish = self.sample_turn(replay, generator)
            display_tokens = tokens[:-1] if tokens and tokens[-1] in self.stop_ids else tokens
            response = self.tokenizer.decode(display_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            turn = {"turn_id": index, "audio_start_seconds": chunk["start_seconds"],
                    "audio_end_seconds": chunk["end_seconds"], "input_seconds": chunk["input_seconds"],
                    "response": response, "token_ids": tokens, "old_logprobs": probabilities,
                    "finish_reason": finish, "prompt_token_ids": replay["prompt_ids"], "_replay": replay,
                    "native_parity_max_logprob_error": native_error}
            trajectory["turns"].append(turn)
            if finish != "stop":
                trajectory.update(valid=False, error="truncated_generation")
                break
            try:
                if tokens[-1] != self.tokenizer.convert_tokens_to_ids("<|im_end|>"):
                    raise ValueError("endoftext instead of the trained ChatML turn boundary")
                tag, delta = parse_response(response)
                if tokens[0] != PROTOCOL_IDS[tag]:
                    raise ValueError("protocol tag must be generated as its trained single special token")
                if any(token in self.tokenizer.all_special_ids for token in tokens[1:-1]):
                    raise ValueError("special token embedded in assistant translation")
            except ValueError as exc:
                trajectory.update(valid=False, error=str(exc))
                break
            # WAIT formatting whitespace stays in raw response/tokens/history,
            # not in the user-visible translation or emitted-text timing.
            turn.update(tag=tag, delta=delta if tag != "<|WAIT|>" else "")
            if tag != "<|WAIT|>" and delta:
                deltas.append(delta)
            # Preserve the sampled terminal token as an action. The newline is an
            # environment-inserted separator, not part of the loss mask.
            history = replay["prompt_ids"] + tokens + self.enc("\n")
        trajectory["translation"] = "".join(deltas)
        trajectory["complete_input"] = len(trajectory["turns"]) == len(task["chunks"])
        trajectory["final_wait"] = bool(trajectory["turns"] and
                                        trajectory["turns"][-1].get("tag") == "<|WAIT|>")
        trajectory["audio_cache"] = {"encoded_chunks": self._audio_cache.misses,
                                     "cache_hits": self._audio_cache.hits}
        return trajectory

    def score_turn(self, turn: dict, reference: bool = False):
        torch = self.torch
        replay, actions = turn["_replay"], turn["token_ids"]
        if not actions:
            raise ValueError("cannot score empty action sequence")
        self.model.train(torch.is_grad_enabled())
        self.base.audio_tower.eval()
        self.base.visual.eval()
        context = self.model.disable_adapter() if reference else contextlib.nullcontext()
        with context:
            # The last action is only a target; do not forward an unused token.
            embeddings = self._embeddings(replay, actions[:-1])
            if torch.is_grad_enabled():
                embeddings.requires_grad_(True)
            positions = replay["position_ids"].to(self.device)
            count = len(actions) - 1
            if count:
                extra = torch.arange(int(positions.max()) + 1, int(positions.max()) + count + 1,
                                     device=self.device)[None, None].expand(3, 1, -1)
                positions = torch.cat((positions, extra), dim=-1)
            output = self._text_forward(embeddings, positions, embeddings.shape[1], use_cache=False)
            hidden = output.last_hidden_state[0, len(replay["prompt_ids"]) - 1:]
            logits = self.base.lm_head(hidden).float()
            labels = torch.tensor(actions, device=self.device)
            return -torch.nn.functional.cross_entropy(logits, labels, reduction="none")

    def save(self, directory: str):
        Path(directory).mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(directory, safe_serialization=True)
        self.processor.save_pretrained(directory)

    def clear_audio_cache(self):
        """Release current-task cache references after its group is consumed."""
        self._audio_cache.clear()


def load_backend(config: dict, device):
    return HFThinkerBackend(config, device)
