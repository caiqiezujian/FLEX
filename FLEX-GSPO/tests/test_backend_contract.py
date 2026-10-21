"""CPU contracts for causal audio caching, raw actions, and adapter reload.

These checks do not load model weights and are not a GPU/model parity test.
The real backend separately checks native multi-audio and replay log-probs.
"""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flex_gspo.backend import (CurrentTaskAudioCache, HFThinkerBackend, PROTOCOL_IDS,
                               parse_response, public_trajectory, validate_adapter_state,
                               validate_raw_sampling)


class RawPolicyContractTests(unittest.TestCase):
    def test_unmodified_sampling_only(self):
        for settings in ({}, {"temperature": 1, "top_p": 1, "top_k": -1}):
            validate_raw_sampling(settings)
        for settings in ({"temperature": .7}, {"temperature": float("nan")},
                         {"top_p": .9}, {"top_k": 20}, {"top_k": .5},
                         {"top_k": False}, {"max_new_tokens": 0},
                         {"max_new_tokens": True}, {"max_context_tokens": 1.2}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_raw_sampling(settings)

    def test_wait_has_no_translation_but_keeps_exact_whitespace(self):
        self.assertEqual(parse_response("<|WAIT|>"), ("<|WAIT|>", ""))
        self.assertEqual(parse_response("<|WAIT|> "), ("<|WAIT|>", " "))
        with self.assertRaises(ValueError):
            parse_response("<|WAIT|>你好")

    def test_append_only_delta_is_not_stripped_or_rewritten(self):
        text = "<|FLEX|> 面向学生，"
        self.assertEqual(parse_response(text), ("<|FLEX|>", " 面向学生，"))
        self.assertEqual(parse_response("<|DIRECT|>We"), ("<|DIRECT|>", "We"))

    def test_protocol_and_control_tokens_fail_closed(self):
        for text in ("hello", " <|DIRECT|>hello", "<|DIRECT|>",
                     "<|FLEX|>  ", "<|DIRECT|>x<|WAIT|>", "<|DIRECT|>x<|im_end|>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_response(text)

    def test_public_record_keeps_raw_actions_and_old_probabilities(self):
        value = {"turns": [{"token_ids": [151677, 12, 151645], "old_logprobs": [-1., -2., -3.],
                            "_replay": object(), "response": "<|DIRECT|>x"}], "_private": object()}
        result = public_trajectory(value)
        self.assertNotIn("_private", result)
        self.assertNotIn("_replay", result["turns"][0])
        self.assertEqual(result["turns"][0]["token_ids"], value["turns"][0]["token_ids"])
        self.assertEqual(result["turns"][0]["old_logprobs"], value["turns"][0]["old_logprobs"])


class CausalAudioCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="flex-audio-cache-")
        self.audio = Path(self.temporary.name) / "source.wav"
        self.audio.write_bytes(b"fixture-not-real-audio")
        self.chunks = [{"start_seconds": i, "end_seconds": i + 1, "input_seconds": 1}
                       for i in range(3)]
        self.task = {"id": "one", "audio_path": str(self.audio), "chunks": self.chunks}
        self.cache = CurrentTaskAudioCache()

    def tearDown(self):
        self.temporary.cleanup()

    def test_lazy_arrival_and_reuse_across_eight_candidates(self):
        calls = []
        self.cache.begin(self.task)
        self.assertEqual(len(self.cache.items), 0)  # no future audio is eagerly encoded
        first = {"features": object()}
        for candidate in range(8):
            self.cache.begin(self.task)
            value = self.cache.get(self.chunks[0], lambda: calls.append(0) or first)
            self.assertIs(value, first)
        self.assertEqual(calls, [0])
        self.assertEqual(self.cache.hits, 7)
        self.assertEqual(self.cache.misses, 1)
        self.assertEqual(len(self.cache.items), 1)
        self.cache.get(self.chunks[1], lambda: calls.append(1) or {})
        self.assertEqual(calls, [0, 1])

    def test_cache_is_current_task_only(self):
        self.cache.begin(self.task)
        self.cache.get(self.chunks[0], lambda: {"old": 1})
        other = dict(self.task, id="two")
        self.cache.begin(other)
        self.assertFalse(self.cache.items)
        self.assertEqual(self.cache.misses, 0)
        self.assertEqual(self.cache.get(self.chunks[0], lambda: {"new": 2}), {"new": 2})

    def test_changed_file_invalidates_same_id_cache(self):
        self.cache.begin(self.task)
        self.cache.get(self.chunks[0], lambda: {"old": 1})
        self.audio.write_bytes(b"changed-content-with-different-size")
        self.cache.begin(self.task)
        self.assertFalse(self.cache.items)

    def test_changed_schedule_invalidates_same_id_cache(self):
        self.cache.begin(self.task)
        self.cache.get(self.chunks[0], lambda: {})
        self.cache.begin(dict(self.task, chunks=self.chunks[:1]))
        self.assertFalse(self.cache.items)

    def test_labels_and_references_do_not_enter_cache_identity(self):
        self.cache.begin(self.task)
        item = self.cache.get(self.chunks[0], lambda: {"same": 1})
        self.cache.begin(dict(self.task, source_text="secret", reference_text="secret"))
        self.assertIs(self.cache.get(self.chunks[0], lambda: None), item)

    def test_failed_load_not_cached(self):
        self.cache.begin(self.task)
        def fail():
            raise RuntimeError("infrastructure decode failure")
        with self.assertRaises(RuntimeError):
            self.cache.get(self.chunks[0], fail)
        self.assertEqual(self.cache.misses, 0)
        self.assertFalse(self.cache.items)
        self.assertEqual(self.cache.get(self.chunks[0], lambda: {"ok": 1}), {"ok": 1})

    def test_interval_must_belong_to_active_task(self):
        with self.assertRaises(RuntimeError):
            self.cache.get(self.chunks[0], lambda: {})
        self.cache.begin(self.task)
        with self.assertRaises(ValueError):
            self.cache.get({"start_seconds": 5, "end_seconds": 6, "input_seconds": 1}, lambda: {})

    def test_invalid_chunk_rejected(self):
        for chunk in ({"start_seconds": -1, "end_seconds": 1, "input_seconds": 2},
                      {"start_seconds": 0, "end_seconds": 2, "input_seconds": 1},
                      {"start_seconds": 0, "end_seconds": float("nan"), "input_seconds": 1}):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                self.cache.begin(dict(self.task, chunks=[chunk]))

    def test_clear_does_not_mutate_replay_owned_features(self):
        self.cache.begin(self.task)
        owned = self.cache.get(self.chunks[0], lambda: {"features": [1, 2, 3]})
        self.cache.clear()
        self.assertEqual(owned["features"], [1, 2, 3])
        self.assertFalse(self.cache.items)


class AdapterStateContractTests(unittest.TestCase):
    def test_exact_shape_and_key_match(self):
        expected = {"module.lora_A.weight": SimpleNamespace(shape=(8, 16))}
        validate_adapter_state(expected, {"module.lora_A.weight": SimpleNamespace(shape=(8, 16))})
        for actual in ({}, {"unexpected.lora_A.weight": SimpleNamespace(shape=(8, 16))},
                       {"module.lora_A.weight": SimpleNamespace(shape=(4, 16))}):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                validate_adapter_state(expected, actual)

    def test_base_model_and_empty_state_not_accepted(self):
        for state in ({}, {"model.embed_tokens.weight": SimpleNamespace(shape=(10, 5))}):
            with self.assertRaises(ValueError):
                validate_adapter_state(state, state)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional torch not installed")
class ReplayTensorContractTests(unittest.TestCase):
    def backend(self):
        backend = HFThinkerBackend.__new__(HFThinkerBackend)
        backend.torch, backend.device = torch, torch.device("cpu")
        class TinyBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Embedding(20, 3)
                self.lm_head = torch.nn.Linear(3, 20, bias=False)
                self.audio_tower = torch.nn.Identity()
                self.visual = torch.nn.Identity()
            def get_input_embeddings(self):
                return self.embeddings
        backend.base = TinyBase()
        backend.model = backend.base
        return backend

    def test_embeddings_replace_only_observed_audio_spans(self):
        backend = self.backend()
        source = torch.tensor([[7., 8., 9.], [10., 11., 12.]])
        replay = {"prompt_ids": [1, 2, 3, 4], "audio_spans": [(1, source)]}
        value = backend._embeddings(replay, [5])
        self.assertEqual(tuple(value.shape), (1, 5, 3))
        self.assertTrue(torch.equal(value[0, 1:3], source))
        self.assertTrue(torch.equal(value[0, 0], backend.base.embeddings.weight[1]))
        self.assertTrue(torch.equal(value[0, 4], backend.base.embeddings.weight[5]))
        self.assertFalse(value.requires_grad)
        value[0, 1] = 0
        self.assertTrue(torch.equal(source[0], torch.tensor([7., 8., 9.])))

    def test_replay_targets_include_stop_but_not_environment_separators(self):
        backend = self.backend()
        captures = []
        def forward(embeddings, positions, length, cache=None, *, use_cache=None):
            captures.append((embeddings.shape[1], positions.clone(), use_cache))
            # A tiny deterministic recurrence stands in for causal context.
            return SimpleNamespace(last_hidden_state=embeddings.cumsum(dim=1))
        backend._text_forward = forward
        replay = {"prompt_ids": [1, 2, 3], "audio_spans": [],
                  "position_ids": torch.arange(3)[None, None].expand(3, 1, -1)}
        turn = {"_replay": replay, "token_ids": [4, 5, 6]}
        values = backend.score_turn(turn)
        self.assertEqual(tuple(values.shape), (3,))
        self.assertEqual(captures[0][0], 5)  # prompt + actions except terminal target
        self.assertFalse(captures[0][2])
        self.assertEqual(captures[0][1].shape[-1], 5)
        self.assertFalse(backend.base.audio_tower.training)
        self.assertFalse(backend.base.visual.training)
        (-values.sum()).backward()
        self.assertIsNotNone(backend.base.lm_head.weight.grad)
        with torch.no_grad():
            second = backend.score_turn(turn)
        self.assertTrue(torch.allclose(values.detach(), second))


@unittest.skipIf(torch is None, "optional torch not installed")
class AdapterLocalReloadTests(unittest.TestCase):
    def exercise(self, *, bad_key=False, nonfinite=False, bad_scaling=False):
        with tempfile.TemporaryDirectory(prefix="flex-adapter-load-") as directory:
            active = SimpleNamespace(r=8, lora_alpha=32, lora_dropout=0.0, bias="none",
                                     use_dora=False, use_rslora=False, fan_in_fan_out=False,
                                     rank_pattern={}, alpha_pattern={},
                                     target_modules={"model.layers.0.self_attn.q_proj"})
            saved = dict(vars(active), target_modules=list(active.target_modules), peft_type="LORA")
            if bad_scaling:
                saved["lora_alpha"] = 64
            (Path(directory) / "adapter_config.json").write_text(json.dumps(saved), encoding="utf-8")
            state = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(8, 3)}
            incoming = {key: value.clone() for key, value in state.items()}
            if bad_key:
                incoming = {"not_the_adapter.lora_A.weight": torch.zeros(8, 3)}
            if nonfinite:
                next(iter(incoming.values()))[0, 0] = float("nan")
            loader = Mock(return_value=incoming)
            setter = Mock(return_value=SimpleNamespace(missing_keys=["frozen.base.weight"], unexpected_keys=[]))
            modules = {
                "peft": SimpleNamespace(get_peft_model_state_dict=Mock(return_value=state),
                                         set_peft_model_state_dict=setter),
                "safetensors": SimpleNamespace(),
                "safetensors.torch": SimpleNamespace(load_file=loader),
            }
            backend = HFThinkerBackend.__new__(HFThinkerBackend)
            backend.torch = torch
            backend._progress_callback = None
            backend.model = SimpleNamespace(peft_config={"default": active})
            with patch.dict(sys.modules, modules):
                if bad_key or nonfinite or bad_scaling:
                    with self.assertRaises(ValueError):
                        backend._load_adapter(directory)
                    setter.assert_not_called()
                else:
                    backend._load_adapter(directory)
                    loader.assert_called_once_with(str(Path(directory) / "adapter_model.safetensors"), device="cpu")
                    setter.assert_called_once_with(backend.model, incoming)

    def test_complete_local_state_loads_before_training(self):
        self.exercise()

    def test_mismatched_weights_never_partially_modify_actor(self):
        self.exercise(bad_key=True)

    def test_nonfinite_weights_never_modify_actor(self):
        self.exercise(nonfinite=True)

    def test_changed_lora_scaling_rejected(self):
        self.exercise(bad_scaling=True)


if __name__ == "__main__":
    unittest.main()
