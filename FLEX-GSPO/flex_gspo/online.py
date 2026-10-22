"""State transitions for causal multi-turn rollout, independent of CUDA/Swift.

Swift's MultiTurnScheduler owns token-exact history, loss masks, and logprobs.
This mixin ONLY advances the audio environment and records scoring metadata.
It never repairs an assistant output, forces a tag, or adds a reference prompt.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import time

from .audio import encode_chunk_ref
from .data import validate_task
from .prompts import EOF_PROMPT, initial_messages
from .protocol import parse_response
from .trajectory import TRAJECTORY_SCHEMA_VERSION, validate_turn_schedule

LOG = logging.getLogger(__name__)
STATE_KEY = "_flex_episode_state"


def unpack_task(value):
    task = json.loads(value) if isinstance(value, str) else copy.deepcopy(value)
    validate_task(task)
    if (task["source_lang"], task["target_lang"]) != ("en", "zh"):
        raise ValueError("This release supports EN->ZH only")
    if not isinstance(task.get("reference_text"), str) or not task["reference_text"].strip():
        raise ValueError("QALCO requires the machine reference (also for length normalization)")
    return task


class FlexStreamHooks:
    async def on_trajectory_start(self, requests):
        for request in requests:
            data = copy.deepcopy(request.data_dict or {})
            if "flex_task" not in data:
                raise ValueError("Missing flex_task: use 01_prepare.sh and preserve dataset extra columns")
            task = unpack_task(data["flex_task"])
            if request.messages != initial_messages():
                raise ValueError("Initial actor messages differ from SFT prompt + one <audio> user turn")
            expected = [encode_chunk_ref(task, 0)]
            if request.audios != expected:
                raise ValueError("Initial audio must be exactly the first virtual chunk (never the whole WAV)")
            if request.images or request.videos:
                raise ValueError("Audio-only FLEX does not accept images or videos")
            required_turns = len(task["chunks"]) + 1
            if self.max_turns is not None and required_turns > self.max_turns:
                raise ValueError(f"Task {task['id']} needs {required_turns} turns "
                                 f"(audio + one EOF), max_turns={self.max_turns}")
            data[STATE_KEY] = {"task": task, "turns": [], "invalid_reason": None,
                               "done": False, "eof_started_at": None,
                               "next_turn_ready": 1}
            request.data_dict = data
            request.messages = copy.deepcopy(request.messages)
            request.audios = list(expected)

    @staticmethod
    def _state(request):
        try:
            return request.data_dict[STATE_KEY]
        except (TypeError, KeyError) as exc:
            raise RuntimeError("Framework did not initialize the FLEX multi-turn scheduler") from exc

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        state = self._state(infer_request)
        task = state["task"]
        if state["done"] or current_turn != len(state["turns"]) + 1:
            raise RuntimeError("Out-of-order or repeated FLEX turn")
        if state["next_turn_ready"] != current_turn:
            raise RuntimeError("FLEX environment did not prepare the next turn")
        audio_count = len(task["chunks"])
        is_eof = current_turn == audio_count + 1
        expected_audio_count = min(current_turn, audio_count)
        if current_turn > audio_count + 1:
            raise RuntimeError("FLEX permits exactly one EOF finalization turn")
        if len(infer_request.audios) != expected_audio_count:
            raise RuntimeError("Audio count/history mismatch")
        if infer_request.audios != [encode_chunk_ref(task, i) for i in range(expected_audio_count)]:
            raise RuntimeError("Audio history differs from the causal chunk plan")
        if is_eof:
            user_messages = [message for message in infer_request.messages if message["role"] == "user"]
            if (state["eof_started_at"] is None or not user_messages
                    or user_messages[-1]["content"] != EOF_PROMPT):
                raise RuntimeError("EOF response requires the explicit text-only EOF observation")
        token_ids = response_choice.token_ids
        if not isinstance(token_ids, list) or not token_ids or any(type(t) is not int for t in token_ids):
            raise RuntimeError("Sampling must return exact generated token IDs; retokenization is forbidden")
        logprobs = response_choice.logprobs
        values = logprobs.get("content") if isinstance(logprobs, dict) else None
        if not isinstance(values, list) or len(values) != len(token_ids):
            raise RuntimeError("Missing/misaligned sampled logprobs; check the installed rollout backend")
        for value in values:
            number = value.get("logprob") if isinstance(value, dict) else None
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise RuntimeError("Sampling returned a non-finite log probability")
        response = response_choice.message.content
        if not isinstance(response, str):
            raise RuntimeError("Sampling must expose generated response text as well as token IDs")
        reason = response_choice.finish_reason
        if reason not in ("stop", "eos", "length"):
            raise RuntimeError(f"Unexpected inference finish_reason={reason!r}")
        error = None
        try:
            parse_response(response)
        except ValueError as exc:
            error = str(exc)
        if reason == "length":
            error = "Generation hit the per-turn token/context limit"
        turn_record = {
            "response": response,
            "kind": "eof" if is_eof else "audio",
            "audio_end_seconds": (task["duration_seconds"] if is_eof
                                  else task["chunks"][current_turn - 1]["end_seconds"]),
            "finish_reason": reason,
        }
        if is_eof:
            # Includes local scheduling / collective waiting, not GPU kernel time.
            turn_record["eof_elapsed_seconds"] = max(0.0, time.monotonic() - state["eof_started_at"])
        state["turns"].append(turn_record)
        schedule = validate_turn_schedule(task, state["turns"])
        state["invalid_reason"] = error
        state["done"] = bool(error or is_eof)
        state["next_turn_ready"] = None
        trajectory = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "turns": copy.deepcopy(state["turns"]),
            "valid": error is None,
            **schedule,
            "invalid_reason": error,
        }
        if state["done"]:
            LOG.info("FLEX rollout complete id=%s turns=%d/%d valid=%s final_action=%s",
                     task["id"], current_turn, audio_count + 1, error is None,
                     response.split(">", 1)[0] + ">" if ">" in response else "invalid")
        # Essential: propagate ALL observed audios back into the training sample.
        # Otherwise Swift would keep only the first chunk while training all turns.
        return {"done": state["done"], "rollout_infos": {
            "audios": list(infer_request.audios), "flex_trajectory": trajectory,
        }}

    def check_finished(self, infer_request, response_choice, current_turn):
        return self._state(infer_request)["done"]

    def step(self, infer_request, response_choice, current_turn):
        state = self._state(infer_request)
        if state["done"] or len(state["turns"]) != current_turn:
            raise RuntimeError("Cannot append audio to a finished or unobserved turn")
        if state["next_turn_ready"] is not None:
            raise RuntimeError("FLEX next observation was already prepared")
        audio_count = len(state["task"]["chunks"])
        if current_turn < audio_count:
            infer_request.messages.append({"role": "user", "content": "<audio>"})
            infer_request.audios.append(encode_chunk_ref(state["task"], current_turn))
        elif current_turn == audio_count and state["eof_started_at"] is None:
            infer_request.messages.append({"role": "user", "content": EOF_PROMPT})
            state["eof_started_at"] = time.monotonic()
        else:
            raise RuntimeError("FLEX cannot append more than one EOF observation")
        state["next_turn_ready"] = current_turn + 1
        return {"infer_request": infer_request}
