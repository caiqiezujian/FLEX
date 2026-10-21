"""CPU preparation -> causal waveform rollout -> real HTTP reward integration.

The policy responses are controlled test fixtures. The audio reader, scheduler
hooks, MetricX HTTP client, QALCO scorer, and ORM delegate are real. The HTTP
service binds only localhost; no cloud/private service or GPU is contacted.
This does not validate model generation, backpropagation, or Megatron/vLLM.
"""

from __future__ import annotations

from array import array
import asyncio
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
import wave

from flex_gspo.audio import materialize_chunk_refs
from flex_gspo.online import FlexStreamHooks
from flex_gspo.online_reward import OnlineQalcoReward, load_reward_config
from flex_gspo.prepare import PrepareOptions, prepare_dataset


DEPENDENCIES = all(importlib.util.find_spec(name) is not None for name in ("numpy", "soundfile", "requests"))


@unittest.skipUnless(DEPENDENCIES, "Install numpy, soundfile and requests for CPU pipeline integration")
class OnlinePipelineTests(unittest.TestCase):
    def test_raw_data_to_complete_trajectory_http_reward_without_offline_candidates(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "original_wavs"
            audio_dir.mkdir()
            audio = audio_dir / "clip#example.wav"
            samples = array("h", [1000] * 16000 + [2000] * 16000 + [3000] * 4000)
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(samples.tobytes())
            source = "The meeting starts at nine."
            reference = "会议九点开始。"
            raw = root / "raw.jsonl"
            raw.write_text(json.dumps({"key": "clip#example", "lang": "en", "en": {"txt": source},
                                       "zh": {"txt": reference}}, ensure_ascii=False) + "\n", encoding="utf-8")
            report = prepare_dataset(PrepareOptions(str(raw), str(audio_dir), str(root / "prepared"), workers=2))
            self.assertTrue(report["manifest_published"])
            row = json.loads((root / "prepared" / "train.jsonl").read_text(encoding="utf-8"))
            task = json.loads(row["flex_task"])
            original_inventory = (audio.stat().st_size, audio.stat().st_mtime_ns)

            class Scheduler(FlexStreamHooks):
                max_turns = 64

            scheduler = Scheduler()
            plans = [
                ["<|DIRECT|>会议", "<|WAIT|>", "<|DIRECT|>九点开始。"],
                ["<|WAIT|>", "<|WAIT|>", "<|DIRECT|>会议九点开始。"],
                ["<|WAIT|>", "<|WAIT|>", "<|WAIT|>"],
                ["invalid response without a tag"],
                ["<|DIRECT|>胡扯", "<|WAIT|>", "<|DIRECT|>胡扯"],
                ["<|DIRECT|>会议", "<|DIRECT|>九点开始。", "<|WAIT|>"],
            ]
            requests = [SimpleNamespace(messages=deepcopy(row["messages"]), audios=list(row["audios"]),
                                        data_dict={"flex_task": row["flex_task"]}, images=[], videos=[],
                                        uuid=f"candidate-{index}") for index in range(len(plans))]
            asyncio.run(scheduler.on_trajectory_start(requests))
            infos, completions = [], []
            for request, plan in zip(requests, plans):
                for turn, text in enumerate(plan, 1):
                    # This is the same materializer called by the Swift template.
                    payloads = materialize_chunk_refs(request.audios)
                    self.assertEqual(len(payloads), turn)
                    waveforms = [sf.read(io.BytesIO(payload), dtype="float32")[0] for payload in payloads]
                    self.assertTrue(all(len(waveform) == 16000 for waveform in waveforms))
                    self.assertAlmostEqual(float(waveforms[0][0]), 1000 / 32768)
                    if turn == 3:
                        self.assertAlmostEqual(float(waveforms[-1][0]), 3000 / 32768)
                        self.assertTrue((waveforms[-1][4000:] == 0).all())
                    chat = json.dumps(request.messages, ensure_ascii=False)
                    self.assertNotIn(source, chat)
                    self.assertNotIn(reference, chat)
                    choice = SimpleNamespace(message=SimpleNamespace(content=text), token_ids=[10, 11],
                                             logprobs={"content": [{"logprob": -0.2}, {"logprob": -0.3}]},
                                             finish_reason="stop")
                    request.messages.append({"role": "assistant", "content": text})
                    result = asyncio.run(scheduler.on_turn_end(request, choice, turn))
                    if result["done"]:
                        infos.append(result["rollout_infos"])
                        completions.append(text)
                        break
                    scheduler.step(request, choice, turn)
                self.assertTrue(result["done"])

            received = []

            class MetricXHandler(BaseHTTPRequestHandler):
                def do_POST(self):
                    payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    received.append((self.path, payload))
                    scores = [0.5 if mt == reference else 20.0 for mt in payload["mts"]]
                    encoded = json.dumps({"scores": scores}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)

                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), MetricXHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                config = load_reward_config()
                config["reward"]["metricx"].update(service_url=f"http://127.0.0.1:{server.server_port}",
                                                   service_id="cpu-loopback-fixture", mode="reference")
                reward = OnlineQalcoReward(config, log_dir=str(root / "reward_logs"))
                scores = reward(completions, flex_task=[row["flex_task"]] * len(plans), rollout_infos=infos)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            self.assertEqual(len(scores), 6)
            # Same final translation/quality: useful earlier emission is better.
            self.assertGreater(scores[0], scores[1])
            self.assertGreater(scores[5], scores[1])
            # All WAIT / malformed action are not passed to the quality service.
            self.assertEqual(scores[2], -4.0)
            self.assertEqual(scores[3], -4.0)
            self.assertEqual(reward.last_records[4]["components"]["tier"], "quality_fail")
            self.assertEqual(reward.last_records[0]["translation"], reference)
            self.assertEqual(reward.last_records[5]["translation"], reference)
            self.assertEqual(reward.last_records[5]["trajectory"]["turns"][-1]["response"], "<|WAIT|>")
            self.assertEqual(len(received), 1)
            endpoint, payload = received[0]
            self.assertEqual(endpoint, "/metricx24")
            self.assertEqual(payload, {"srcs": [source, source], "mts": [reference, "胡扯胡扯"],
                                       "refs": [reference, reference]})
            self.assertEqual(infos[0]["audios"], requests[0].audios)
            self.assertEqual(len(infos[0]["audios"]), len(task["chunks"]))
            log_files = list((root / "reward_logs").glob("*.jsonl"))
            self.assertEqual(len(log_files), 1)
            self.assertEqual(len(log_files[0].read_text(encoding="utf-8").splitlines()), 6)
            self.assertEqual((audio.stat().st_size, audio.stat().st_mtime_ns), original_inventory)
            self.assertEqual(list(audio_dir.iterdir()), [audio])


if __name__ == "__main__":
    unittest.main()
