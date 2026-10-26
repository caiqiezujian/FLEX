#!/usr/bin/env python3
"""Check COMET22 transport, batch order consistency, and small quality regressions.

No actor, CUDA, or local COMET installation is needed. Passing synthetic checks
does NOT validate a reward on real data; use --cases-jsonl for reviewed cases.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

if __package__:
    from ._bootstrap import PROJECT_ROOT
else:
    from _bootstrap import PROJECT_ROOT

from flex_gspo.comet22 import ServiceCOMET22, ensure_comet22_ready
from flex_gspo.rewards import RewardError


SYNTHETIC_CASES = [
    {"id": "negation_and_repetition", "source_text": "The meeting will not start at nine tomorrow morning.",
     "reference_text": "会议明天上午九点不会开始。", "preferred": "会议明天上午九点不会开始。",
     "dispreferred": ["会议明天上午九点会开始。", "我喜欢它们。" * 13, "这只猫正在睡觉。"],
     "required_dispreferred_indices": [0, 2]},
    {"id": "short_speech", "source_text": "We leave on Friday.",
     "reference_text": "我们星期五出发。", "preferred": "我们周五出发。",
     "dispreferred": ["我们已经在星期一回来了。", "天气很好。"]},
]


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise RewardError("Quality regression cases must be a nonempty JSONL/list")
    for case in cases:
        if not isinstance(case, dict):
            raise RewardError("Each quality case must be an object")
        for key in ("id", "source_text", "reference_text", "preferred"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise RewardError(f"Quality case requires nonempty {key}")
        worse = case.get("dispreferred")
        if not isinstance(worse, list) or not worse or any(not isinstance(s, str) or not s.strip() for s in worse):
            raise RewardError("Quality case requires nonempty dispreferred text array")
        required = case.get("required_dispreferred_indices", list(range(len(worse))))
        if (not isinstance(required, list) or not required
                or any(type(i) is not int or not 0 <= i < len(worse) for i in required)
                or len(set(required)) != len(required)):
            raise RewardError("required_dispreferred_indices must identify distinct dispreferred candidates")
    return cases


def probe(config, cases=None, *, scorer=None, order_tolerance=0.001):
    try:
        settings = ensure_comet22_ready(config["reward"]["comet22"])
    except (KeyError, TypeError):
        raise RewardError("Configure reward.comet22 before probing the service") from None
    scorer = scorer or ServiceCOMET22(settings)
    cases = validate_cases(SYNTHETIC_CASES if cases is None else cases)
    results = []
    for case in cases:
        candidates = [case["preferred"], *case["dispreferred"]]
        logging.info("COMET22 probe case=%s candidates=%d", case["id"], len(candidates))
        scores = scorer.score_batch(case["source_text"], candidates, case["reference_text"])
        reversed_scores = scorer.score_batch(case["source_text"], list(reversed(candidates)), case["reference_text"])
        restored = list(reversed(reversed_scores))
        order_delta = max(abs(first - second) for first, second in zip(scores, restored))
        # Deliberately no universal COMET quality floor. The test is paired
        # ranking against reviewed bad outputs, not an accuracy percentage.
        passed = [scores[0] > score + 1e-6 for score in scores[1:]]
        required = case.get("required_dispreferred_indices", list(range(len(passed))))
        results.append({
            "id": case["id"], "scores": scores,
            "preferred_above_each_dispreferred": passed,
            "required_dispreferred_indices": required,
            "quality_regression_passed": all(passed[i] for i in required),
            "all_paired_rankings_passed": all(passed),
            "reverse_batch_max_score_difference": order_delta,
            "batch_order_consistency_passed": order_delta <= order_tolerance,
        })
    return {
        "service_connectivity_verified": True,
        "remote_model_identity_verified": False,
        "remote_truncation_behavior_verified": False,
        "declared_service_id": settings["service_id"],
        "declared_model_id": settings["model_id"],
        "score_direction": "higher_is_better",
        "quality_regression_passed": all(item["quality_regression_passed"] for item in results),
        "batch_order_consistency_passed": all(item["batch_order_consistency_passed"] for item in results),
        "case_count": len(results), "results": results,
        "note": "A small paired regression check, not quality calibration or proof against reward hacking. "
                "Default repeated-text ranking is diagnostic: the deterministic degeneration guard must reject it. "
                "Reverse-order consistency cannot independently prove service identity or correct semantic alignment. "
                "Review real audio/transcript/reference failures before long training; no GPU training was tested.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.json"))
    parser.add_argument("--cases-jsonl", help="Reviewed cases: id, source_text, reference_text, preferred, dispreferred[]")
    parser.add_argument("--output", help="Save the complete probe report as JSON")
    parser.add_argument("--transport-only", action="store_true",
                        help="Diagnostic only: return success despite failed quality/order checks; still report failures")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        cases = None
        if args.cases_jsonl:
            with Path(args.cases_jsonl).open(encoding="utf-8-sig") as handle:
                cases = [json.loads(line) for line in handle if line.strip()]
        result = probe(config, cases)
        serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        if not args.transport_only and not (result["quality_regression_passed"] and result["batch_order_consistency_passed"]):
            logging.error("COMET22 quality/order regression failed; inspect the report before training")
            return 2
        return 0
    except (OSError, ValueError, KeyError, TypeError, RewardError) as exc:
        message = str(exc) if isinstance(exc, RewardError) else type(exc).__name__
        logging.error("COMET22 probe failed: %s", message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
