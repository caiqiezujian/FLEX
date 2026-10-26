#!/usr/bin/env python3
"""Probe the configured MetricX service with synthetic text, without a GPU."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

if __package__:
    from ._bootstrap import PROJECT_ROOT
else:
    from _bootstrap import PROJECT_ROOT

from flex_gspo.metricx import ServiceMetricX, ensure_metricx_ready, validate_metricx_settings
from flex_gspo.rewards import RewardError


SOURCE = "The meeting will start at nine tomorrow morning."
CANDIDATES = ["会议将于明天上午九点开始。", "明天上午九点开会。"]
REFERENCE = "会议将在明天早上九点开始。"


def probe(config, mode=None):
    """A successful reply proves this text request worked, not quality validity."""
    ensure_metricx_ready(config)
    settings = validate_metricx_settings(config["reward"]["metricx"])
    modes = ("qe", "reference") if mode == "both" else (mode or settings["mode"],)
    if any(item not in ("qe", "reference") for item in modes):
        raise RewardError("Probe mode must be qe, reference or both")
    results = {}
    for selected in modes:
        logging.info("MetricX connectivity probe: mode=%s candidates=%s", selected, len(CANDIDATES))
        scorer = ServiceMetricX(dict(settings, mode=selected))
        raw = scorer.score_batch(SOURCE, CANDIDATES, REFERENCE if selected == "reference" else None)
        results[selected] = {"scores": raw, "qalco_quality": [1 - value / 25 for value in raw]}
    return {
        "service_connectivity_verified": True,
        "remote_model_identity_verified": False,
        "declared_service_id": settings["service_id"],
        "tested_modes": list(modes),
        "candidate_count_per_mode": len(CANDIDATES),
        "score_direction": "lower_is_better",
        "results": results,
        "note": "Synthetic text transport check only. Array order follows the service contract; "
                "quality calibration, truncation behavior, remote model identity and GPU training are not verified.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.json"))
    parser.add_argument("--mode", choices=("qe", "reference", "both"),
                        help="default: use configured mode; both checks the two endpoints separately")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        result = probe(config, args.mode)
    except (OSError, ValueError, KeyError, TypeError, RewardError) as exc:
        # Service errors are already sanitized; never echo file contents or payloads.
        message = str(exc) if isinstance(exc, RewardError) else type(exc).__name__
        logging.error("MetricX probe failed: %s", message)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
