"""QALCO v5: reference MetricX + HPO-inspired quality-gated delay.

MetricX's raw error e is LOWER-is-better. The optimizer's quality signal is
q=-e, not COMET and not 1-e/25. The latter is retained ONLY as a [0,1]
diagnostic for historical monitoring interfaces. Group metadata preserves e
exactly, including the e==5 quality-gate boundary.

Adaptations from HPO are explicit: whole-trajectory scoring, FLEX's existing
all-unit positive source-clock lag + EOF proxy, scale floors/clipping, and
quality-qualified brevity/repetition/failure handling. No extra model, audio
alignment, reference leakage into actor observations, or HTTP calls here.
"""
from __future__ import annotations

import copy
import math
import statistics

from .gated_reward import _number, _quantile
from .rewards import RewardError


DEFAULT_V5 = {
    "max_metricx_error": 5.0,
    "quality_weight": 1.0,
    "latency_weight": 0.5,
    "quality_std_floor": 0.5,
    "latency_std_floor": 0.5,
    "normalized_component_clip": 2.0,
    "latency_cost_cap_seconds": 10.0,
    "brevity_weight": 0.08,
    "repetition_weight": 0.20,
    "anchor_metricx_tolerance": 0.75,
    "anchor_min_candidates": 3,
    "anchor_quantile": 0.25,
    "brevity_slack_units": 2.0,
    "brevity_slack_ratio": 0.05,
    "brevity_denominator_floor": 20.0,
    "failure_margin": 0.25,
}
QUALITY_TRANSFORM = "one_minus_metricx_error_over_25_diagnostic_only"


def validate_v5_settings(settings):
    if not isinstance(settings, dict):
        raise RewardError("v5 settings must be an object")
    result = copy.deepcopy(settings)
    supplied = result.get("v5", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_V5):
        raise RewardError("Unsupported reward.v5 settings")
    values = {**DEFAULT_V5, **supplied}
    for key, value in values.items():
        if key == "anchor_min_candidates":
            if type(value) is not int or value < 3:
                raise RewardError("v5 anchor_min_candidates must be an integer >=3")
        else:
            values[key] = _number(value, "reward.v5." + key)
    for key in ("quality_weight", "quality_std_floor", "latency_std_floor",
                "normalized_component_clip", "latency_cost_cap_seconds",
                "brevity_denominator_floor", "failure_margin"):
        if values[key] <= 0:
            raise RewardError("reward.v5." + key + " must be positive")
    for key in ("max_metricx_error", "anchor_metricx_tolerance", "quality_std_floor"):
        if values[key] > 25:
            raise RewardError("reward.v5." + key + " must be <=25 MetricX error points")
    if values["max_metricx_error"] >= 25:
        raise RewardError("v5 max_metricx_error must be below 25")
    if values["anchor_quantile"] > 1 or values["brevity_slack_ratio"] > 1:
        raise RewardError("v5 anchor_quantile/brevity_slack_ratio must be <=1")
    if values["latency_weight"] > values["quality_weight"]:
        raise RewardError("v5 latency_weight must not exceed quality_weight")
    if any(values[key] > 10 for key in ("quality_weight", "latency_weight",
            "normalized_component_clip", "brevity_weight", "repetition_weight", "failure_margin")):
        raise RewardError("v5 signal weights/clipping/margin must be <=10")
    if values["latency_cost_cap_seconds"] > 120 or values["latency_std_floor"] > values["latency_cost_cap_seconds"]:
        raise RewardError("v5 latency scale floor must be <= cap <=120 seconds")
    budget = _number(result.get("latency_budget_seconds", 3.), "latency_budget_seconds", positive=True)
    if budget > values["latency_cost_cap_seconds"]:
        raise RewardError("v5 diagnostic latency budget must be <= latency cost cap")
    result["latency_budget_seconds"] = budget
    # A derived compatibility diagnostic, never an independent COMET threshold.
    floor = 1. - values["max_metricx_error"] / 25.
    if "quality_floor" in result:
        observed = _number(result["quality_floor"], "quality_floor", maximum=1)
        if not math.isclose(observed, floor, abs_tol=1e-12, rel_tol=0):
            raise RewardError("v5 quality_floor is derived from max_metricx_error; configure the raw MetricX threshold")
    result.update(quality_floor=floor, v5=values)
    return result


def provisional_v5(raw_error, latency, repetition, settings):
    """Absolute diagnostics only; final rewards require the complete G8."""
    settings = validate_v5_settings(settings)
    v = settings["v5"]
    error = _number(raw_error, "metricx_error", maximum=25)
    delay = _number(latency, "latency")
    repeat = _number(repetition, "repetition", maximum=1)
    passed = error <= v["max_metricx_error"]
    delay_pass = delay <= settings["latency_budget_seconds"]
    raw_cost = min(delay, v["latency_cost_cap_seconds"])
    cost = raw_cost if passed else v["latency_cost_cap_seconds"]
    return {
        "metricx_quality_reward": -error, "quality_base": -error,
        "quality_transform": QUALITY_TRANSFORM,
        "metricx_error_threshold": v["max_metricx_error"],
        "quality_gate": float(passed), "quality_pass": passed,
        "latency_pass": delay_pass, "feasible": passed and delay_pass,
        "latency_cost_raw": raw_cost, "latency_penalty": cost,
        "latency_cost_cap_seconds": v["latency_cost_cap_seconds"],
        "repeat_penalty": repeat, "weighted_repetition_penalty": v["repetition_weight"] * repeat,
        "brevity_penalty": 0., "weighted_brevity_penalty": 0., "length_penalty": 0.,
        "tier": "quality_fail" if not passed else "latency_fail" if not delay_pass else "feasible",
        "reward": 0.,
    }


def _moments(values, floor):
    mean = statistics.fmean(values) if values else 0.
    std = statistics.pstdev(values) if len(values) > 1 else 0.
    return mean, std, max(std, floor)


def finalize_group(items, settings):
    """Finalize ONE complete prompt group, preserving candidate order/evidence.

Only scored, valid, nonempty candidates define the component moments. Missing
outputs are policy failures, not invented MetricX scores. Center-only upstream
GSPO is mandatory; positive relative advantage is not semantic certification.
"""
    settings = validate_v5_settings(settings)
    v = settings["v5"]
    if not isinstance(items, (list, tuple)) or len(items) < 2:
        raise RewardError("v5 requires a complete group of at least two candidates")
    required = {"eligible", "quality", "metricx_error", "latency", "units", "repetition", "reward"}
    rows = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or required - set(item) or type(item["eligible"]) is not bool:
            raise RewardError(f"Malformed v5 group item {index}; real raw MetricX scores are required")
        row = copy.deepcopy(item)
        if type(row["units"]) is not int or row["units"] < int(row["eligible"]):
            raise RewardError("v5 emitted units must be nonnegative; eligible requires >=1")
        _number(row["reward"], "provisional reward", minimum=None)
        row["repetition"] = _number(row["repetition"], "repetition", maximum=1)
        if row["latency"] is not None or row["eligible"]:
            row["latency"] = _number(row["latency"], "latency")
        if row["eligible"]:
            error = row["metricx_error"] = _number(row["metricx_error"], "metricx_error", maximum=25)
            quality = row["quality"] = _number(row["quality"], "quality diagnostic", maximum=1)
            if not math.isclose(quality, 1. - error / 25., rel_tol=0, abs_tol=1e-12):
                raise RewardError("v5 raw MetricX score disagrees with quality diagnostic")
            row.update(provisional_v5(error, row["latency"], row["repetition"], settings))
        elif row["metricx_error"] is not None or row["quality"] is not None:
            raise RewardError("v5 policy failures must not carry fabricated quality scores")
        rows.append(row)
    valid = [row for row in rows if row["eligible"]]
    qm, qs, qscale = _moments([r["metricx_quality_reward"] for r in valid], v["quality_std_floor"])
    lm, ls, lscale = _moments([r["latency_penalty"] for r in valid], v["latency_std_floor"])
    best_error = min((r["metricx_error"] for r in valid), default=v["max_metricx_error"])
    threshold = min(v["max_metricx_error"], best_error + v["anchor_metricx_tolerance"])
    pool = [r["units"] for r in valid if r["metricx_error"] <= threshold]
    anchor = _quantile(pool, v["anchor_quantile"]) if len(pool) >= v["anchor_min_candidates"] else None
    slack = max(v["brevity_slack_units"], v["brevity_slack_ratio"] * anchor) if anchor is not None else None
    clip = v["normalized_component_clip"]
    for row in valid:
        zq = min(clip, max(-clip, (row["metricx_quality_reward"] - qm) / qscale))
        zl = min(clip, max(-clip, (row["latency_penalty"] - lm) / lscale))
        excess = min(1., max(0., (row["units"] - anchor - slack) / max(anchor, v["brevity_denominator_floor"]))) if anchor is not None else 0.
        bc = v["brevity_weight"] * row["quality_gate"] * excess
        rp = row["weighted_repetition_penalty"]
        reward = v["quality_weight"] * zq - v["latency_weight"] * zl - bc - rp
        row.update(normalized_quality=zq, normalized_latency=zl,
                   weighted_quality_signal=v["quality_weight"] * zq,
                   weighted_latency_signal=v["latency_weight"] * zl,
                   brevity_penalty=excess, length_penalty=excess, group_brevity_excess=excess,
                   weighted_brevity_penalty=bc, reward=reward)
    failure_reward = min((r["reward"] for r in valid), default=0.) - v["failure_margin"]
    for row in rows:
        if not row["eligible"]:
            row.update(reward=failure_reward, normalized_quality=None, normalized_latency=None,
                       weighted_quality_signal=None, weighted_latency_signal=None,
                       metricx_quality_reward=None, weighted_brevity_penalty=0.,
                       weighted_repetition_penalty=0., group_brevity_excess=0., length_penalty=0.)
        row.update(reward_version=5, group_eligible_count=len(valid), group_all_invalid=not valid,
                   group_all_quality_fail=bool(valid) and all(not r["quality_pass"] for r in valid),
                   quality_group_mean=qm if valid else None, quality_group_std=qs if valid else None,
                   metricx_group_mean=-qm if valid else None, metricx_group_std=qs if valid else None,
                   latency_group_mean=lm if valid else None, latency_group_std=ls if valid else None,
                   quality_normalization_scale=qscale, latency_normalization_scale=lscale,
                   group_length_anchor=anchor, anchor_pool_size=len(pool), group_length_slack=slack,
                   anchor_metricx_threshold=threshold, anchor_quality_threshold=1. - threshold / 25.,
                   failure_reward=failure_reward, advantage_scale="group_center_only",
                   normalization_population="eligible_candidates", quality_transform=QUALITY_TRANSFORM,
                   metricx_error_threshold=v["max_metricx_error"],
                   latency_cost_cap_seconds=v["latency_cost_cap_seconds"])
    mean = statistics.fmean(r["reward"] for r in rows)
    for row in rows:
        row["group_reward_mean"] = mean
        row["estimated_advantage"] = row["reward"] - mean
        row["low_quality_positive_advantage"] = bool(row["eligible"] and not row["quality_pass"] and row["estimated_advantage"] > 1e-12)
        if not math.isfinite(row["reward"]) or not math.isfinite(row["estimated_advantage"]):
            raise RewardError("Nonfinite v5 final reward")
    return rows
