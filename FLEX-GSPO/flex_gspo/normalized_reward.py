"""QALCO v4: bounded component scaling on one complete prompt group.

Only genuinely scored, nonempty, protocol-valid trajectories participate in
quality/latency moments. A policy failure is NOT a fabricated COMET score.
The upstream trainer must use scale_rewards=none: it subtracts the full G8
mean, but does not divide this already scaled signal by another group std.
This objective is an experimental GSPO variant, not a semantic verifier.
"""
from __future__ import annotations

import copy
import math
import statistics

from .gated_reward import _number, _latency_cost, _quantile
from .rewards import RewardError


DEFAULT_V4 = {
    "quality_weight": 1.0, "latency_weight": 0.25,
    "quality_std_floor": 0.03, "latency_std_floor": 0.10,
    "normalized_component_clip": 2.0,
    "quality_gate_low": 0.80, "quality_gate_high": 0.85,
    "latency_cost_cap": 2.0, "latency_linear_weight": 0.05,
    "latency_over_budget_weight": 0.20,
    "brevity_weight": 0.08, "repetition_weight": 0.20,
    "anchor_quality_tolerance": 0.03, "anchor_min_candidates": 3,
    "anchor_quantile": 0.25, "brevity_slack_units": 2.0,
    "brevity_slack_ratio": 0.05, "brevity_denominator_floor": 20.0,
    "failure_margin": 0.25,
}


def validate_v4_settings(settings):
    if not isinstance(settings, dict):
        raise RewardError("v4 settings must be an object")
    result = copy.deepcopy(settings)
    result["quality_floor"] = _number(result.get("quality_floor", .8), "quality_floor", maximum=1, positive=True)
    result["latency_budget_seconds"] = _number(result.get("latency_budget_seconds", 3.), "latency_budget_seconds", positive=True)
    supplied = result.get("v4", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_V4):
        raise RewardError("Unsupported reward.v4 settings")
    values = {**DEFAULT_V4, **supplied}
    for key, value in values.items():
        if key == "anchor_min_candidates":
            if type(value) is not int or value < 3:
                raise RewardError("v4 anchor_min_candidates must be an integer >=3")
        else:
            values[key] = _number(value, "reward.v4." + key)
    for key in ("quality_weight", "quality_std_floor", "latency_std_floor",
                "normalized_component_clip", "latency_cost_cap", "brevity_denominator_floor", "failure_margin"):
        if values[key] <= 0:
            raise RewardError("reward.v4." + key + " must be positive")
    for key in ("quality_gate_low", "quality_gate_high", "anchor_quality_tolerance", "anchor_quantile", "brevity_slack_ratio"):
        if values[key] > 1:
            raise RewardError("reward.v4." + key + " must be <=1")
    if not result["quality_floor"] <= values["quality_gate_low"] < values["quality_gate_high"]:
        raise RewardError("v4 requires quality_floor <= gate_low < gate_high <=1")
    if values["latency_weight"] > values["quality_weight"]:
        raise RewardError("v4 latency_weight must not exceed quality_weight")
    # Bound signals rather than accepting arbitrarily large finite parameters.
    if any(values[key] > 10 for key in ("quality_weight", "latency_weight", "normalized_component_clip", "latency_cost_cap",
                                      "brevity_weight", "repetition_weight", "failure_margin")):
        raise RewardError("v4 signal weights/clipping/margin must be <=10")
    result["v4"] = values
    return result


def _clip01(value):
    return min(1., max(0., value))


def provisional_v4(quality, latency, repetition, settings):
    """Unnormalized diagnostics only; forbidden as the final training reward."""
    settings = validate_v4_settings(settings)
    q = _number(quality, "quality", maximum=1)
    d = _number(latency, "latency")
    p = _number(repetition, "repetition", maximum=1)
    v = settings["v4"]
    gate = _clip01((q - v["quality_gate_low"]) / (v["quality_gate_high"] - v["quality_gate_low"]))
    raw = _latency_cost(d, settings["latency_budget_seconds"], v)
    gated = (1 - gate) * v["latency_cost_cap"] + gate * raw
    qp, lp = q >= settings["quality_floor"], d <= settings["latency_budget_seconds"]
    return {"quality_base": q, "quality_gate": gate, "latency_cost_raw": raw,
            "latency_penalty": gated, "repeat_penalty": p, "brevity_penalty": 0.,
            "weighted_brevity_penalty": 0., "weighted_repetition_penalty": v["repetition_weight"] * p,
            "length_penalty": 0., "reward": 0., "quality_pass": qp, "latency_pass": lp,
            "feasible": qp and lp, "tier": "quality_fail" if not qp else "latency_fail" if not lp else "feasible"}


def _moments(values, floor):
    mean = statistics.fmean(values) if values else 0.
    std = statistics.pstdev(values) if len(values) > 1 else 0.
    return mean, std, max(std, floor)


def finalize_group(items, settings):
    """Return same-order copies; callers guarantee this is exactly one full G8.

Failures stay below the worst scored candidate by a small fixed margin, rather
than a distant -4 sentinel. This reduces, but cannot abolish, poor translations
having positive relative advantage. An all-failure group has zero advantage.
Neither a positive advantage nor eligibility certifies semantic correctness.
"""
    settings = validate_v4_settings(settings)
    v = settings["v4"]
    if not isinstance(items, (list, tuple)) or len(items) < 2:
        raise RewardError("v4 requires a complete group of at least two candidates")
    rows = []
    required = {"eligible", "quality", "latency", "units", "repetition", "reward"}
    for index, item in enumerate(items):
        if not isinstance(item, dict) or required - set(item) or type(item["eligible"]) is not bool:
            raise RewardError(f"Malformed v4 group item {index}")
        row = copy.deepcopy(item)
        if type(row["units"]) is not int or row["units"] < int(row["eligible"]):
            raise RewardError("v4 emitted units must be nonnegative; eligible requires >=1")
        _number(row["reward"], "provisional reward", minimum=None)
        row["repetition"] = _number(row["repetition"], "repetition", maximum=1)
        for key, maximum in (("quality", 1), ("latency", None)):
            if row[key] is not None or row["eligible"]:
                row[key] = _number(row[key], key, maximum=maximum)
        if row["eligible"]:
            row.update(provisional_v4(row["quality"], row["latency"], row["repetition"], settings))
        rows.append(row)
    valid = [row for row in rows if row["eligible"]]
    qm, qs, qscale = _moments([r["quality"] for r in valid], v["quality_std_floor"])
    lm, ls, lscale = _moments([r["latency_penalty"] for r in valid], v["latency_std_floor"])
    threshold = max(settings["quality_floor"], max((r["quality"] for r in valid), default=0.) - v["anchor_quality_tolerance"])
    pool = [r["units"] for r in valid if r["quality"] >= threshold]
    anchor = _quantile(pool, v["anchor_quantile"]) if len(pool) >= v["anchor_min_candidates"] else None
    slack = max(v["brevity_slack_units"], v["brevity_slack_ratio"] * anchor) if anchor is not None else None
    clip = v["normalized_component_clip"]
    for row in valid:
        zq = min(clip, max(-clip, (row["quality"] - qm) / qscale))
        zl = min(clip, max(-clip, (row["latency_penalty"] - lm) / lscale))
        excess = _clip01((row["units"] - anchor - slack) / max(anchor, v["brevity_denominator_floor"])) if anchor is not None else 0.
        bc = v["brevity_weight"] * row["quality_gate"] * excess
        rp = row["weighted_repetition_penalty"]
        reward = v["quality_weight"] * zq - v["latency_weight"] * zl - bc - rp
        row.update(normalized_quality=zq, normalized_latency=zl,
                   weighted_quality_signal=v["quality_weight"] * zq,
                   weighted_latency_signal=v["latency_weight"] * zl,
                   brevity_penalty=excess, length_penalty=excess, group_brevity_excess=excess,
                   weighted_brevity_penalty=bc, reward=reward,
                   # Absolute diagnostic only, NOT fed back to the optimizer.
                   utility_raw=row["quality"] - row["latency_penalty"] - bc - rp)
    failure_reward = min((r["reward"] for r in valid), default=0.) - v["failure_margin"]
    for row in rows:
        if not row["eligible"]:
            row.update(reward=failure_reward, normalized_quality=None, normalized_latency=None,
                       weighted_quality_signal=None, weighted_latency_signal=None,
                       weighted_brevity_penalty=0., weighted_repetition_penalty=0.,
                       group_brevity_excess=0., length_penalty=0., utility_raw=None)
        row.update(reward_version=4, group_eligible_count=len(valid), group_all_invalid=not valid,
                   group_all_quality_fail=bool(valid) and all(r["quality"] < settings["quality_floor"] for r in valid),
                   quality_group_mean=qm if valid else None, quality_group_std=qs if valid else None,
                   latency_group_mean=lm if valid else None, latency_group_std=ls if valid else None,
                   quality_normalization_scale=qscale, latency_normalization_scale=lscale,
                   group_length_anchor=anchor, anchor_pool_size=len(pool), group_length_slack=slack,
                   anchor_quality_threshold=threshold, failure_reward=failure_reward,
                   advantage_scale="group_center_only", normalization_population="eligible_candidates")
    mean = statistics.fmean(r["reward"] for r in rows)
    for row in rows:
        row["group_reward_mean"] = mean
        row["estimated_advantage"] = row["reward"] - mean
        row["low_quality_positive_advantage"] = bool(row["eligible"] and row["quality"] < settings["quality_floor"] and row["estimated_advantage"] > 1e-12)
        if not math.isfinite(row["reward"]) or not math.isfinite(row["estimated_advantage"]):
            raise RewardError("Nonfinite v4 final reward")
    return rows
