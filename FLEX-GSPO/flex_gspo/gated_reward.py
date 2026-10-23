"""Pure-Python QALCO v3 quality-gated, complete-group trajectory rewards.

The caller must supply the complete candidates for ONE prompt instance before
calling ``finalize_group``. This module performs no distributed communication,
quality-model calls, group-normalisation, or token-level credit assignment.
COMET quality and source-clock latency remain proxies, not semantic guarantees.
"""

import copy
import math

from .rewards import RewardError


DEFAULT_V3 = {
    "quality_deficit_weight": 2.0,
    "quality_gate_low": 0.75,
    "quality_gate_high": 0.85,
    "latency_cost_cap": 2.0,
    "latency_linear_weight": 0.05,
    "latency_over_budget_weight": 0.20,
    "brevity_weight": 0.08,
    "repetition_weight": 0.20,
    "anchor_quality_tolerance": 0.03,
    "anchor_min_candidates": 3,
    "anchor_quantile": 0.25,
    "brevity_slack_units": 2.0,
    "brevity_slack_ratio": 0.05,
    "brevity_denominator_floor": 20.0,
    "invalid_reward": -4.0,
}


def _number(value, name, minimum=0.0, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RewardError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise RewardError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise RewardError(f"{name} must be a finite number")
    if (minimum is not None and number < minimum) or (positive and number <= 0) \
            or (maximum is not None and number > maximum):
        raise RewardError(f"{name} is outside its permitted range")
    return number


def validate_v3_settings(settings):
    """Copy full reward settings, fill v3 defaults, and reject unsafe values.

Outer provider/service/version validation belongs to the reward dispatcher;
unrelated outer fields are preserved. Unknown *v3* fields are rejected so
misspelled reward controls cannot silently fall back to defaults.
"""
    if not isinstance(settings, dict):
        raise RewardError("v3 reward settings must be an object")
    result = copy.deepcopy(settings)
    result["quality_floor"] = _number(result.get("quality_floor", 0.8),
                                      "quality_floor", maximum=1, positive=True)
    result["latency_budget_seconds"] = _number(
        result.get("latency_budget_seconds", 3.0), "latency_budget_seconds", positive=True)
    supplied = result.get("v3", {})
    if not isinstance(supplied, dict):
        raise RewardError("reward.v3 must be an object")
    unknown = set(supplied) - set(DEFAULT_V3)
    if unknown:
        raise RewardError("Unsupported reward.v3 setting: " + ", ".join(sorted(unknown)))
    values = dict(DEFAULT_V3)
    values.update(supplied)
    for key in values:
        if key == "anchor_min_candidates":
            if type(values[key]) is not int or values[key] < 3:
                raise RewardError("reward.v3.anchor_min_candidates must be an integer >= 3")
            continue
        values[key] = _number(values[key], "reward.v3." + key,
                              minimum=None if key == "invalid_reward" else 0.0)
    for key in ("quality_gate_low", "quality_gate_high", "anchor_quality_tolerance",
                "anchor_quantile", "brevity_slack_ratio"):
        if values[key] > 1:
            raise RewardError(f"reward.v3.{key} must be <= 1")
    if not values["quality_gate_low"] < values["quality_gate_high"]:
        raise RewardError("reward.v3 quality_gate_low must be below quality_gate_high")
    if not values["quality_gate_low"] <= result["quality_floor"] <= values["quality_gate_high"]:
        raise RewardError("quality_floor must be within the v3 quality gate interval")
    for key in ("latency_cost_cap", "brevity_denominator_floor"):
        if values[key] <= 0:
            raise RewardError(f"reward.v3.{key} must be positive")
    width = values["quality_gate_high"] - values["quality_gate_low"]
    if values["brevity_weight"] > width + 1e-12:
        raise RewardError("v3 brevity_weight must not exceed gate width (quality monotonicity)")
    if values["invalid_reward"] != -4.0:
        raise RewardError("reward.v3.invalid_reward must remain -4")
    # Conservative bound for ANY Q in [0,1], nonnegative latency, C/P in [0,1].
    # Strict inequality keeps an invalid trajectory worse even at extremes.
    lower_bound = -(values["quality_deficit_weight"] * result["quality_floor"]
                    + values["latency_cost_cap"] + values["brevity_weight"]
                    + values["repetition_weight"])
    if not values["invalid_reward"] < lower_bound:
        raise RewardError("v3 penalties can make a valid reward no better than invalid_reward")
    result["v3"] = values
    return result


def _clip(value):
    return min(1.0, max(0.0, value))


def _latency_cost(latency, budget, values):
    """Bounded raw D, safely evaluated even for extremely large finite delay."""
    z = latency / budget
    cap = values["latency_cost_cap"]
    linear = values["latency_linear_weight"]
    quadratic = values["latency_over_budget_weight"]
    if not math.isfinite(z):
        return cap if linear or quadratic else 0.0
    cost = linear * z
    if cost >= cap:
        return cap
    over = max(0.0, z - 1.0)
    if quadratic:
        if over >= math.sqrt(cap / quadratic):
            return cap
        cost += quadratic * over * over
    return min(cap, cost)


def _combine(quality, latency, brevity, repetition, settings):
    quality = _number(quality, "quality", maximum=1)
    latency = _number(latency, "latency")
    brevity = _number(brevity, "brevity", maximum=1)
    repetition = _number(repetition, "repetition", maximum=1)
    values = settings["v3"]
    floor, budget = settings["quality_floor"], settings["latency_budget_seconds"]
    deficit = values["quality_deficit_weight"] * max(0.0, floor - quality)
    gate = _clip((quality - values["quality_gate_low"])
                 / (values["quality_gate_high"] - values["quality_gate_low"]))
    raw_cost = _latency_cost(latency, budget, values)
    latency_penalty = (1.0 - gate) * values["latency_cost_cap"] + gate * raw_cost
    weighted_brevity = values["brevity_weight"] * gate * brevity
    weighted_repetition = values["repetition_weight"] * repetition
    reward = quality - deficit - latency_penalty - weighted_brevity - weighted_repetition
    if not math.isfinite(reward) or reward <= values["invalid_reward"]:
        raise RewardError("v3 valid reward violated its finite/invalid-floor invariant")
    quality_pass, latency_pass = quality >= floor, latency <= budget
    return {
        "quality_base": quality,
        "quality_deficit_penalty": deficit,
        "quality_gate": gate,
        "latency_cost_raw": raw_cost,
        "latency_penalty": latency_penalty,
        "brevity_penalty": brevity,
        "repeat_penalty": repetition,
        "group_length_anchor": None,
        "anchor_pool_size": 0,
        "group_brevity_excess": brevity,
        "weighted_brevity_penalty": weighted_brevity,
        "weighted_repetition_penalty": weighted_repetition,
        "length_penalty": brevity,
        "reward": reward,
        "tier": "quality_fail" if not quality_pass else "latency_fail" if not latency_pass else "feasible",
        "quality_pass": quality_pass,
        "latency_pass": latency_pass,
        "feasible": quality_pass and latency_pass,
    }


def combine_v3(quality, latency, brevity, repetition, settings):
    """Compute one valid trajectory's unnormalised reward and log components.

``brevity_penalty``/``repeat_penalty`` are unweighted C/P; ``weighted_*``
fields are the actual deductions. ``latency_cost_raw`` is bounded D before
quality gating, while ``latency_penalty`` is the actual gated deduction.
    """
    return _combine(quality, latency, brevity, repetition, validate_v3_settings(settings))


def _quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def finalize_group(items, settings):
    """Return same-order copied records scored using one complete prompt group.

``eligible`` must already exclude invalid protocol, empty/incomplete output,
and severe loops. Missing/nonfinite scores on *eligible* records are upstream
failures and raise; they are never silently converted into policy penalties.
The provisional input ``reward`` is checked but never used to rank the pool.
    """
    settings = validate_v3_settings(settings)
    if not isinstance(items, (list, tuple)) or not items:
        raise RewardError("v3 finalize_group requires a nonempty complete prompt group")
    checked = []
    required = {"eligible", "quality", "latency", "units", "repetition", "reward"}
    for index, item in enumerate(items):
        if not isinstance(item, dict) or required - set(item):
            raise RewardError(f"v3 group item {index} is missing required fields")
        if type(item["eligible"]) is not bool:
            raise RewardError(f"v3 group item {index}.eligible must be boolean")
        if type(item["units"]) is not int or item["units"] < int(item["eligible"]):
            raise RewardError(f"v3 group item {index}.units must be a nonnegative integer; eligible needs >=1")
        row = dict(item)
        for key, maximum in (("quality", 1), ("latency", None)):
            if item[key] is None and not item["eligible"]:
                continue
            row[key] = _number(item[key], f"v3 group item {index}.{key}", maximum=maximum)
        row["repetition"] = _number(item["repetition"], f"v3 group item {index}.repetition", maximum=1)
        _number(item["reward"], f"v3 group item {index}.reward", minimum=None)
        checked.append(row)
    values = settings["v3"]
    eligible = [row for row in checked if row["eligible"]]
    threshold = max(settings["quality_floor"],
                    max(row["quality"] for row in eligible) - values["anchor_quality_tolerance"]) \
        if eligible else settings["quality_floor"]
    pool = [row["units"] for row in eligible if row["quality"] >= threshold]
    anchor = _quantile(pool, values["anchor_quantile"]) \
        if len(pool) >= values["anchor_min_candidates"] else None
    slack = max(values["brevity_slack_units"], values["brevity_slack_ratio"] * anchor) \
        if anchor is not None else None
    output = []
    for row in checked:
        if row["eligible"]:
            excess = _clip((row["units"] - anchor - slack)
                           / max(anchor, values["brevity_denominator_floor"])) \
                if anchor is not None else 0.0
            combined = _combine(row["quality"], row["latency"], excess, row["repetition"], settings)
        else:
            combined = {
                "quality_base": row["quality"], "quality_deficit_penalty": None,
                "quality_gate": 0.0, "latency_cost_raw": None, "latency_penalty": None,
                "brevity_penalty": 0.0, "repeat_penalty": row["repetition"],
                "group_brevity_excess": 0.0, "weighted_brevity_penalty": 0.0,
                "weighted_repetition_penalty": 0.0, "length_penalty": 0.0,
                "reward": values["invalid_reward"], "tier": "invalid",
                "quality_pass": False, "latency_pass": False, "feasible": False,
            }
        row.update(combined)
        row.update(group_length_anchor=anchor, anchor_pool_size=len(pool),
                   group_length_slack=slack, anchor_quality_threshold=threshold)
        output.append(row)
    return output
