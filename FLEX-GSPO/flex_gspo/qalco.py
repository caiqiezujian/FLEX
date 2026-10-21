"""QALCO: complete-trajectory quality/latency feasibility before efficiency.

Both mean lag and LAAL are source-clock, target-length scheduling proxies, not
semantic latency or wall-clock inference delay. Length excess is not semantic redundancy. MetricX
is a fallible quality proxy, not calibrated accuracy or independent fluency.
"""

from bisect import bisect_left
import math
import re

from .protocol import collect_translation, parse_response
from .rewards import RewardError


UNIT_PATTERN = re.compile(
    r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\u3040-\u30ff]"
    r"|[A-Za-z\u00c0-\u024f]+(?:['’\-][A-Za-z\u00c0-\u024f]+)*"
    r"|\d+(?:[.,]\d+)*")
TIME_TOLERANCE = 1 / 16000 + 1e-6


def _number(value, name, minimum=0, maximum=None, exclusive_minimum=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RewardError(f"{name} must be a finite number")
    if value < minimum or (exclusive_minimum and value == minimum) \
            or (maximum is not None and value > maximum):
        raise RewardError(f"{name} is outside its permitted range")
    return float(value)


def validate_settings(config):
    if not isinstance(config, dict):
        raise RewardError("reward configuration must be an object")
    settings = config.get("reward", config)
    if not isinstance(settings, dict) or settings.get("provider") != "qalco":
        raise RewardError("QALCO v1 requires reward.provider=qalco")
    allowed = {"provider", "quality_floor",
               "latency_budget_seconds", "reference_length_ratio", "latency_weight",
               "length_weight", "latency_metric", "metricx"}
    if set(settings) - allowed:
        raise RewardError("Unsupported QALCO setting; allowed fields: " + ", ".join(sorted(allowed)))
    settings = dict(settings)
    # Online training defaults to the monotonic all-output-unit metric.
    # Historical speech-LAAL can be selected explicitly for comparisons only.
    settings.setdefault("latency_metric", "mean_lag")
    if settings["latency_metric"] not in ("mean_lag", "speech_laal"):
        raise RewardError("latency_metric must be mean_lag or speech_laal")
    settings["quality_floor"] = _number(
        settings.get("quality_floor", 0.85), "quality_floor", maximum=1, exclusive_minimum=True)
    settings["latency_budget_seconds"] = _number(
        settings.get("latency_budget_seconds", 3.0), "latency_budget_seconds", exclusive_minimum=True)
    settings["reference_length_ratio"] = _number(
        settings.get("reference_length_ratio", 1.2), "reference_length_ratio", minimum=1)
    for name, default in (("latency_weight", 0.2), ("length_weight", 0.1)):
        settings[name] = _number(settings.get(name, default), name)
    from .metricx import validate_metricx_settings
    settings["metricx"] = validate_metricx_settings(settings.get("metricx"))
    return settings


def text_units(text):
    return list(UNIT_PATTERN.finditer(text))


def validate_task(task):
    for name in ("source_text", "reference_text"):
        if not isinstance(task.get(name), str) or not task[name].strip():
            raise RewardError(f"{task.get('id')}: {name} is required for QALCO v1")
    if not text_units(task["reference_text"]):
        raise RewardError(f"{task.get('id')}: reference has no measurable text units")
    if task.get("source_lang") != "en" or task.get("target_lang") != "zh":
        raise RewardError("QALCO v1 is calibrated/configured only for en->zh; migration needs a new audit")
    duration = _number(task.get("duration_seconds"), "duration_seconds", exclusive_minimum=True)
    chunks = task.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise RewardError("QALCO task requires nonempty chronological chunks")
    previous = 0.0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise RewardError("task chunk must be an object")
        end = _number(chunk.get("end_seconds"), "chunk.end_seconds", exclusive_minimum=True)
        if end <= previous or end > duration + TIME_TOLERANCE:
            raise RewardError("task chunks must be strictly increasing within source duration")
        previous = end
    if abs(previous - duration) > TIME_TOLERANCE:
        raise RewardError("task chunk schedule does not cover the complete audio")


def validate_tasks(tasks, config):
    settings = validate_settings(config)
    for task in tasks:
        validate_task(task)
    return settings


def emission_schedule(task, turns):
    """Assign a whole unit to the turn completing it, even across split deltas."""
    duration = float(task["duration_seconds"])
    pieces, endpoints, times = [], [], []
    offset, previous = 0, 0.0
    for index, turn in enumerate(turns):
        time = _number(turn.get("audio_end_seconds"), "turn.audio_end_seconds")
        if time < previous or time > duration + TIME_TOLERANCE:
            raise RewardError("invalid/nonmonotonic trajectory timing; this is not a model action error")
        expected = float(task["chunks"][index]["end_seconds"])
        if abs(time - expected) > TIME_TOLERANCE:
            raise RewardError("trajectory timing disagrees with task chunk schedule")
        previous = time
        action, delta = parse_response(turn["response"])
        if action != "WAIT":
            pieces.append(delta)
            offset += len(delta)
            endpoints.append(offset)
            times.append(min(time, duration))
    text = "".join(pieces)
    spans = text_units(text)
    completed_at = [times[bisect_left(endpoints, span.end())] for span in spans]
    return text, completed_at


def latency_statistics(duration, emission_times, reference_units):
    """Official speech-LAAL convention, g(t) - (t-1)*S/max(N,M).

    Tau ends at the FIRST unit emitted when the source has been fully consumed;
    the rest of that EOF batch is excluded from LAAL, and separately reported.
    """
    duration = _number(duration, "duration_seconds", exclusive_minimum=True)
    if type(reference_units) is not int or reference_units < 1:
        raise RewardError("reference_units must be a positive integer")
    if not emission_times:
        raise RewardError("empty output has no defined LAAL")
    previous = 0.0
    for time in emission_times:
        _number(time, "emission_time")
        if time < previous or time > duration + TIME_TOLERANCE:
            raise RewardError("invalid emission schedule")
        previous = time
    times = [min(float(t), duration) for t in emission_times]
    count = len(times)
    tau = next((i + 1 for i, time in enumerate(times)
                if time >= duration - TIME_TOLERANCE), count)
    denominator = max(count, reference_units)
    raw = sum(time - i * duration / denominator for i, time in enumerate(times[:tau])) / tau
    # Unlike tau-truncated LAAL, use ALL output units and a fixed denominator
    # for a fixed translation. Delaying any unit cannot reduce this metric.
    # Positive parts prevent early output from cancelling later positive lag.
    mean_lag = sum(max(0.0, time - i * duration / denominator)
                   for i, time in enumerate(times)) / count
    gaps = [b - a for a, b in zip([0.0] + times, times + [duration])]
    return {
        "raw_laal_seconds": raw,
        "laal_seconds": max(0.0, raw),
        "mean_lag_seconds": mean_lag,
        "diagnostics": {
            "first_output_seconds": times[0],
            "max_no_text_gap_seconds": max(gaps),
            "eof_text_fraction": sum(t >= duration - TIME_TOLERANCE for t in times) / count,
            "emitted_units": count,
            "reference_units": reference_units,
            "laal_tau": tau,
        },
    }


def length_penalty(candidate_units, reference_units, ratio):
    if candidate_units < 0 or reference_units <= 0:
        raise RewardError("invalid text unit counts for length penalty")
    return min(1.0, max(0.0, candidate_units / (ratio * reference_units) - 1.0))


def combine_reward(quality, latency, length_excess, settings):
    quality = _number(quality, "quality", maximum=1)
    latency = _number(latency, "latency")
    length_excess = _number(length_excess, "length_penalty", maximum=1)
    floor = _number(settings["quality_floor"], "quality_floor", maximum=1, exclusive_minimum=True)
    budget = _number(settings["latency_budget_seconds"], "latency_budget_seconds", exclusive_minimum=True)
    latency_weight = _number(settings["latency_weight"], "latency_weight")
    length_weight = _number(settings["length_weight"], "length_weight")
    quality_pass, latency_pass = quality >= floor, latency <= budget
    if not quality_pass:
        reward, tier = -2.0 - min(1.0, (floor - quality) / floor), "quality_fail"
    elif not latency_pass:
        reward, tier = -1.0 - min(1.0, (latency - budget) / budget), "latency_fail"
    else:
        reward = (quality + latency_weight * (1 - latency / budget)
                  + length_weight * (1 - length_excess)) / (1 + latency_weight + length_weight)
        tier = "feasible"
    return {"reward": reward, "tier": tier, "quality_pass": quality_pass,
            "latency_pass": latency_pass, "feasible": quality_pass and latency_pass}


def _invalid(error, empty=False):
    return {"reward": -4.0, "components": {
        "protocol_valid": empty, "empty_translation": empty,
        "quality": None, "metricx_error": None, "laal_seconds": None,
        "mean_lag_seconds": None, "latency_seconds": None,
        "raw_laal_seconds": None, "length_penalty": None, "tier": "invalid",
        "quality_pass": False, "latency_pass": False, "feasible": False,
        "error": str(error),
    }}


class QalcoScorer:
    def __init__(self, config, quality_model=None):
        self.settings = validate_settings(config)
        # Test injection is explicit; production build_scorer has no mock fallback.
        if quality_model is None:
            from .metricx import build_metricx
            quality_model = build_metricx(self.settings["metricx"])
        self.quality_model = quality_model

    def score(self, task, trajectory):
        return self.score_group(task, [trajectory])[0]

    def score_group(self, task, trajectories):
        validate_task(task)
        records, pending, hypotheses = [None] * len(trajectories), [], []
        reference_units = len(text_units(task["reference_text"]))
        for index, trajectory in enumerate(trajectories):
            if trajectory.get("mock", False):
                raise RewardError("mock trajectories cannot receive real training rewards")
            if not trajectory.get("valid", True):
                records[index] = _invalid(trajectory.get("error") or "invalid trajectory")
                continue
            turns = trajectory.get("turns")
            try:
                if not isinstance(turns, list) or len(turns) != len(task["chunks"]):
                    raise ValueError("incomplete episode")
                if trajectory.get("complete_input", True) is not True:
                    raise ValueError("incomplete input")
                if any(turn.get("finish_reason") not in ("stop", "eos") for turn in turns):
                    raise ValueError("truncated or abnormal generation")
                translation = collect_translation(turns)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                records[index] = _invalid(exc)
                continue
            # Stored timestamps are trusted infrastructure, not generated model tokens.
            translation, times = emission_schedule(task, turns)
            if not translation.strip() or not times:
                records[index] = _invalid("no measurable translation", empty=True)
                continue
            statistics = latency_statistics(task["duration_seconds"], times, reference_units)
            excess = length_penalty(len(times), reference_units, self.settings["reference_length_ratio"])
            pending.append((index, statistics, excess))
            hypotheses.append(translation)
        if pending:
            # Identical complete translations share one frozen quality evaluation.
            unique = list(dict.fromkeys(hypotheses))
            raw_scores = self.quality_model.score_batch(
                task["source_text"], unique, task["reference_text"])
            if len(raw_scores) != len(unique):
                raise RewardError("quality scorer batch-size mismatch")
            lookup = {}
            for text, raw in zip(unique, raw_scores):
                lookup[text] = _number(raw, "metricx_error", maximum=25)
            for (index, statistics, excess), translation in zip(pending, hypotheses):
                raw = lookup[translation]
                quality = 1 - raw / 25
                use_mean_lag = self.settings["latency_metric"] == "mean_lag"
                latency = statistics["mean_lag_seconds" if use_mean_lag else "laal_seconds"]
                combined = combine_reward(quality, latency, excess, self.settings)
                reward = combined.pop("reward")
                records[index] = {"reward": reward, "components": {
                    "protocol_valid": True, "empty_translation": False,
                    "quality": quality, "metricx_error": raw, **statistics,
                    "length_penalty": excess, **combined,
                    "latency_seconds": latency,
                    "latency_metric": ("source_clock_mean_positive_lag" if use_mean_lag
                                       else "utterance_speech_laal"),
                    "quality_metric": "metricx24_" + self.settings["metricx"]["mode"],
                    "prefix_support_checked": False,
                }}
        return records
