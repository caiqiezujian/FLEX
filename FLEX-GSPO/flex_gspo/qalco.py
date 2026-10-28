"""QALCO: complete-trajectory quality/latency feasibility before efficiency.

Mean lag/LAAL are target-length scheduling proxies, not semantic latency.
v2 adds a named deterministic EOF text cost, never measured wall-clock time.
Length excess is not semantic redundancy. COMET22 (v2) and historical MetricX
(v1) are fallible proxies, not calibrated accuracy or independent fluency.
"""

from bisect import bisect_left
import math
import re

from .protocol import collect_translation, parse_response
from .rewards import RewardError
from .trajectory import validate_turn_schedule


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
    allowed = {"provider", "quality_floor", "version", "quality_provider", "comet22", "eof_delay_seconds",
               "latency_budget_seconds", "reference_length_ratio", "latency_weight",
               "length_weight", "latency_metric", "metricx", "v3", "v4"}
    if set(settings) - allowed:
        raise RewardError("Unsupported QALCO setting; allowed fields: " + ", ".join(sorted(allowed)))
    settings = dict(settings)
    settings.setdefault("version", 1)  # Historical audit configs keep their exact old formula.
    if type(settings["version"]) is not int or settings["version"] not in (1, 2, 3, 4):
        raise RewardError("reward.version must be 1 (historical audit), 2, 3, or 4 (component-scaled experiment)")
    if "v3" in settings and settings["version"] != 3:
        raise RewardError("reward.v3 is only valid with reward.version=3")
    if "v4" in settings and settings["version"] != 4:
        raise RewardError("reward.v4 is only valid with reward.version=4")
    if settings["version"] in (3, 4) and any(name in settings for name in
            ("reference_length_ratio", "latency_weight", "length_weight")):
        raise RewardError("QALCO v3/v4 does not use the old reference-length/tier weights; use versioned settings")
    settings.setdefault("quality_provider", "metricx" if settings["version"] == 1 else "comet22")
    expected_provider = "metricx" if settings["version"] == 1 else "comet22"
    if settings["quality_provider"] != expected_provider:
        raise RewardError(f"QALCO v{settings['version']} requires quality_provider={expected_provider}")
    settings["eof_delay_seconds"] = _number(settings.get("eof_delay_seconds", 0.5),
                                           "eof_delay_seconds", exclusive_minimum=True)
    # Online training defaults to the monotonic all-output-unit metric.
    # Historical speech-LAAL can be selected explicitly for comparisons only.
    settings.setdefault("latency_metric", "mean_lag")
    if settings["latency_metric"] not in ("mean_lag", "speech_laal"):
        raise RewardError("latency_metric must be mean_lag or speech_laal")
    if settings["version"] >= 2 and settings["latency_metric"] != "mean_lag":
        raise RewardError("QALCO v2/v3/v4 requires mean_lag; speech_laal is for historical comparison only")
    settings["quality_floor"] = _number(
        settings.get("quality_floor", 0.8 if settings["version"] in (3, 4) else 0.85), "quality_floor", maximum=1, exclusive_minimum=True)
    settings["latency_budget_seconds"] = _number(
        settings.get("latency_budget_seconds", 3.0), "latency_budget_seconds", exclusive_minimum=True)
    if settings["version"] not in (3, 4):
        settings["reference_length_ratio"] = _number(
            settings.get("reference_length_ratio", 1.2), "reference_length_ratio", minimum=1)
        for name, default in (("latency_weight", 0.2), ("length_weight", 0.1)):
            settings[name] = _number(settings.get(name, default), name)
    elif settings["version"] == 3:
        from .gated_reward import validate_v3_settings
        settings = validate_v3_settings(settings)
    else:
        from .normalized_reward import validate_v4_settings
        settings = validate_v4_settings(settings)
    if settings["quality_provider"] == "metricx" or "metricx" in settings:
        from .metricx import validate_metricx_settings
        settings["metricx"] = validate_metricx_settings(settings.get("metricx"))
    if settings["quality_provider"] == "comet22":
        from .comet22 import validate_comet22_settings
        settings["comet22"] = validate_comet22_settings(settings.get("comet22", {}))
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


def emission_schedule(task, turns, eof_delay_seconds=0.0):
    """Assign units to their completion turn; EOF may carry an explicit proxy cost."""
    duration = float(task["duration_seconds"])
    pieces, endpoints, times = [], [], []
    offset, previous = 0, 0.0
    for index, turn in enumerate(turns):
        time = _number(turn.get("audio_end_seconds"), "turn.audio_end_seconds")
        if time < previous or time > duration + TIME_TOLERANCE:
            raise RewardError("invalid/nonmonotonic trajectory timing; this is not a model action error")
        is_eof = turn.get("kind") == "eof"
        if is_eof:
            if index != len(task["chunks"]) or index != len(turns) - 1:
                raise RewardError("EOF must occur exactly once after all audio turns")
            expected = duration
        else:
            if index >= len(task["chunks"]):
                raise RewardError("Extra audio turn outside the chunk schedule")
            expected = float(task["chunks"][index]["end_seconds"])
        if abs(time - expected) > TIME_TOLERANCE:
            raise RewardError("trajectory timing disagrees with task chunk schedule")
        previous = time
        action, delta = parse_response(turn["response"])
        if action != "WAIT":
            pieces.append(delta)
            offset += len(delta)
            endpoints.append(offset)
            times.append(min(time, duration) + (eof_delay_seconds if is_eof else 0.0))
    text = "".join(pieces)
    spans = text_units(text)
    completed_at = [times[bisect_left(endpoints, span.end())] for span in spans]
    return text, completed_at


def latency_statistics(duration, emission_times, reference_units, eof_delay_seconds=0.0):
    """Official speech-LAAL convention, g(t) - (t-1)*S/max(N,M).

    Tau ends at the FIRST unit emitted when the source has been fully consumed;
    the rest of that EOF batch is excluded from LAAL, and separately reported.
    """
    duration = _number(duration, "duration_seconds", exclusive_minimum=True)
    eof_delay_seconds = _number(eof_delay_seconds, "eof_delay_seconds")
    if type(reference_units) is not int or reference_units < 1:
        raise RewardError("reference_units must be a positive integer")
    if not emission_times:
        raise RewardError("empty output has no defined LAAL")
    previous = 0.0
    for time in emission_times:
        _number(time, "emission_time")
        if time < previous or time > duration + eof_delay_seconds + TIME_TOLERANCE:
            raise RewardError("invalid emission schedule")
        previous = time
    times = [min(float(t), duration + eof_delay_seconds) for t in emission_times]
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
    gaps = [b - a for a, b in zip([0.0] + times, times + [max(duration, times[-1])])]
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
    if settings.get("version") in (3, 4):
        raise RewardError("QALCO v3/v4 must use complete-group finalization, not the historical tier formula")
    quality = _number(quality, "quality", maximum=1)
    latency = _number(latency, "latency")
    length_excess = _number(length_excess, "length_penalty", maximum=1)
    floor = _number(settings["quality_floor"], "quality_floor", maximum=1, exclusive_minimum=True)
    budget = _number(settings["latency_budget_seconds"], "latency_budget_seconds", exclusive_minimum=True)
    latency_weight = _number(settings["latency_weight"], "latency_weight")
    length_weight = _number(settings["length_weight"], "length_weight")
    quality_pass, latency_pass = quality >= floor, latency <= budget
    if settings.get("version", 1) == 2 and not quality_pass:
        # Keep strict quality priority, but do not make waiting free below the floor.
        deficit = (floor - quality) / floor
        lag_cost = latency / (budget + latency)
        reward = -2.0 - (deficit + latency_weight * lag_cost + length_weight * length_excess) / (1 + latency_weight + length_weight)
        tier = "quality_fail"
    elif settings.get("version", 1) == 2 and not latency_pass:
        # Smooth non-saturating lateness, retaining quality/length discrimination.
        over = latency - budget
        reward = -1.0 - (over / (budget + over) + (1 - quality) + length_weight * length_excess) / (2 + length_weight)
        tier = "latency_fail"
    elif not quality_pass:
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
            if self.settings["quality_provider"] == "comet22":
                from .comet22 import ServiceCOMET22
                quality_model = ServiceCOMET22(self.settings["comet22"])
            else:
                from .metricx import build_metricx
                quality_model = build_metricx(self.settings["metricx"])
        self.quality_model = quality_model

    def score(self, task, trajectory):
        return self.score_group(task, [trajectory])[0]

    def score_group(self, task, trajectories):
        validate_task(task)
        records, pending, hypotheses = [None] * len(trajectories), [], []
        # v2/v3/v4 share the accepted EOF and protocol contract; group versions change
        # the objective and repetition checks, not the actor observation.
        v2 = self.settings["version"] >= 2
        group_reward = self.settings["version"] in (3, 4)
        eof_cost = self.settings["eof_delay_seconds"] if v2 else 0.0
        reference_units = len(text_units(task["reference_text"]))
        for index, trajectory in enumerate(trajectories):
            if trajectory.get("mock", False):
                raise RewardError("mock trajectories cannot receive real training rewards")
            if v2:
                if type(trajectory.get("schema_version")) is not int or trajectory["schema_version"] != 2:
                    raise RewardError("QALCO v2 requires scheduler trajectory schema_version=2; update the whole project")
                schedule = validate_turn_schedule(task, trajectory.get("turns"))
                for flag in ("complete_input", "episode_complete", "eof_completed"):
                    if type(trajectory.get(flag)) is not bool or trajectory[flag] != schedule[flag]:
                        raise RewardError(f"QALCO v2 trajectory {flag} disagrees with audio/EOF schedule")
            if not trajectory.get("valid", True):
                records[index] = _invalid(trajectory.get("error") or "invalid trajectory")
                continue
            turns = trajectory.get("turns")
            try:
                if not isinstance(turns, list) or len(turns) != len(task["chunks"]) + int(v2):
                    raise ValueError("incomplete episode")
                if v2 and (trajectory.get("episode_complete") is not True or trajectory.get("eof_completed") is not True
                           or turns[-1].get("kind") != "eof"):
                    raise ValueError("missing final EOF response")
                if trajectory.get("complete_input", True) is not True:
                    raise ValueError("incomplete input")
                if any(turn.get("finish_reason") not in ("stop", "eos") for turn in turns):
                    raise ValueError("truncated or abnormal generation")
                translation = collect_translation(turns)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                records[index] = _invalid(exc)
                continue
            # Stored timestamps are trusted infrastructure, not generated model tokens.
            validate_turn_schedule(task, turns)
            translation, times = emission_schedule(task, turns, eof_cost)
            if not translation.strip() or not times:
                records[index] = _invalid("no measurable translation", empty=True)
                continue
            statistics = latency_statistics(task["duration_seconds"], times, reference_units, eof_cost)
            excess = 0.0 if group_reward else length_penalty(len(times), reference_units, self.settings["reference_length_ratio"])
            guard = {}
            if v2:
                if group_reward:
                    from .degeneration_v3 import degeneration_report_v3 as degeneration_report
                else:
                    from .degeneration import degeneration_report
                guard = degeneration_report(task["source_text"], task["reference_text"], turns)
                if guard["severe_repetition"]:
                    records[index] = _invalid("severe cyclic repetition")
                    records[index]["components"].update({"protocol_valid": True, "tier": "degenerate",
                        "severe_repetition": True, "repetition_diagnostics": guard, **statistics,
                        "length_penalty": excess, "quality_metric": "comet22_reference"})
                    continue
                statistics["diagnostics"].update({
                    "eof_generation_seconds": turns[-1].get("eof_elapsed_seconds"),
                    "eof_delay_proxy_seconds": eof_cost,
                    "eof_turn_text": parse_response(turns[-1]["response"])[0] != "WAIT",
                })
            pending.append((index, statistics, excess, guard))
            hypotheses.append(translation)
        if pending:
            # Identical complete translations share one frozen quality evaluation.
            unique = list(dict.fromkeys(hypotheses))
            raw_scores = self.quality_model.score_batch(
                task["source_text"], unique, task["reference_text"])
            if not isinstance(raw_scores, list) or len(raw_scores) != len(unique):
                raise RewardError("quality scorer batch-size mismatch")
            lookup = {}
            for text, raw in zip(unique, raw_scores):
                lookup[text] = _number(raw, "comet22_score" if v2 else "metricx_error", maximum=1 if v2 else 25)
            for (index, statistics, excess, guard), translation in zip(pending, hypotheses):
                raw = lookup[translation]
                quality = raw if v2 else 1 - raw / 25
                use_mean_lag = self.settings["latency_metric"] == "mean_lag"
                latency = statistics["mean_lag_seconds" if use_mean_lag else "laal_seconds"]
                if self.settings["version"] == 4:
                    from .normalized_reward import provisional_v4
                    combined = provisional_v4(quality, latency, guard["repetition_penalty"], self.settings)
                elif group_reward:
                    from .gated_reward import combine_v3
                    # This is provisional: the distributed trainer boundary
                    # will compute C from the actual full GSPO group. Never
                    # normalize/group candidates in a rank-local ORM callback.
                    combined = combine_v3(quality, latency, 0.0, guard["repetition_penalty"], self.settings)
                else:
                    combined = combine_reward(quality, latency, excess, self.settings)
                reward = combined.pop("reward")
                records[index] = {"reward": reward, "components": {
                    "protocol_valid": True, "empty_translation": False,
                    "quality": quality, "metricx_error": None if v2 else raw, **statistics,
                    "comet22_score": raw if v2 else None,
                    "reward_version": self.settings["version"],
                    "severe_repetition": False, "repetition_diagnostics": guard,
                    "length_penalty": excess, **combined,
                    "latency_seconds": latency,
                    "latency_metric": (("source_clock_mean_positive_lag_plus_eof_proxy" if v2 else "source_clock_mean_positive_lag") if use_mean_lag
                                       else "utterance_speech_laal"),
                    "quality_metric": "comet22_reference" if v2 else "metricx24_" + self.settings["metricx"]["mode"],
                    "prefix_support_checked": False,
                }}
        if group_reward:
            for record in records:
                components = record["components"]
                eligible = components["tier"] not in ("invalid", "degenerate")
                components.update({"reward_version": self.settings["version"], "eligible": eligible,
                    "group_finalized": False,
                    "emitted_units": (components.get("diagnostics") or {}).get("emitted_units", 0),
                    "repetition_penalty": (components.get("repetition_diagnostics") or {}).get("repetition_penalty", 0.0)})
        return records
