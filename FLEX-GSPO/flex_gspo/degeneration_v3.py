"""Conservative exact adjacent-loop diagnostics for the v3 reward.

This detects surface repetition, not semantic hallucination or redundancy.
Normalization is used only by the detector. The actor's emitted translation
and turns are never rewritten. The v2 detector remains unchanged.
"""
from __future__ import annotations

import re

from .degeneration import _HAN, _clause_like, _primitive_period, _supported_by_reference, _tokens
from .protocol import collect_translation


MAX_PERIOD = 64
MIN_REPETITIONS = 4
_ZH_GRAMMAR = re.compile(
    r"不是|没有|正在|已经|可以|不能|需要|应该|必须|因为|所以|但是|然而|如果|虽然|为了|由于"
    r"|(?:我|你|您|他|她|它|这|那|我们|他们).*(?:是|有|在|不|没|会|能|要|让)"
    r"|(?:在|把|被|让|给|将|能|会|要).+"
)
_EN_SUBJECT = {"i", "you", "he", "she", "it", "we", "they", "this", "that", "there"}
_EN_AUXILIARY = {"am", "is", "are", "was", "were", "be", "have", "has", "had",
                 "do", "does", "did", "can", "could", "will", "would", "must", "should"}
_EN_FUNCTION = {"a", "an", "the", "of", "for", "to", "with", "and", "in", "on", "at", "from"}
_ENTITY_ENDINGS = ("公司", "集团", "大学", "学院", "研究所", "研究院", "委员会",
                   "联合会", "协会", "基金会", "中心", "实验室", "银行", "政府")


def _runs(tokens, minimum_period=4):
    """Yield primitive runs as (phrase, start, stop, count), bounded period."""
    total = len(tokens)
    for period in range(minimum_period, min(MAX_PERIOD, total // MIN_REPETITIONS) + 1):
        start = 0
        while start + MIN_REPETITIONS * period <= total:
            if any(tokens[start] != tokens[start + copy * period] for copy in (1, 2, 3)):
                start += 1
                continue
            phrase = tokens[start:start + period]
            if (tokens[start:start + MIN_REPETITIONS * period] != phrase * MIN_REPETITIONS
                    or _primitive_period(phrase) != period):
                start += 1
                continue
            stop = start + MIN_REPETITIONS * period
            while tokens[stop:stop + period] == phrase:
                stop += period
            yield phrase, start, stop, (stop - start) // period
            start = stop


def _phrase_evidence(phrase):
    joined, words = "".join(phrase), set(phrase)
    han_count = sum(bool(_HAN.fullmatch(unit)) for unit in phrase)
    if han_count:
        clause_like = _clause_like(phrase) or bool(_ZH_GRAMMAR.search(joined))
        # A sufficiently long independent phrase may receive a soft cost even
        # without a recognized predicate. Obvious organization names stay out.
        long_phrase = (len(phrase) >= 10 and len(words) >= 5
                       and not joined.endswith(_ENTITY_ENDINGS))
    else:
        clause_like = _clause_like(phrase) or bool(words & _EN_SUBJECT and words & _EN_AUXILIARY)
        long_phrase = (len(phrase) >= 8 and len(words) >= 5
                       and sum(unit in _EN_FUNCTION for unit in phrase) >= 2)
    return clause_like, clause_like or long_phrase


def _mark(mask, start, stop):
    mask[start:stop] = b"\x01" * (stop - start)


def degeneration_report_v3(source, reference, turns):
    """Return severe-loop flag and a bounded surface-repetition cost.

    Eligible loops have a primitive period of 4..64 normalized units and at
    least four exact adjacent copies. ``repetition_penalty`` is the UNION of
    units from the fourth copy onward divided by all output units. The severe
    flag instead uses the UNION of full clause-like loop spans at >=60%.

    Comparable exact reference repetition is exempted. Comparable source
    repetition is conservatively exempted by its structural count/coverage;
    this does not claim any cross-language lexical or semantic alignment.
    Ordinary anchors, short entities, punctuation and three-copy parallelism
    do not trigger a penalty. Invalid output protocol raises ValueError.
    """
    source_units, reference_units = _tokens(source), _tokens(reference)
    output_units = _tokens(collect_translation(turns))
    total = len(output_units)
    penalty_mask, severe_mask = bytearray(total), bytearray(total)
    source_runs = list(_runs(source_units, minimum_period=1))
    details, exemption_reasons = [], {}
    detected = eligible = exempted = 0
    for phrase, start, stop, copies in _runs(output_units):
        detected += 1
        period = len(phrase)
        coverage = (stop - start) / total
        clause_like, soft_eligible = _phrase_evidence(phrase)
        exemption = None
        source_evidence = None
        if _supported_by_reference(phrase, copies, reference_units):
            exemption = "comparable_exact_repetition_in_reference"
        else:
            for source_phrase, source_start, source_stop, source_copies in source_runs:
                source_coverage = (source_stop-source_start) / len(source_units)
                if (source_copies >= max(MIN_REPETITIONS, copies-1)
                        and source_coverage >= min(.6, coverage)):
                    exemption = "comparable_structural_repetition_in_source"
                    source_evidence = {"period_units": len(source_phrase), "repetitions": source_copies,
                                       "coverage_fraction": source_coverage,
                                       "cross_language_semantic_match_verified": False}
                    break
        if exemption:
            exempted += 1
            exemption_reasons[exemption] = exemption_reasons.get(exemption, 0) + 1
        elif soft_eligible:
            eligible += 1
            _mark(penalty_mask, start + 3*period, stop)
            if clause_like:
                _mark(severe_mask, start, stop)
        else:
            exemption = "short_entity_or_fragment_without_clause_evidence"
            exemption_reasons[exemption] = exemption_reasons.get(exemption, 0) + 1
        detail = {"repeated_phrase": "".join(phrase) if _HAN.search("".join(phrase)) else " ".join(phrase),
                  "period_units": period, "repetitions": copies, "start_unit": start,
                  "stop_unit": stop, "coverage_fraction": coverage, "clause_like": clause_like,
                  "soft_eligible": soft_eligible, "exemption_reason": exemption}
        if source_evidence:
            detail["source_evidence"] = source_evidence
        details.append(detail)
    coverage_units, extra_units = sum(severe_mask), sum(penalty_mask)
    details.sort(key=lambda item: (item["coverage_fraction"], item["repetitions"]), reverse=True)
    strongest = details[0] if details else {}
    return {
        "severe_repetition": bool(total and coverage_units / total >= .6),
        "repetition_penalty": extra_units / total if total else 0.0,
        "guard": "exact_adjacent_clause_loop_v3",
        "semantic_redundancy_checked": False,
        "semantic_hallucination_checked": False,
        "output_units": total,
        "repeated_excess_units": extra_units,
        "clause_loop_coverage_units": coverage_units,
        "coverage_fraction": coverage_units / total if total else 0.0,
        "detected_runs": detected, "eligible_runs": eligible, "exempted_runs": exempted,
        "exemption_reasons": exemption_reasons,
        "exemption_reason": strongest.get("exemption_reason"),
        "repeated_phrase": strongest.get("repeated_phrase"),
        "repetitions": strongest.get("repetitions", 0),
        "runs": details[:16], "run_details_truncated": len(details) > 16,
        "minimum_primitive_period": 4, "maximum_period": MAX_PERIOD,
        "minimum_repetitions": MIN_REPETITIONS,
    }
