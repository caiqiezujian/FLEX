"""Conservative exact-loop detection, independent of learned quality metrics.

This is NOT a semantic coverage/redundancy metric. It only catches dominant,
contiguous repetitions of a clause-like phrase. Ordinary reused FLEX anchors,
sentence punctuation, repeated entities and faithful source repetitions must
not automatically become failed translations. Ambiguous cases stay diagnostic.

No text is changed, discarded or sent to a service by this module. All token
normalization below is for detection only; the actor's actual output is kept.
"""
from __future__ import annotations

import re
import unicodedata

from .protocol import collect_translation


_UNITS = re.compile(
    r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\u3040-\u30ff]"
    r"|[A-Za-z\u00c0-\u024f]+(?:['’\-][A-Za-z\u00c0-\u024f]+)*"
    r"|\d+(?:[.,]\d+)*")
_HAN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")
_EN_SUBJECT = {"i", "you", "he", "she", "it", "we", "they", "this", "that", "there"}
_EN_VERB = {
    "am", "is", "are", "was", "were", "be", "have", "has", "had", "do", "does",
    "did", "like", "likes", "love", "want", "need", "can", "will", "must", "should",
}
_ZH_CLAUSE = re.compile(
    r"(?:我|你|您|他|她|它|这|那|我们|他们).*(?:是|有|在|不|没|会|能|要|让|喜欢|需要|认为)"
    r"|(?:因此|所以|因为|但是|然而|已经|正在|可以|不能|没有|这是|那是)")


def _tokens(text):
    if not isinstance(text, str):
        raise TypeError("Repetition guard text inputs must be strings")
    return _UNITS.findall(unicodedata.normalize("NFKC", text).casefold())


def _primitive_period(phrase):
    """Do not let e.g. a repeated name/word masquerade as a four-unit phrase."""
    size = len(phrase)
    for period in range(1, size):
        if size % period == 0 and phrase == phrase[:period] * (size // period):
            return period
    return size


def _clause_like(phrase):
    joined = "".join(phrase)
    if _HAN.search(joined):
        return bool(_ZH_CLAUSE.search(joined))
    words = set(phrase)
    return bool(words & _EN_SUBJECT and words & _EN_VERB)


def _runs(tokens, minimum_period, minimum_repetitions=4):
    """Find exact adjacent loops with bounded phrase length; no quadratic matrix."""
    total = len(tokens)
    for period in range(minimum_period, min(64, total // minimum_repetitions) + 1):
        start = 0
        while start + minimum_repetitions * period <= total:
            # Most prose cannot be a loop at this position. Reject via scalar
            # comparisons before allocating phrase/four-repeat slices.
            if (tokens[start] != tokens[start + period]
                    or tokens[start] != tokens[start + 2 * period]
                    or tokens[start] != tokens[start + 3 * period]):
                start += 1
                continue
            phrase = tokens[start:start + period]
            if (tokens[start:start + minimum_repetitions * period]
                    != phrase * minimum_repetitions):
                start += 1
                continue
            stop = start + minimum_repetitions * period
            while tokens[stop:stop + period] == phrase:
                stop += period
            yield phrase, (stop - start) // period, (stop - start) / total
            start = stop


def _supported_by_reference(phrase, repetitions, reference):
    # Require comparable repetition, not merely that the phrase appears once.
    # Matching two occurrences cannot excuse a thirteen-copy output loop.
    required = max(4, repetitions - 1)
    target = phrase * required
    return any(reference[start:start + len(target)] == target
               for start in range(max(0, len(reference) - len(target) + 1)))


def degeneration_report(source_text, reference_text, turns):
    """Report a narrow high-confidence severe-loop condition.

    A hard flag needs a primitive phrase >=4 units, >=4 adjacent repetitions,
    >=60% of all output units, and a clause-like rather than name-only phrase.
    Punctuation, whitespace, WAIT turns and output chunk boundaries do not
    interrupt detection. Real source/reference repetition is conservatively
    exempted. Cross-lingual source evidence is structural, NOT semantic proof.

    Invalid protocol raises ValueError: callers should validate the trajectory
    first. A negative result does not certify that the translation is good.
    """
    source, reference = _tokens(source_text), _tokens(reference_text)
    output = _tokens(collect_translation(turns))
    report = {
        "severe_repetition": False,
        "repeated_phrase": None,
        "repetitions": 0,
        "coverage_fraction": 0.0,
        "output_units": len(output),
        "exemption_reason": None,
        "guard": "dominant_exact_clause_loop_v1",
        "semantic_redundancy_checked": False,
    }
    strongest = None
    for phrase, count, coverage in _runs(output, 4):
        if _primitive_period(phrase) != len(phrase):
            continue
        if strongest is None or (coverage, count) > (strongest[2], strongest[1]):
            strongest = phrase, count, coverage
        if coverage < 0.6 or not _clause_like(phrase):
            continue
        exemption = None
        if _supported_by_reference(phrase, count, reference):
            exemption = "comparable_exact_repetition_in_reference"
        else:
            # The source language differs, so literal source-target overlap is
            # not required. A similarly dominant, similarly repeated SOURCE
            # loop may faithfully yield a repeated target phrase; retain it.
            for _, source_count, source_coverage in _runs(source, 1, max(4, count - 1)):
                if source_count >= count - 1 and source_coverage >= 0.6:
                    exemption = "comparable_dominant_repetition_in_source"
                    break
        details = {
            "repeated_phrase": "".join(phrase) if _HAN.search("".join(phrase)) else " ".join(phrase),
            "repetitions": count,
            "coverage_fraction": coverage,
            "exemption_reason": exemption,
        }
        if exemption is None:
            return {**report, **details, "severe_repetition": True}
        report.update(details)
    if report["repeated_phrase"] is None and strongest is not None:
        phrase, count, coverage = strongest
        report.update({
            "repeated_phrase": "".join(phrase) if _HAN.search("".join(phrase)) else " ".join(phrase),
            "repetitions": count,
            "coverage_fraction": coverage,
            "exemption_reason": ("not_clause_like" if not _clause_like(phrase)
                                 else "loop_not_dominant"),
        })
    return report
