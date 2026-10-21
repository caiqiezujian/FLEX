"""Protocol checks without model/GPU dependencies. Text is never silently repaired."""

TAGS = ("<|WAIT|>", "<|FLEX|>", "<|DIRECT|>")


def parse_response(response):
    if not isinstance(response, str):
        raise ValueError("response must be a string")
    tag = next((tag for tag in TAGS if response.startswith(tag)), None)
    if tag is None:
        raise ValueError("response must start with exactly one WAIT/FLEX/DIRECT tag")
    delta = response[len(tag):]
    if any(other in delta for other in TAGS):
        raise ValueError("multiple protocol tags in one response")
    if "<|" in delta or "|>" in delta:
        raise ValueError("embedded control token in translation")
    if tag == TAGS[0]:
        if delta.strip():
            raise ValueError("WAIT must not carry translation text")
        return "WAIT", ""
    if not delta.strip():
        raise ValueError("FLEX/DIRECT must carry non-empty translation text")
    return tag[2:-2], delta


def collect_translation(turns):
    """Preserve actual emitted whitespace; do not add artificial word separators."""
    return "".join(parse_response(turn["response"])[1] for turn in turns)
