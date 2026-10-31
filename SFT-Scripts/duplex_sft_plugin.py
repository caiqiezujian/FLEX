"""SWIFT external template plugin for SFT from original audio + chunk ranges.

Supports the inspected SWIFT 3.12.1 and 4.x template import layouts. This is
not a new trainer; Megatron and the native Omni feature/packing paths remain.
"""
from pathlib import Path
from copy import deepcopy
import os
import sys

PROJECT = str(Path(__file__).resolve().parent)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from duplex_audio import PREFIX, SAMPLE_RATE, materialize_chunk_refs

if os.environ.get("FLEX_SFT_ATTENTION_COMPAT") == "1":
    from duplex_attention_compat import install_attention_compat, install_swift_loader_compat
    install_attention_compat()
    install_swift_loader_compat()

try:
    from swift.template.register import TEMPLATE_MAPPING, register_template
    from swift.template.templates.qwen import Qwen3OmniTemplate
except ModuleNotFoundError as exc:
    # Only fall back when the new namespace is absent. Do not hide a missing
    # dependency inside an installed new-style SWIFT package.
    if exc.name not in ("swift.template", "swift.template.register", "swift.template.templates", "swift.template.templates.qwen"):
        raise
    from swift.llm.template.register import TEMPLATE_MAPPING, register_template
    from swift.llm.template.template.qwen import Qwen3OmniTemplate

if not hasattr(Qwen3OmniTemplate, "_preprocess_inputs"):
    raise RuntimeError("Installed SWIFT lacks the required _preprocess_inputs template hook")


class FlexDuplexQwen3OmniTemplate(Qwen3OmniTemplate):
    def _preprocess_inputs(self, inputs):
        audios = inputs.audios
        virtual = [isinstance(value, str) and value.startswith(PREFIX) for value in audios]
        if any(virtual):
            if not all(virtual):
                raise ValueError("Do not mix full recordings and virtual chunks in a single session")
            sr = getattr(self, "sampling_rate", SAMPLE_RATE)
            if sr != SAMPLE_RATE:
                raise ValueError(f"FLEX virtual chunks require a 16000 Hz processor, got {sr}")
            # Only the selected waveform enters the native feature extractor.
            # Keeping bytes here also preserves old SWIFT loader compatibility.
            inputs.audios = materialize_chunk_refs(audios)
        return super()._preprocess_inputs(inputs)


# Preserve the installed native chat wrappers, suffix, thinking settings and
# packing behavior. Only the audio loading hook and template name change.
_meta = deepcopy(TEMPLATE_MAPPING["qwen3_omni"])
_meta.template_type = "flex_duplex_qwen3_omni"
_meta.template_cls = FlexDuplexQwen3OmniTemplate
register_template(_meta)
