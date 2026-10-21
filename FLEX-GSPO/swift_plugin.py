"""Loaded by Megatron-SWIFT --external_plugins, not a replacement trainer.

Supported contract is audited against ms-swift f15bb91439d756d4a366b51956d72d1f088b7af6.
See check_environment.py before using another installation.
"""
from pathlib import Path
import sys

# External plugins may be imported from an unrelated working directory.
PROJECT = str(Path(__file__).resolve().parent)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from swift.rewards import ORM, orms
from swift.rollout import MultiTurnScheduler, multi_turns
from swift.template import register_template
from swift.template.templates.qwen import Qwen3OmniTemplate, QwenTemplateMeta

from flex_gspo.audio import materialize_chunk_refs
from flex_gspo.online import FlexStreamHooks
from flex_gspo.online_reward import OnlineQalcoReward


class FlexQwen3OmniTemplate(Qwen3OmniTemplate):
    def _preprocess_inputs(self, inputs):
        # Both teacher-forced training and vLLM see identical independently
        # sliced waveforms. The parent's normal processor/feature expansion is
        # retained; only virtual chunk addressing is new.
        inputs.audios = materialize_chunk_refs(inputs.audios)
        return super()._preprocess_inputs(inputs)


class FlexStreamScheduler(FlexStreamHooks, MultiTurnScheduler):
    pass


class QalcoOnlineORM(ORM):
    def __init__(self, **kwargs):
        super().__init__()
        self.scorer = OnlineQalcoReward()

    def __call__(self, completions, **kwargs):
        return self.scorer(completions, **kwargs)


register_template(QwenTemplateMeta(
    "flex_qwen3_omni", template_cls=FlexQwen3OmniTemplate, default_system=None,
    # Original duplex SFT has no inserted <think> prefix.
    thinking_prefix="", non_thinking_prefix="", is_thinking=False,
))
multi_turns["flex_stream"] = FlexStreamScheduler
orms["qalco_online"] = QalcoOnlineORM
