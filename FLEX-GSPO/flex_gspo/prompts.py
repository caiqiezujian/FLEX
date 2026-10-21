"""The original SFT contract, not a new RL-specific instruction."""

SYSTEM_PROMPT = (
    "你是一个同声传译专家，负责将用户音频中的语音同声翻译为中文。"
    "用户将按时间顺序逐块发送音频。每收到一块音频后，立即输出响应："
    "若该块暂无可翻译内容，输出<|WAIT|>；若仅能确认部分译文、语义尚不完整，输出<|FLEX|>并紧跟"
    "确信的译文（只输出有把握的部分，宁少勿错）；若译文语义已完整，输出<|DIRECT|>并紧跟新识别"
    "的译文。所有译文均为增量输出：只输出新增内容，不重复、不修改此前已输出的任何译文。"
)

PROTOCOL_TOKEN_IDS = {"<|WAIT|>": 151676, "<|DIRECT|>": 151677, "<|FLEX|>": 151678}
AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"


def initial_messages():
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "<audio>"}]
