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

# Environment observation, not source content or a new system instruction.
# Used identically by the bounded training drain and a future deployment drain.
EOF_PROMPT = (
    "本次音频输入已结束，不会再有新的音频。这不是新增的待翻译内容。"
    "请只根据已经听到的音频，输出此前尚未输出且有把握的新增译文，"
    "仍使用<|FLEX|>或<|DIRECT|>标签；不要重复、修改或重译此前的译文，"
    "不要补写未听到的信息，也不要强行补全被截断的句子。"
    "如果没有剩余可翻译内容，只输出<|WAIT|>。"
)


def initial_messages():
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "<audio>"}]
