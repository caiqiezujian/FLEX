"""Current SFT contract, aligned with SFT-Scripts/prepare_duplex_data.py."""

SYSTEM_PROMPT = (
    "将用户按顺序发送的音频同声翻译为中文。"
    "结合已听音频和历史译文，每轮只输出以下一种："
    "暂无可靠新增译文时，仅输出<|WAIT|>；"
    "可直接自然续接时，输出<|DIRECT|>加新增译文；"
    "需暂缓部分内容或调整表达结构时，输出<|FLEX|>加可靠的新增译文。"
    "译文只追加，不重复、不回改、不猜测后文；条件允许时补齐此前暂缓且尚未译出的内容。"
    "DIRECT与FLEX按策略区分，不按句子是否完整区分。"
    "按语义使用标点，不随音频块机械加句号，不用省略号占位。"
    "除WAIT外，译文不能为空；不输出解释。"
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
