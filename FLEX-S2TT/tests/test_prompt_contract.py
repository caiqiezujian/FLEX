"""Check trained prompt wording and direction routing without GPU dependencies."""
import ast
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    '_standalone_prompt_support', ROOT / 'tests' / 'test_wait_stop.py')
support = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = support
spec.loader.exec_module(support)
model, cli = support.model, support.cli

EXPECTED_TEMPLATE = (
    '将用户按顺序发送的音频同声翻译为{target_lang}。'
    '结合已听音频和历史译文，每轮只输出以下一种：'
    '暂无可靠新增译文时，仅输出<|WAIT|>；'
    '可直接自然续接时，输出<|DIRECT|>加新增译文；'
    '需暂缓部分内容或调整表达结构时，输出<|FLEX|>加可靠的新增译文。'
    '译文只追加，不重复、不回改、不猜测后文；条件允许时补齐此前暂缓且尚未译出的内容。'
    'DIRECT与FLEX按策略区分，不按句子是否完整区分。'
    '按语义使用标点，不随音频块机械加句号，不用省略号占位。'
    '除WAIT外，译文不能为空；不输出解释。'
)
EXPECTED_IDS = {'<|WAIT|>': 151676, '<|DIRECT|>': 151677, '<|FLEX|>': 151678}
ROUTES = (('en-zh', 'zh', '中文'), ('zh-en', 'en', 'English'))


def literal_assignment(path, name):
    values = [
        ast.literal_eval(node.value)
        for node in ast.parse(path.read_text(encoding='utf-8-sig')).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name
                for target in node.targets)
    ]
    if len(values) != 1:
        raise AssertionError(f'Expected one literal {name} in {path}')
    return values[0]


class PromptContractTests(unittest.TestCase):
    def test_local_prompt_and_protocol_contract(self):
        engine = model.ModelEngine(backend='mock')
        self.addCleanup(engine.close)
        self.assertEqual(tuple(engine.metadata['directions']), tuple(route[0] for route in ROUTES))
        self.assertEqual(dict(zip(model.PROTOCOL_TAGS, model.PROTOCOL_IDS)), EXPECTED_IDS)
        for _, target, word in ROUTES:
            with self.subTest(target=target):
                session = engine.new_session(target)
                self.addCleanup(session.close)
                self.assertEqual(session.target_lang, target)
                self.assertEqual(session.system_prompt, EXPECTED_TEMPLATE.format(target_lang=word))

    def test_cli_routes_current_prompt_into_actual_model_requests(self):
        with tempfile.TemporaryDirectory(prefix='flex-prompt-contract-') as directory:
            root = Path(directory)
            audio = root / 'input.wav'
            support.write_wav(audio, 1.25)
            for direction, target, word in ROUTES:
                with self.subTest(direction=direction):
                    args = cli.parse_args([
                        '--backend', 'mock', '--audio', str(audio),
                        '--output-dir', str(root / direction),
                        '--direction', direction, '--pace', 'fast', '--no-warmup'])
                    self.assertEqual(args.direction, direction)
                    replies = ['<|WAIT|>\n\n', '<|DIRECT|>ok<|im_end|>']
                    with support.fake_engine(root, replies) as (engine, observed):
                        with cli.AudioReader(audio) as reader, cli.OutputFiles(args.output_dir) as output:
                            state = cli.run_stream(args, engine, reader, output,
                                                   cli.RunState(), stdout=io.StringIO())
                        expected = EXPECTED_TEMPLATE.format(target_lang=word)
                        for tokens, _ in observed.calls:
                            prompt = observed.tok.decode(tokens)
                            self.assertTrue(prompt.startswith('<|im_start|>system\n' + expected
                                                              + '<|im_end|>\n'))
                            self.assertEqual(prompt.count('<|im_start|>system\n'), 1)
                        self.assertEqual(len(observed.calls), 2)
                        self.assertEqual(state.processed_samples, 20000)
                        self.assertEqual(state.turns, 2)

    def test_full_repository_sft_gspo_and_web_contracts_match(self):
        repo = ROOT.parent
        sft = repo / 'SFT-Scripts' / 'prepare_duplex_data.py'
        tokens = repo / 'SFT-Scripts' / 'duplex_protocol_tokens.txt'
        gspo = repo / 'FLEX-GSPO' / 'flex_gspo' / 'prompts.py'
        web = repo / 'flex_web' / 'model.py'
        if not all(path.is_file() for path in (sft, tokens, gspo, web)):
            self.skipTest('Cross-repository check needs the full checkout; local contract tests still run')
        trained = literal_assignment(sft, 'PROTOCOL_SYSTEM_PROMPT_TEMPLATE')
        self.assertEqual(model.PROTOCOL_SYSTEM_PROMPT_TEMPLATE, trained)
        self.assertEqual(literal_assignment(web, 'PROTOCOL_SYSTEM_PROMPT_TEMPLATE'), trained)
        self.assertEqual(literal_assignment(gspo, 'SYSTEM_PROMPT'),
                         model.SYSTEM_PROMPTS['zh'])
        self.assertEqual(tuple(tokens.read_text(encoding='utf-8-sig').splitlines()),
                         model.PROTOCOL_TAGS)
        self.assertEqual(literal_assignment(gspo, 'PROTOCOL_TOKEN_IDS'), EXPECTED_IDS)
        for _, target, word in ROUTES:
            with self.subTest(target=target):
                self.assertEqual(model.SYSTEM_PROMPTS[target], trained.format(target_lang=word))


if __name__ == '__main__':
    unittest.main()
