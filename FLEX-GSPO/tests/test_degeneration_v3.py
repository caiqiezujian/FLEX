"""Surface-loop guards preserve raw actor output and faithful repetition."""
import copy
import unittest

from flex_gspo.degeneration import _tokens
from flex_gspo.degeneration_v3 import degeneration_report_v3


def turns(*responses):
    return [{"response": response} for response in responses]


class DegenerationV3Tests(unittest.TestCase):
    def report(self, text, source="The board discussed the budget and its next meeting.",
               reference="董事会讨论了预算和下次会议安排。"):
        return degeneration_report_v3(source, reference, turns("<|DIRECT|>" + text))

    def test_subjectless_chinese_loop_is_detected_without_case_specific_rule(self):
        phrase = "自然在收听频道上线"
        context = "董事会今天讨论预算项目安排并检查设备采购计划以及市场报告与客户反馈和员工培训日程随后宣布会议结束大家返回办公地点准备提交新研究方案"
        self.assertEqual(len(_tokens(phrase)), 9)
        self.assertEqual(len(_tokens(context)), 65)
        result = self.report(phrase*14 + context)
        self.assertEqual(result["output_units"], 191)
        self.assertAlmostEqual(result["coverage_fraction"], 126/191)
        self.assertAlmostEqual(result["repetition_penalty"], 99/191)
        self.assertTrue(result["severe_repetition"])

    def test_different_subjectless_clause_uses_general_grammar_cue(self):
        result = self.report("正在向所有观众介绍最新研究结果。" * 5)
        self.assertTrue(result["severe_repetition"])
        self.assertAlmostEqual(result["repetition_penalty"], .4)

    def test_small_four_copy_loop_receives_soft_cost_only(self):
        phrase = "这不是建筑物"
        context = "董事会今天讨论预算项目安排并检查设备采购计划以及市场报告与客户反馈和员工培训日程随后宣布会议结束大家返回办公地点准备提交新研究方案"
        result = self.report(context + phrase*4)
        self.assertFalse(result["severe_repetition"])
        self.assertGreater(result["repetition_penalty"], 0)
        self.assertAlmostEqual(result["repetition_penalty"], 6/89)

    def test_severe_uses_full_loop_coverage_not_excess_coverage(self):
        result = self.report("这不是建筑物。" * 4)
        self.assertTrue(result["severe_repetition"])
        self.assertEqual(result["coverage_fraction"], 1)
        self.assertEqual(result["repetition_penalty"], .25)

    def test_english_clause_loop(self):
        result = self.report("I do not know. " * 5)
        self.assertTrue(result["severe_repetition"])
        self.assertAlmostEqual(result["repetition_penalty"], .4)

    def test_multiple_disjoint_loops_use_union_coverage(self):
        result = self.report("我不明白原因。" * 4 + "这不是建筑物。" * 4)
        self.assertTrue(result["severe_repetition"])
        self.assertEqual(result["coverage_fraction"], 1)
        self.assertEqual(result["repetition_penalty"], .25)

    def test_overlapping_loops_count_excess_units_once(self):
        result = self.report(("这不是建筑物。" * 4 + "今天。") * 4)
        # Outer period26*4: excess26; four inner periods6*4: excess6
        # each, with the final6 already contained by the outer excess span.
        self.assertEqual(result["output_units"], 104)
        self.assertEqual(result["repeated_excess_units"], 44)
        self.assertAlmostEqual(result["repetition_penalty"], 44/104)
        self.assertLessEqual(result["repetition_penalty"], 1)

    def test_short_entities_words_and_punctuation_are_protected(self):
        for text in ("好。"*32, "张三。"*16, "阿尔伯特。"*5,
                     "University of North Carolina. "*5, "北京大学。"*8,
                     "中国国家铁路集团有限公司。"*5, "。。。。"*8):
            with self.subTest(text=text):
                result = self.report(text)
                self.assertFalse(result["severe_repetition"])
                self.assertEqual(result["repetition_penalty"], 0)

    def test_long_nonclausal_phrase_can_have_soft_cost_without_hard_flag(self):
        result = self.report("提高生产效率促进资源合理分配优化管理流程。"*4)
        self.assertFalse(result["severe_repetition"])
        self.assertEqual(result["repetition_penalty"], .25)

    def test_three_copy_parallelism_and_normal_flex_anchors_have_zero_cost(self):
        for text in ("我们可以做到。"*3,
                     "需求增长了，增长来自海外市场。增长速度有所放缓，但增长仍在继续。"):
            result = self.report(text)
            self.assertFalse(result["severe_repetition"])
            self.assertEqual(result["repetition_penalty"], 0)

    def test_reference_comparable_refrain_is_exempted(self):
        text = "我们可以做到。"*5
        result = self.report(text, reference=text)
        self.assertFalse(result["severe_repetition"])
        self.assertEqual(result["repetition_penalty"], 0)
        self.assertEqual(result["exemption_reason"], "comparable_exact_repetition_in_reference")

    def test_reference_once_cannot_excuse_long_loop(self):
        result = self.report("我喜欢它们。"*13, reference="我喜欢它们。")
        self.assertTrue(result["severe_repetition"])
        self.assertGreater(result["repetition_penalty"], .7)

    def test_source_structural_repetition_is_exempted_without_semantic_claim(self):
        result = self.report("我们可以做到。"*5, source="We can do this. "*5,
                             reference="我们可以做到。")
        self.assertFalse(result["severe_repetition"])
        self.assertEqual(result["repetition_penalty"], 0)
        self.assertEqual(result["exemption_reason"], "comparable_structural_repetition_in_source")
        self.assertFalse(result["runs"][0]["source_evidence"]["cross_language_semantic_match_verified"])

    def test_small_source_filler_cannot_excuse_thirteen_copies(self):
        self.assertTrue(self.report("我喜欢它们。"*13,
            source="Yes yes yes yes. Let us start now.")["severe_repetition"])

    def test_reference_exempts_low_coverage_soft_loop(self):
        prefix = "董事会今天讨论预算和设备采购并审议研究项目的进度。"
        text = prefix + "这不是建筑物。"*4
        result = self.report(text, reference=text)
        self.assertEqual(result["repetition_penalty"], 0)

    def test_wait_chunk_boundaries_and_punctuation_do_not_hide_loop(self):
        responses = []
        for _ in range(5):
            responses += ["<|FLEX|>我喜欢", "<|WAIT|>", "<|DIRECT|>它们！ "]
        result = degeneration_report_v3("We held a meeting.", "我们开了会。", turns(*responses))
        self.assertTrue(result["severe_repetition"])
        self.assertAlmostEqual(result["repetition_penalty"], .4)

    def test_raw_translation_is_not_modified_and_tags_do_not_change_score(self):
        trajectory = turns(*["<|FLEX|>我 喜欢它们！ "]*5)
        before = copy.deepcopy(trajectory)
        result = degeneration_report_v3("We met.", "我们见面了。", trajectory)
        self.assertEqual(trajectory, before)
        other = degeneration_report_v3("We met.", "我们见面了。",
                                      turns(*["<|DIRECT|>我 喜欢它们！ "]*5))
        self.assertEqual(result, other)
        self.assertFalse(result["semantic_hallucination_checked"])

    def test_empty_output_and_bad_protocol(self):
        for trajectory in ([], turns("<|WAIT|>"), turns("<|DIRECT|>。。。。")):
            result = degeneration_report_v3("We met.", "我们见面了。", trajectory)
            self.assertEqual(result["repetition_penalty"], 0)
            self.assertFalse(result["severe_repetition"])
        with self.assertRaises(ValueError):
            degeneration_report_v3("We met.", "我们见面了。", turns("unmarked output"))


if __name__ == "__main__":
    unittest.main()
