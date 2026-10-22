import copy
import unittest

from flex_gspo.degeneration import degeneration_report


def turns(*responses):
    return [{"response": response} for response in responses]


class DegenerationTests(unittest.TestCase):
    def report(self, text, source="The board met today to discuss its annual budget.",
               reference="董事会今天开会讨论年度预算。"):
        return degeneration_report(source, reference, turns("<|DIRECT|>" + text))

    def test_observed_thirteen_copy_failure_across_turns(self):
        trajectory = turns(*["<|DIRECT|>我喜欢它们。"] * 13)
        result = degeneration_report("Let me put my tea down.", "让我把茶放下。", trajectory)
        self.assertTrue(result["severe_repetition"])
        self.assertEqual(result["repetitions"], 13)
        self.assertEqual(result["coverage_fraction"], 1)

    def test_single_turn_and_punctuation_variations(self):
        result = self.report("我喜欢它们。 我 喜欢它们！我喜欢它们，\n我喜欢它们？")
        self.assertTrue(result["severe_repetition"])
        self.assertEqual(result["repetitions"], 4)

    def test_chunk_boundaries_and_wait_do_not_hide_a_loop(self):
        responses = []
        for _ in range(5):
            responses.extend(["<|FLEX|>我喜欢", "<|WAIT|>", "<|DIRECT|>它们。"])
        self.assertTrue(degeneration_report("We held a meeting.", "我们开了会。",
                                           turns(*responses))["severe_repetition"])

    def test_ordinary_flex_semantic_anchors_not_rejected(self):
        self.assertFalse(self.report(
            "需求增长了，增长来自海外市场。增长速度有所放缓，但增长仍在继续。"
        )["severe_repetition"])

    def test_short_words_and_names_not_rejected_by_multiple_period(self):
        for text in ("好。" * 32, "张三。" * 16, "阿尔伯特。" * 5,
                     "University of North Carolina. " * 5):
            with self.subTest(text=text):
                self.assertFalse(self.report(text)["severe_repetition"])

    def test_reference_supports_repeated_refrain(self):
        text = "我们可以做到。" * 5
        result = self.report(text, reference=text)
        self.assertFalse(result["severe_repetition"])
        self.assertEqual(result["exemption_reason"], "comparable_exact_repetition_in_reference")

    def test_single_reference_mention_does_not_excuse_thirteen_copies(self):
        self.assertTrue(self.report("我喜欢它们。" * 13,
                                    reference="我喜欢它们。")["severe_repetition"])

    def test_cross_lingual_source_repetition_is_conservatively_exempted(self):
        result = self.report("我们可以做到。" * 5,
                             source="We can do this. " * 5,
                             reference="我们可以做到。")
        self.assertFalse(result["severe_repetition"])
        self.assertEqual(result["exemption_reason"], "comparable_dominant_repetition_in_source")

    def test_source_small_filler_repetition_not_enough_for_large_loop(self):
        self.assertTrue(self.report("我喜欢它们。" * 13,
                                    source="Yes yes yes yes. Let us start now.")["severe_repetition"])

    def test_two_or_three_copies_are_not_hard_failures(self):
        for count in (2, 3):
            self.assertFalse(self.report("我喜欢它们。" * count)["severe_repetition"])

    def test_small_repeat_in_otherwise_long_output_is_diagnostic(self):
        text = "我喜欢它们。" * 4 + "董事会今天开会讨论年度预算以及计划中的设备采购并审议了研究项目。" * 2
        result = self.report(text)
        self.assertFalse(result["severe_repetition"])
        self.assertEqual(result["exemption_reason"], "loop_not_dominant")

    def test_tag_choice_does_not_change_guard(self):
        a = turns(*["<|FLEX|>我喜欢它们。"] * 5)
        b = turns(*["<|DIRECT|>我喜欢它们。"] * 5)
        self.assertEqual(degeneration_report("We met.", "我们见面了。", a),
                         degeneration_report("We met.", "我们见面了。", b))

    def test_english_clause_loop(self):
        self.assertTrue(self.report("I do not know. " * 5)["severe_repetition"])

    def test_empty_wait_and_punctuation_only_not_repetition(self):
        for trajectory in ([], turns("<|WAIT|>"), turns("<|DIRECT|>。。。。")):
            result = degeneration_report("We met.", "我们见面了。", trajectory)
            self.assertFalse(result["severe_repetition"])

    def test_does_not_modify_raw_turns(self):
        trajectory = turns(*["<|DIRECT|>我喜欢它们。"] * 13)
        before = copy.deepcopy(trajectory)
        degeneration_report("We met.", "我们见面了。", trajectory)
        self.assertEqual(before, trajectory)

    def test_bad_protocol_raises_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            degeneration_report("We met.", "我们见面了。", turns("unmarked output"))


if __name__ == "__main__":
    unittest.main()
