"""Current service configuration/text contract; no in-process MetricX backend."""
import unittest

from flex_gspo.metricx import validate_metricx_settings, validate_metricx_text
from flex_gspo.rewards import RewardError


class MetricXConfigurationTests(unittest.TestCase):
    def test_defaults_are_service_qe_with_unconfigured_endpoint(self):
        settings = validate_metricx_settings({})
        self.assertEqual(settings, {"backend": "service", "mode": "qe",
                                    "service_url": "", "service_id": ""})

    def test_configuration_validation_does_not_mutate_input(self):
        original = {"backend": "service"}
        validate_metricx_settings(original)
        self.assertEqual(original, {"backend": "service"})

    def test_configuration_must_be_object(self):
        for settings in (None, [], "", True):
            with self.subTest(settings=settings), self.assertRaises(RewardError):
                validate_metricx_settings(settings)

    def test_only_service_backend_and_supported_fields_are_accepted(self):
        with self.assertRaisesRegex(RewardError, "backend must be service"):
            validate_metricx_settings({"backend": "unsupported_backend"})
        for key in ("unknown_option", "service_ur1", "api_key"):
            with self.subTest(key=key), self.assertRaisesRegex(RewardError, "Unsupported MetricX setting"):
                validate_metricx_settings({key: "invalid setting"})

    def test_modes_are_explicit(self):
        self.assertEqual(validate_metricx_settings({"mode": "reference"})["mode"], "reference")
        for mode in ("auto", "QE", None):
            with self.subTest(mode=mode), self.assertRaises(RewardError):
                validate_metricx_settings({"mode": mode})

    def test_service_details_reject_types_and_surrounding_whitespace(self):
        for key in ("service_url", "service_id"):
            for value in (False, 1, None, "  pending", "pending "):
                with self.subTest(key=key, value=value), self.assertRaises(RewardError):
                    validate_metricx_settings({key: value})

    def test_text_contract_requires_source_and_candidate(self):
        for source, target in ((None, "中文"), ("", "中文"), ("源文", " "),
                               ("源文", None), ([], "中文")):
            with self.subTest(source=source, target=target), self.assertRaises(RewardError):
                validate_metricx_text(source, target)

    def test_reference_is_auxiliary_in_qe_and_required_in_reference_mode(self):
        validate_metricx_text("English", "中文", None, mode="qe")
        validate_metricx_text("English", "中文", "参考", mode="reference")
        with self.assertRaises(RewardError):
            validate_metricx_text("English", "中文", None, mode="reference")
        with self.assertRaises(RewardError):
            validate_metricx_text("English", "中文", "参考", mode="unknown")


if __name__ == "__main__":
    unittest.main()
