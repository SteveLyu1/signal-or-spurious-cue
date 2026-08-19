from __future__ import annotations

import json
import sys
import unittest
from unittest.mock import patch
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api_pilot"))

import pilot_runner


class PromptTests(unittest.TestCase):
    def test_system_prompt_explicitly_requests_json(self) -> None:
        self.assertIn("JSON", pilot_runner.SYSTEM_PROMPT)

    def test_user_prompt_contains_provider_neutral_schema_contract(self) -> None:
        prompt = pilot_runner.build_user_prompt("M0")
        self.assertIn("top-level object has only `forecasts`", prompt)
        self.assertIn("only `item_id` and `probabilities`", prompt)
        self.assertNotIn('"probabilities":[0', prompt)

    def test_conditions_change_only_metadata_semantics(self) -> None:
        no_country = pilot_runner.build_user_prompt("M0")
        random_country = pilot_runner.build_user_prompt("MR")
        self.assertIn("No country metadata", no_country)
        self.assertIn("uniformly", random_country)
        self.assertIn("contains no information", random_country)
        for question, answer in pilot_runner.PROFILE:
            self.assertIn(question, no_country)
            self.assertIn(answer, no_country)
            self.assertIn(question, random_country)
            self.assertIn(answer, random_country)
        for target in pilot_runner.TARGETS:
            self.assertIn(target["item_id"], no_country)
            self.assertIn(target["item_id"], random_country)

    def test_prompt_has_no_real_respondent_identifier(self) -> None:
        prompt = pilot_runner.build_user_prompt("M0")
        self.assertNotIn("uniqid", prompt.lower())
        self.assertNotIn("respondent_id", prompt.lower())


class ValidationTests(unittest.TestCase):
    def valid_response(self) -> dict:
        return {
            "forecasts": [
                {
                    "item_id": target["item_id"],
                    "probabilities": [0.1] * 10,
                }
                for target in pilot_runner.TARGETS
            ]
        }

    def test_valid_forecast(self) -> None:
        error, parsed = pilot_runner.validate_forecast(self.valid_response())
        self.assertAlmostEqual(error, 0.0)
        self.assertEqual(set(parsed), {target["item_id"] for target in pilot_runner.TARGETS})

    def test_wrong_sum_is_rejected(self) -> None:
        response = self.valid_response()
        response["forecasts"][0]["probabilities"] = [0.2] * 10
        with self.assertRaisesRegex(ValueError, "sum error"):
            pilot_runner.validate_forecast(response)

    def test_duplicate_item_is_rejected(self) -> None:
        response = self.valid_response()
        response["forecasts"][1]["item_id"] = response["forecasts"][0]["item_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pilot_runner.validate_forecast(response)

    def test_non_finite_probability_is_rejected(self) -> None:
        response = self.valid_response()
        response["forecasts"][0]["probabilities"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid probability"):
            pilot_runner.validate_forecast(response)

    def test_condition_shift_summary(self) -> None:
        baseline = {
            target["item_id"]: [0.1] * 10 for target in pilot_runner.TARGETS
        }
        treatment = {item_id: list(values) for item_id, values in baseline.items()}
        treatment[pilot_runner.TARGETS[0]["item_id"]][0] = 0.2
        treatment[pilot_runner.TARGETS[0]["item_id"]][1] = 0.0
        common = {
            "provider": "test",
            "model_requested": "test-model",
            "model_returned": "test-model",
            "success": True,
            "status_code": 200,
            "latency_ms": 1,
        }
        results = [
            pilot_runner.CallResult(
                condition="M0", parsed_forecasts=baseline, **common
            ),
            pilot_runner.CallResult(
                condition="MR", parsed_forecasts=treatment, **common
            ),
        ]
        summary = pilot_runner.condition_shift_summary(results)["test"]
        self.assertAlmostEqual(summary["max_absolute_probability_change"], 0.1)
        self.assertAlmostEqual(summary["items"][0]["total_variation_distance"], 0.1)



class SecurityTests(unittest.TestCase):
    def test_base_url_normalization(self) -> None:
        self.assertEqual(
            pilot_runner.normalized_base_url("https://example.test"),
            "https://example.test/v1",
        )
        self.assertEqual(
            pilot_runner.normalized_base_url("https://example.test/v1/"),
            "https://example.test/v1",
        )

    def test_error_redaction(self) -> None:
        message = "Authorization Bearer sk-example-secret failed"
        redacted = pilot_runner.redact_message(message)
        self.assertNotIn("sk-example-secret", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_report_does_not_accept_credentials(self) -> None:
        result = pilot_runner.CallResult(
            provider="openai",
            model_requested="test",
            model_returned="test",
            condition="M0",
            success=True,
            status_code=200,
            latency_ms=1,
        )
        serialized = json.dumps(result.__dict__)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("authorization", serialized.lower())

    def test_transport_error_is_structured_without_retry(self) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("synthetic outage"),
        ) as mocked:
            status, data, _ = pilot_runner.http_json(
                "GET", "https://example.test/v1/models", "not-a-real-key"
            )
        self.assertEqual(status, 0)
        self.assertEqual(data["error"]["type"], "transport_error")
        self.assertEqual(mocked.call_count, 1)


if __name__ == "__main__":
    unittest.main()
