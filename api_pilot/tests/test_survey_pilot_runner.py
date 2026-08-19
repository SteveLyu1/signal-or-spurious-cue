from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api_pilot"))

import survey_pilot_runner as runner


def test_credential_or_quota_error_detects_qwen_arrearage():
    assert runner.is_credential_or_quota_error({
        "status_code": 400,
        "error_type": "Arrearage",
        "error_message": "account is in good standing; overdue-payment",
    })
    assert runner.is_credential_or_quota_error({
        "status_code": 402, "error_type": "api_error", "error_message": "",
    })
    assert not runner.is_credential_or_quota_error({
        "status_code": 429, "error_type": "rate_limit", "error_message": "",
    })


def sample_profile() -> dict:
    evidence = [
        {"item_id": item_id, "answer_code": "1", "answer": "Synthetic answer"}
        for item_id in (
            "A006", "A165", "B008", "C001", "C039",
            "D059", "E023", "E035", "E037", "F063",
        )
    ]
    targets = [
        {"item_id": item_id, "answer": "Hidden synthetic answer"}
        for item_id in ("A124_09", "E025", "E027", "A030", "A124_06", "A004")
    ]
    target_options = {
        "A124_09": [{"answer_code": "0", "answer": "No"}, {"answer_code": "1", "answer": "Yes"}],
        "E025": [{"answer_code": "1", "answer": "Have done"}, {"answer_code": "2", "answer": "Might do"}, {"answer_code": "3", "answer": "Would never do"}],
        "E027": [{"answer_code": "1", "answer": "Have done"}, {"answer_code": "2", "answer": "Might do"}, {"answer_code": "3", "answer": "Would never do"}],
        "A030": [{"answer_code": "0", "answer": "No"}, {"answer_code": "1", "answer": "Yes"}],
        "A124_06": [{"answer_code": "0", "answer": "No"}, {"answer_code": "1", "answer": "Yes"}],
        "A004": [
            {"answer_code": "1", "answer": "Very"}, {"answer_code": "2", "answer": "Rather"},
            {"answer_code": "3", "answer": "Not very"}, {"answer_code": "4", "answer": "Not at all"},
        ],
    }
    return {
        "profile_id": "P-CN-001",
        "true_country_code": "CN",
        "true_country_label": "China",
        "balanced_random_country_label": "France",
        "evidence": evidence,
        "shuffled_evidence": list(reversed(evidence)),
        "targets": targets,
        "target_options": target_options,
    }


class PromptTests(unittest.TestCase):
    def test_four_metadata_conditions(self) -> None:
        profile = sample_profile()
        m0 = runner.metadata_text(profile, "M0")
        mo = runner.metadata_text(profile, "MO")
        mr = runner.metadata_text(profile, "MR")
        mv = runner.metadata_text(profile, "MV")
        self.assertNotIn("France", m0)
        self.assertIn("France", mo)
        self.assertNotIn("random", mo.lower())
        self.assertIn("uniformly", mr)
        self.assertIn("China", mv)
        self.assertIn("country in which", mv)

    def test_repeat_condition_is_prompt_identical_to_m0(self) -> None:
        profile = sample_profile()
        m0, _ = runner.build_prompt(profile, "M0", "correct")
        repeated, _ = runner.build_prompt(profile, "M0_REPEAT", "correct")
        self.assertEqual(m0, repeated)

    def test_repeat_task_count(self) -> None:
        profiles = []
        for country in runner.COUNTRIES:
            for index in range(4):
                profile = sample_profile()
                profile["profile_id"] = f"P-{country}-{index + 1:03d}"
                profile["true_country_code"] = country
                profile["true_country_label"] = runner.COUNTRY_LABELS[country]
                profiles.append(profile)
        tasks = runner.build_tasks(
            profiles,
            ["deepseek", "openai"],
            {"deepseek": "d", "openai": "o"},
            repeat_m0_profiles_per_country=4,
        )
        self.assertEqual(len(tasks), 48)
        self.assertEqual({task["condition"] for task in tasks}, {"M0_REPEAT"})
        limited = runner.limit_tasks_balanced(tasks, ["deepseek", "openai"], 6)
        self.assertEqual(len(limited), 12)
        self.assertEqual({task["condition"] for task in limited}, {"M0_REPEAT"})

    def test_balanced_core_repeats_are_matched_by_provider_and_cell(self) -> None:
        profiles = []
        for country in runner.COUNTRIES:
            profile = sample_profile()
            profile["profile_id"] = f"P-{country}-001"
            profile["true_country_code"] = country
            profile["true_country_label"] = runner.COUNTRY_LABELS[country]
            profiles.append(profile)
        providers = ["deepseek", "qwen"]
        tasks = runner.build_tasks(
            profiles,
            providers,
            {"deepseek": "d", "qwen": "q"},
            target_batch_size=5,
            target_batch_limit=1,
        )
        combined = runner.add_balanced_core_repeats(tasks, providers, 1)
        repeats = [task for task in combined if task["experiment"] == "repeat"]
        self.assertEqual(len(repeats), len(providers) * len(runner.CELLS))
        for repeat in repeats:
            self.assertTrue(repeat["call_id"].endswith("/REPEAT-01"))
            self.assertEqual(
                repeat["source_call_id"], repeat["call_id"].removesuffix("/REPEAT-01")
            )

    def test_v2_includes_codes_and_scale_meanings(self) -> None:
        bundle = runner.load_prompt_bundle(
            ROOT / "api_pilot" / "prompts" / "survey-confirmatory-v2.json"
        )
        prompt, _ = runner.build_prompt(sample_profile(), "MR", "correct", bundle)
        self.assertIn("Recorded answer: [1] Synthetic answer", prompt)
        self.assertIn("Scale: 1 = Incomes should be made more equal", prompt)
        self.assertIn("[1] Have done | [2] Might do | [3] Would never do", prompt)
        self.assertIn("sampled uniformly", prompt)
        self.assertIn("6 masked item IDs", prompt)
        self.assertNotIn("<<TARGET_COUNT>>", prompt)

    def test_v2_call_id_does_not_collide_with_v1(self) -> None:
        legacy = runner.make_call_id("openai", "m", "p", "M0", "correct", "survey-pilot-v1")
        revised = runner.make_call_id("openai", "m", "p", "M0", "correct", "survey-confirmatory-v2")
        self.assertNotEqual(legacy, revised)
        self.assertIn("survey-confirmatory-v2", revised)

    def test_v3_conditions_change_only_the_metadata_block(self) -> None:
        bundle = runner.load_prompt_bundle(
            ROOT / "api_pilot" / "prompts" / "survey-confirmatory-v3.json"
        )
        profile = sample_profile()
        prompts = {
            condition: runner.build_prompt(profile, condition, "correct", bundle)[0]
            for condition in ("M0", "MO", "MR", "MV")
        }

        self.assertNotIn("<metadata>", prompts["M0"])
        self.assertNotIn("country", prompts["M0"].lower())
        self.assertIn("Task: Forecast the masked answers", prompts["M0"])
        for condition in ("MO", "MR", "MV"):
            self.assertIn("Task: Forecast the masked answers", prompts[condition])
            self.assertIn("JSON contract", prompts[condition])
            self.assertIn("<metadata>", prompts[condition])

    def test_v3_mr_discloses_assignment_without_normative_instruction(self) -> None:
        bundle = runner.load_prompt_bundle(
            ROOT / "api_pilot" / "prompts" / "survey-confirmatory-v3.json"
        )
        mr = runner.metadata_text(sample_profile(), "MR", bundle)
        self.assertIn("uniformly", mr)
        self.assertIn("independently", mr)
        self.assertNotIn("should", mr.lower())
        self.assertNotIn("no information", mr.lower())
        self.assertNotIn("uninformative", mr.lower())

    def test_v3_m0_repeat_is_identical(self) -> None:
        bundle = runner.load_prompt_bundle(
            ROOT / "api_pilot" / "prompts" / "survey-confirmatory-v3.json"
        )
        m0, _ = runner.build_prompt(sample_profile(), "M0", "correct", bundle)
        repeat, _ = runner.build_prompt(
            sample_profile(), "M0_REPEAT", "correct", bundle
        )
        self.assertEqual(m0, repeat)

    def test_all_five_confirmatory_providers_are_supported(self) -> None:
        self.assertEqual(runner.DEFAULT_MISTRAL_MODEL, "mistral-medium-3-5")
        self.assertEqual(runner.DEFAULT_OPENAI_MODEL, "gpt-5.4-2026-03-05")
        self.assertEqual(runner.DEFAULT_DEEPSEEK_MODEL, "deepseek-v4-pro")
        self.assertEqual(runner.DEFAULT_GEMINI_MODEL, "gemini-3.5-flash")
        self.assertEqual(runner.DEFAULT_QWEN_MODEL, "qwen3.7-max-2026-06-08")
        self.assertEqual(set(runner.PROVIDERS), {"openai", "deepseek", "mistral", "gemini", "qwen"})
        self.assertTrue(runner.MISTRAL_BASE_URL.startswith("https://"))
        self.assertTrue(runner.GEMINI_BASE_URL.startswith("https://"))
        self.assertTrue(runner.QWEN_BASE_URL.startswith("https://"))

    def test_target_batch_ids_prevent_checkpoint_collisions(self) -> None:
        profile = sample_profile()
        tasks = runner.build_tasks(
            [profile],
            ["deepseek"],
            {"deepseek": "d"},
            bundle=runner.load_prompt_bundle(
                ROOT / "api_pilot" / "prompts" / "survey-confirmatory-v2.json"
            ),
            target_batch_size=2,
        )
        self.assertEqual(len(tasks), 18)
        self.assertEqual({task["batch_id"] for task in tasks}, {"B01", "B02", "B03"})
        self.assertEqual(len({task["call_id"] for task in tasks}), len(tasks))
        self.assertTrue(all(len(task["targets"]) == 2 for task in tasks))

    def test_auxiliary_manifest_selects_cells_batches_and_experiment(self) -> None:
        profile = sample_profile()
        tasks = runner.build_tasks(
            [profile],
            ["deepseek"],
            {"deepseek": "d"},
            target_batch_size=2,
            cells=(("MO", "shuffled"), ("MV", "shuffled")),
            target_batch_ids=("B01", "B03"),
            experiment_id="mechanism",
        )
        self.assertEqual(len(tasks), 4)
        self.assertEqual({task["batch_id"] for task in tasks}, {"B01", "B03"})
        self.assertEqual(
            {(task["condition"], task["evidence_variant"]) for task in tasks},
            {("MO", "shuffled"), ("MV", "shuffled")},
        )
        self.assertEqual({task["experiment"] for task in tasks}, {"mechanism"})
        self.assertEqual(len({task["call_id"] for task in tasks}), 4)

    def test_task_manifest_rejects_duplicate_cells_and_missing_profiles(self) -> None:
        manifest = {
            "version": "v1",
            "status": "frozen",
            "experiment_id": "mechanism",
            "profile_ids": ["P-CN-001"],
            "cells": [["MO", "shuffled"], ["MO", "shuffled"]],
            "target_batches": ["B01"],
            "profile_file_sha256": "a",
            "prompt_bundle_sha256": "b",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate cells"):
                runner.load_task_manifest(path)
            manifest["cells"] = [["MO", "shuffled"]]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = runner.load_task_manifest(path)
            loaded["profile_ids"] = ["missing"]
            with self.assertRaisesRegex(ValueError, "missing profiles"):
                runner.apply_task_manifest([sample_profile()], loaded)

    def test_task_manifest_rejects_empty_or_duplicate_batches(self) -> None:
        manifest = {
            "version": "v1",
            "status": "frozen",
            "experiment_id": "mechanism",
            "profile_ids": ["P-CN-001"],
            "cells": [["MO", "shuffled"]],
            "target_batches": [],
            "profile_file_sha256": "a",
            "prompt_bundle_sha256": "b",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target batch"):
                runner.load_task_manifest(path)
            manifest["target_batches"] = ["B01", "B01"]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate target batches"):
                runner.load_task_manifest(path)

    def test_legacy_bundle_batch_ids_also_prevent_checkpoint_collisions(self) -> None:
        profile = sample_profile()
        tasks = runner.build_tasks(
            [profile],
            ["deepseek"],
            {"deepseek": "d"},
            bundle=runner.load_prompt_bundle(
                ROOT / "api_pilot" / "prompts" / "survey-pilot-v1.json"
            ),
            target_batch_size=2,
        )
        self.assertEqual(len({task["call_id"] for task in tasks}), len(tasks))

    def test_v2_call_id_changes_when_prompt_changes(self) -> None:
        first = runner.make_call_id(
            "deepseek", "m", "p", "M0", "correct",
            "survey-confirmatory-v2", "B01", "a" * 64,
        )
        second = runner.make_call_id(
            "deepseek", "m", "p", "M0", "correct",
            "survey-confirmatory-v2", "B01", "b" * 64,
        )
        self.assertNotEqual(first, second)

    def test_v2_bundle_covers_confirmatory_manifest(self) -> None:
        bundle = runner.load_prompt_bundle(
            ROOT / "api_pilot" / "prompts" / "survey-confirmatory-v2.json"
        )
        manifest = json.loads(
            (ROOT / "dataset" / "analysis" / "confirmatory_manifest_v2.json")
            .read_text(encoding="utf-8")
        )
        target_ids = {
            item_id
            for batch in manifest["target_batches"].values()
            for item_id in batch
        }
        self.assertEqual(len(target_ids), 20)
        self.assertTrue(target_ids.issubset(bundle["items"]))

    def test_prompt_does_not_contain_hidden_answers_or_ids(self) -> None:
        profile = sample_profile()
        profile["analysis_weight"] = 2.75
        prompt, _ = runner.build_prompt(profile, "M0", "correct")
        self.assertNotIn("Hidden synthetic answer", prompt)
        self.assertNotIn("P-CN-001", prompt)
        self.assertNotIn("uniqid", prompt.lower())
        self.assertNotIn("2.75", prompt)


class ValidationTests(unittest.TestCase):
    def test_variable_option_lengths(self) -> None:
        profile = sample_profile()
        targets = runner.target_spec(profile)
        data = {
            "forecasts": {
                target["item_id"]: [1 / len(target["options"])] * len(target["options"])
                for target in targets
            }
        }
        parsed, error = runner.validate_forecast(data, targets)
        self.assertEqual(set(parsed), {target["item_id"] for target in targets})
        self.assertLess(error, 1e-9)

    def test_wrong_option_length_rejected(self) -> None:
        profile = sample_profile()
        targets = runner.target_spec(profile)
        data = {
            "forecasts": {
                target["item_id"]: [1.0]
                for target in targets
            }
        }
        with self.assertRaisesRegex(ValueError, "wrong probability count"):
            runner.validate_forecast(data, targets)

    def test_mistral_bounded_sum_repair_preserves_relative_weights(self) -> None:
        targets = [{
            "item_id": "Q", "options": [str(i) for i in range(10)]
        }]
        parsed, sums = runner.repair_mistral_probability_sums(
            {"forecasts": {"Q": [0.1] * 9 + [0.2]}}, targets
        )
        self.assertAlmostEqual(sums["Q"], 1.1)
        self.assertAlmostEqual(sum(parsed["Q"]), 1.0)
        self.assertAlmostEqual(parsed["Q"][-1] / parsed["Q"][0], 2.0)

    def test_mistral_sum_repair_rejects_larger_error(self) -> None:
        targets = [{"item_id": "Q", "options": ["1", "2"]}]
        with self.assertRaisesRegex(ValueError, "probability sum error"):
            runner.repair_mistral_probability_sums(
                {"forecasts": {"Q": [0.2, 0.2]}}, targets
            )


class SecurityTests(unittest.TestCase):
    def test_generic_rate_limit_is_not_a_permanent_credential_error(self) -> None:
        result = {
            "status_code": 429,
            "error_type": "429",
            "error_message": "You exceeded your current quota; check plan and billing details.",
        }
        self.assertFalse(runner.is_credential_or_quota_error(result))

    def test_depleted_prepayment_is_a_permanent_credential_error(self) -> None:
        result = {
            "status_code": 429,
            "error_type": "429",
            "error_message": "Your prepayment credits are depleted.",
        }
        self.assertTrue(runner.is_credential_or_quota_error(result))

    def test_profile_loader_rejects_source_identifier(self) -> None:
        profile = sample_profile()
        profile["uniqid"] = "not-allowed"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.jsonl"
            path.write_text(json.dumps(profile) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source identifier"):
                runner.load_profiles(path, 1)

    def test_labelled_key_file_supports_multiple_keys_without_persisting_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keys.txt"
            path.write_text(
                "deepseek:\nsecret-one\ndeepseek:\nsecret-two\nqwen:\nsecret-three\n",
                encoding="utf-8",
            )
            keys = runner.load_key_file(path)
        self.assertEqual(keys["deepseek"], ["secret-one", "secret-two"])
        self.assertEqual(keys["qwen"], ["secret-three"])

    def test_resume_prefers_a_key_other_than_the_previous_failed_slot(self) -> None:
        previous = {"success": False, "credential_slot": 1}
        self.assertEqual(runner.choose_key_index([0, 1], 0, previous), 1)
        self.assertEqual(runner.choose_key_index([0], 0, previous), 0)


class RecoveryTests(unittest.TestCase):
    def test_load_completed_ignores_only_a_truncated_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            path.write_bytes(
                (json.dumps({"call_id": "ok", "success": True}) + "\n").encode()
                + b'{"call_id":"partial"'
            )
            completed = runner.load_completed(path)
        self.assertEqual(set(completed), {"ok"})

    def test_load_completed_rejects_middle_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            path.write_text('{bad}\n{"call_id":"ok","success":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid checkpoint line 1"):
                runner.load_completed(path)

    def test_sqlite_checkpoint_resumes_success_but_not_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            store = runner.CheckpointStore(path, "signature", {"run": "test"})
            store.record({"call_id": "ok", "provider": "qwen", "success": True})
            store.record({"call_id": "retry", "provider": "qwen", "success": False})
            self.assertEqual(store.successful_ids(), {"ok"})
            store.close()
            reopened = runner.CheckpointStore(path, "signature", {"run": "test"})
            self.assertEqual(reopened.successful_ids(), {"ok"})
            reopened.close()

    def test_sqlite_history_is_append_only_when_latest_result_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            store = runner.CheckpointStore(path, "signature", {"run": "test"})
            store.record({"call_id": "same", "provider": "qwen", "success": False})
            store.record({"call_id": "same", "provider": "qwen", "success": True})
            history = store.connection.execute(
                "SELECT COUNT(*) FROM result_history WHERE call_id='same'"
            ).fetchone()[0]
            latest = store.connection.execute(
                "SELECT success FROM results WHERE call_id='same'"
            ).fetchone()[0]
            self.assertEqual(history, 2)
            self.assertEqual(latest, 1)
            store.close()

    def test_empty_sqlite_checkpoint_can_rebind_after_runner_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            store = runner.CheckpointStore(path, "first", {"run": "one"})
            store.close()
            rebound = runner.CheckpointStore(path, "second", {"run": "two"})
            self.assertEqual(rebound.successful_ids(), set())
            rebound.close()

    def test_nonempty_sqlite_checkpoint_rejects_different_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            store = runner.CheckpointStore(path, "first", {"run": "one"})
            store.record({"call_id": "ok", "provider": "qwen", "success": True})
            store.close()
            with self.assertRaisesRegex(ValueError, "different frozen run"):
                runner.CheckpointStore(path, "second", {"run": "two"})

    def test_checkpoint_rejects_a_second_live_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            first = runner.CheckpointStore(path, "signature", {"run": "test"})
            try:
                with self.assertRaisesRegex(RuntimeError, "active runner"):
                    runner.CheckpointStore(path, "signature", {"run": "test"})
            finally:
                first.close()

    def test_checkpoint_recovers_a_stale_lock_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            lock_path = path.with_name(path.name + ".runner.lock")
            lock_path.write_text(
                json.dumps({"pid": 999999, "process_token": "stale"}),
                encoding="utf-8",
            )
            store = runner.CheckpointStore(path, "signature", {"run": "test"})
            token = store.run_lock.token
            store.close()
            owner = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(owner["pid"], os.getpid())
            self.assertNotEqual(owner["process_token"], "stale")
            self.assertEqual(owner["process_token"], token)

    def test_failure_cannot_replace_an_existing_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            store = runner.CheckpointStore(path, "signature", {"run": "test"})
            store.record({
                "call_id": "same", "provider": "qwen", "success": True,
                "parsed_forecasts": {"x": [1.0]},
            })
            store.record({
                "call_id": "same", "provider": "qwen", "success": False,
                "error_type": "late_failure",
            })
            latest = store.latest_results()["same"]
            history_count = store.connection.execute(
                "SELECT COUNT(*) FROM result_history WHERE call_id='same'"
            ).fetchone()[0]
            self.assertTrue(latest["success"])
            self.assertEqual(latest["parsed_forecasts"], {"x": [1.0]})
            self.assertEqual(history_count, 2)
            store.close()

    def test_run_metadata_can_rebind_only_with_empty_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.sqlite3"
            metadata = root / "run.json"
            store = runner.CheckpointStore(checkpoint, "first", {"run": "one"})
            store.close()
            metadata.write_text(
                json.dumps({"run_signature": "first"}), encoding="utf-8"
            )
            runner.write_run_metadata(
                metadata, {"run_signature": "second"}, checkpoint
            )
            self.assertEqual(
                json.loads(metadata.read_text(encoding="utf-8"))["run_signature"],
                "second",
            )

    def test_sqlite_import_rejects_foreign_call_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.sqlite3"
            legacy = Path(directory) / "legacy.jsonl"
            legacy.write_text(
                json.dumps(
                    {"call_id": "foreign", "provider": "qwen", "success": True}
                )
                + "\n",
                encoding="utf-8",
            )
            store = runner.CheckpointStore(checkpoint, "signature", {"run": "test"})
            with self.assertRaises(ValueError):
                store.import_jsonl(legacy, {"expected"})
            store.close()


class ProviderTests(unittest.TestCase):
    def test_qwen_uses_non_thinking_json_request(self) -> None:
        profile = sample_profile()
        task = runner.build_tasks(
            [profile], ["qwen"], {"qwen": "qwen-test"}
        )[0]
        forecasts = {
            target["item_id"]: [1 / len(target["options"])] * len(target["options"])
            for target in task["targets"]
        }
        response = {
            "model": "qwen-test",
            "choices": [{"message": {"content": json.dumps({"forecasts": forecasts})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        with patch.object(runner, "http_json", return_value=(200, response, 1)) as mocked:
            task["attempt"] = 1
            result = runner.call_qwen(task, "secret")
        payload = mocked.call_args.args[3]
        self.assertFalse(payload["enable_thinking"])
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertTrue(result["success"])

    def test_mistral_uses_schema_and_no_reasoning(self) -> None:
        profile = sample_profile()
        task = runner.build_tasks(
            [profile], ["mistral"], {"mistral": "mistral-test"}
        )[0]
        forecasts = {
            target["item_id"]: [1 / len(target["options"])] * len(target["options"])
            for target in task["targets"]
        }
        response = {
            "model": "mistral-test",
            "choices": [{"message": {"content": json.dumps({"forecasts": forecasts})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        with patch.object(runner, "http_json", return_value=(200, response, 1)) as mocked:
            task["attempt"] = 1
            result = runner.call_mistral(task, "secret")
        payload = mocked.call_args.args[3]
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(result["success"])

    def test_progress_document_has_model_and_cell_tables(self) -> None:
        profile = sample_profile()
        tasks = runner.build_tasks([profile], ["qwen"], {"qwen": "qwen-test"})
        tracker = runner.ProgressTracker(
            tasks, {}, {"qwen": "qwen-test"}, {"qwen": 2}, {"qwen": 8}
        )
        document = tracker.render()
        self.assertIn("## Model Progress", document)
        self.assertIn("## Experimental Cell Progress", document)
        self.assertIn("qwen-test", document)
        self.assertIn("Estimated total remaining time", document)

    def test_progress_counts_terminal_invalid_as_complete_but_not_valid(self) -> None:
        profile = sample_profile()
        tasks = runner.build_tasks([profile], ["qwen"], {"qwen": "qwen-test"})
        result = {
            "call_id": tasks[0]["call_id"],
            "provider": "qwen",
            "condition": tasks[0]["condition"],
            "evidence_variant": tasks[0]["evidence_variant"],
            "success": False,
            "terminal_failure": True,
        }
        tracker = runner.ProgressTracker(
            tasks, {result["call_id"]: result}, {"qwen": "qwen-test"},
            {"qwen": 1}, {"qwen": 1},
        )
        document = tracker.render(stopped=True)
        self.assertIn("Terminal invalid outputs: **1**", document)
        self.assertIn("| qwen | `qwen-test` | 1 | 1 | 6 | 0 | 1 |", document)

    def test_checkpoint_terminal_invalid_remains_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            metadata = {"test": "terminal-resume"}
            store = runner.CheckpointStore(path, "signature", metadata)
            store.record({
                "call_id": "retry-me",
                "provider": "mistral",
                "success": False,
                "terminal_failure": True,
                "error_type": "validation_error",
            })
            self.assertNotIn("retry-me", store.completed_ids())
            store.close()

    def test_operational_runner_migration_preserves_scientific_contract(self) -> None:
        previous = {
            "created_at": "before", "run_signature": "old",
            "runner_sha256": "runner-old",
            "validation_retry_policy": "old policy",
            "models": {"mistral": "frozen-model"},
            "prompt_bundle_sha256": "prompt-hash", "planned_calls": 1,
        }
        current = {
            **previous, "created_at": "after", "run_signature": "new",
            "runner_sha256": "runner-new",
            "validation_retry_policy": "retry across cycles",
        }
        self.assertTrue(runner.operationally_compatible_run(previous, current))
        changed_model = {**current, "models": {"mistral": "other-model"}}
        self.assertFalse(
            runner.operationally_compatible_run(previous, changed_model)
        )

    def test_small_pilot_selection_is_matched_across_models(self) -> None:
        profiles = []
        for country in runner.COUNTRIES:
            profile = sample_profile()
            profile["profile_id"] = f"P-{country}-001"
            profile["true_country_code"] = country
            profile["true_country_label"] = runner.COUNTRY_LABELS[country]
            profiles.append(profile)
        providers = ["openai", "deepseek"]
        tasks = runner.build_tasks(
            profiles,
            providers,
            {"openai": "openai-test", "deepseek": "deepseek-test"},
            target_batch_size=5,
            target_batch_limit=1,
        )
        selected = runner.limit_tasks_balanced(tasks, providers, 6)
        matched = {}
        for task in selected:
            cell = (task["condition"], task["evidence_variant"])
            matched.setdefault(task["provider"], {})[cell] = (
                task["profile"]["profile_id"],
                task["batch_id"],
            )
        self.assertEqual(matched["openai"], matched["deepseek"])

    def test_dry_run_progress_is_planned_not_stopping(self) -> None:
        profile = sample_profile()
        tasks = runner.build_tasks(
            [profile], ["qwen"], {"qwen": "qwen-test"}
        )
        tracker = runner.ProgressTracker(
            tasks, {}, {"qwen": "qwen-test"}, {"qwen": 1}, {"qwen": 4}
        )
        document = tracker.render(planned=True)
        self.assertIn("Run status: **PLANNED**", document)
        self.assertNotIn("Run status: **STOPPING**", document)


if __name__ == "__main__":
    unittest.main()
