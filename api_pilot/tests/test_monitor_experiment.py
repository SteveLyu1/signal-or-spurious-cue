import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from api_pilot import monitor_experiment as monitor


class MonitorRendererTests(unittest.TestCase):
    def render_page(self, postprocess_html: str = "<p>后处理测试</p>") -> str:
        return monitor.render_page_v2(
            interval=600,
            overall_status="GATE C INFERENCE COMPLETE",
            overall_class="ok",
            updated_local="2026-07-25 12:00:00 CST",
            checkpoint_utc="2026-07-25 04:00:00 UTC",
            checkpoint_age=12.5,
            planned_total=4320,
            valid_total=4320,
            terminal_total=0,
            pending_total=0,
            provider_rows=["<tr><td>provider</td></tr>"],
            cell_rows=["<tr><td>M0</td></tr>"],
            checkpoint=Path("checkpoint.sqlite3"),
            active_overlays=[],
            postprocess_html=postprocess_html,
            generated_at=datetime(2026, 7, 25, 4, 0, tzinfo=timezone.utc),
        )

    def test_render_page_v2_returns_without_recursion(self) -> None:
        page = self.render_page()
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertEqual(page.count("<!doctype html>"), 1)

    def test_renderer_preserves_readable_chinese(self) -> None:
        page = self.render_page()
        self.assertIn("AAAI 实验推理监控", page)
        self.assertIn("推理完成不等于完整研究 Gate 通过", page)
        self.assertIn("监控页只读取聚合状态", page)
        self.assertNotIn("\ufffd", page)

    def test_render_distinguishes_inference_completion_from_full_gate(self) -> None:
        page = self.render_page(
            "<p><span class='badge warn'>完整 Gate 未开放</span></p>"
        )
        self.assertIn("GATE C INFERENCE COMPLETE", page)
        self.assertIn("完整 Gate 未开放", page)

    def test_postprocess_blocker_is_visible_and_distinct_from_inference(self) -> None:
        status = {
            "state": "complete",
            "audit": {
                "complete": True,
                "planned": 4320,
                "merged": 4320,
                "missing": 0,
                "incomplete": 0,
            },
            "gate": {
                "full_gate_c_pass": False,
                "outstanding_gate_components": {
                    "data_processing_permission": "written permission pending"
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            path.write_text(json.dumps(status), encoding="utf-8")
            section = monitor.render_postprocess_status(path)

        self.assertIn("Gate C 推理已完成", section)
        self.assertIn("完整 Gate 未开放", section)
        self.assertIn("data_processing_permission", section)
        self.assertIn("written permission pending", section)

    def test_postprocess_renders_permission_robustness_and_confirmatory_status(self) -> None:
        status = {
            "audit": {"complete": True, "planned": 4320, "merged": 4320},
            "gate": {"full_gate_c_pass": False, "blockers": []},
            "data_processing_permission": {"status": "pending", "passes": False},
            "robustness": {
                "status": "offline_manifests_frozen_inference_pending",
                "total_planned_calls": 6816,
                "experiments": {
                    "repeat_identical": {
                        "planned_calls": 288,
                        "analysis_status": "pending_inference",
                        "gate_status": "pending",
                    }
                },
            },
            "confirmatory_execution": {
                "planned_calls": 48384,
                "observed_successes": 0,
                "analysis_status": "pending_inference",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            path.write_text(json.dumps(status), encoding="utf-8")
            section = monitor.render_postprocess_status(path)
        self.assertIn("数据处理许可：pending", section)
        self.assertIn("repeat_identical", section)
        self.assertIn("48,384", section)
        self.assertIn("pending_inference", section)


if __name__ == "__main__":
    unittest.main()
