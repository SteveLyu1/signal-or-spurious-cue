#!/usr/bin/env python3
"""Synchronise and evaluate the four five-model robustness panels offline."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __package__:
    from .analyze_robustness_outputs import analyse
    from .sync_five_model_panels import ROOT, main as sync_panels
else:
    # The automation supervisors execute this file by path. In that mode
    # Python adds api_pilot/, rather than the repository root, to sys.path.
    from analyze_robustness_outputs import analyse
    from sync_five_model_panels import ROOT, main as sync_panels


NAMES = ("repeat_identical", "single_item", "wording_b", "option_reverse")
STATUS = ROOT / "api_pilot/outputs/five_model_robustness_gate.json"


def analyse_panel(name: str, panel: dict) -> tuple[str, dict]:
    result = analyse(
        ROOT / panel["manifest"],
        ROOT / panel["output"],
        ROOT / "api_pilot/restricted_outputs/gate_c_v3_complete.jsonl",
        10_000,
    )
    return name, result


def main() -> int:
    sync_panels()
    sync = json.loads(
        (ROOT / "api_pilot/outputs/five_model_panel_sync.json").read_text(encoding="utf-8")
    )
    incomplete = [name for name in NAMES if not sync["panels"][name]["complete"]]
    if incomplete:
        experiments = {
            name: {
                "status": "pending" if name in incomplete else "ready",
                "passes": False,
            }
            for name in NAMES
        }
        STATUS.write_text(
            json.dumps(
                {
                    "status": "pending",
                    "complete": False,
                    "passes": False,
                    "waiting_for": incomplete,
                    "experiments": experiments,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "pending", "waiting_for": incomplete}))
        return 11
    experiments = {}
    with ProcessPoolExecutor(max_workers=len(NAMES)) as pool:
        analysed = dict(pool.map(
            analyse_panel,
            NAMES,
            (sync["panels"][name] for name in NAMES),
        ))
    for name in NAMES:
        result = analysed[name]
        report = ROOT / f"api_pilot/outputs/{name}_five_model_robustness_analysis.json"
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        experiments[name] = {
            "status": result["status"],
            "passes": bool((result.get("gate_decision") or {}).get("passes")),
            "report": report.relative_to(ROOT).as_posix(),
        }
    complete = all(row["status"] == "complete" for row in experiments.values())
    passes = complete and all(row["passes"] for row in experiments.values())
    status = "pass" if passes else ("fail" if complete else "pending")
    STATUS.write_text(
        json.dumps({"status": status, "complete": complete, "passes": passes, "experiments": experiments}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "experiments": experiments}))
    return 0 if passes else (14 if complete else 11)


if __name__ == "__main__":
    raise SystemExit(main())
