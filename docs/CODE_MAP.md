# Code organization and naming

This research repository contains both **study workflow code** and **historical experiment artifacts**. Naming aims to distinguish stable public commands from phase-specific implementation details without silently changing their provenance.

## Public commands

| Descriptive command | Implementation | Purpose |
| --- | --- | --- |
| `python -m survey_audit prepare-survey` | `dataset.tools.prepare_joint_evs_wvs` | Validate licensed Joint EVS/WVS source and prepare metadata |
| `python -m survey_audit api-smoke-test` | `api_pilot.api_smoke_test` | Synthetic provider connectivity / forecast-schema checks |
| `python -m survey_audit build-confirmatory-manifest` | `api_pilot.build_confirmatory_core_manifest` | Construct the versioned confirmatory task manifest |
| `python -m survey_audit run-forecasts` | `api_pilot.survey_pilot_runner` | Issue and resume API-based probability forecasts |
| `python -m survey_audit analyze-confirmatory` | `api_pilot.run_confirmatory_strict_inference` | Analyze the frozen confirmatory panel |
| `python -m survey_audit export-results` | `api_pilot.export_final_results_tables` | Export report tables from validated analysis outputs |
| `python -m survey_audit verify-completion` | `api_pilot.verify_experiment_completion` | Audit expected study artifacts and completion contract |

Each command forwards subsequent flags unchanged to the underlying study entry point. Use `--help` where implemented to inspect the actual arguments. This interface is a naming layer, not a new experimental protocol.

## Filename conventions

- **`prepare_` / `build_`**: construct or validate source inputs and manifests.
- **`run_`**: execute a defined forecasting or statistical analysis stage.
- **`analyze_`**: compute diagnostics or secondary comparisons on existing outputs.
- **`verify_` / `validate_`**: inspect completion and consistency contracts.
- **`export_`**: produce publication-oriented summary artifacts.
- **`api_smoke_test.py`**: canonical name for the synthetic API test; `pilot_runner.py` is retained as a backward-compatible import and command shim.
- **`paired_inference.py`**: descriptive import surface for the original `strict_inference.py` estimators; the historical source path remains unchanged because some analysis provenance hashes that file.

## Why the historical paths remain

The `api_pilot/` directory contains **pilot, confirmatory, corrective, replication, and secondary** stages; its directory name is historical and should not be interpreted as limiting the repo to a pilot study. Likewise, `gate_c`, manifest version identifiers, output basenames, and prompt bundle identifiers are part of the recorded experiment lineage. Mass-renaming them would break command paths, manifest references, and/or checksum comparisons.

New code should use descriptive task-based names. Do not retroactively rename frozen conditions (`M0`, `MO`, `MR`, `MV`), prompts, manifests, or output hashes merely for cosmetic consistency.

## Experimental scope

Primary confirmatory estimates, corrected-panel results, robustness analyses, and diagnostics are not interchangeable. Consult the paper to determine which panel and estimand support a reported result. Archived PowerShell scripts are retained to document the original execution context rather than to serve as the default public entry point.
