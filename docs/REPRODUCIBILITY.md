# Reproducibility and data requirements

## Scope of the release

This repository provides **source code and versioned prompts**, not a self-contained replication package. Full reproduction requires external licensed survey inputs, derived respondent profiles, frozen manifests, provider access, and run-specific outputs that are not committed here.

### Required inputs

1. Joint EVS/WVS 2017–2022, version 5.0, obtained through the authorized distributor.
2. The corresponding documentation/codebook required by `dataset/tools/prepare_joint_evs_wvs.py`.
3. Derived profiles, item metadata, sampling assignments, and frozen task manifests matching the stage being run.
4. Provider credentials and access to the model versions used for the selected experiment.
5. For confirmatory analysis, recorded item-level forecast probabilities plus reference-anchor and survey information matching the designated manifest.

Do not treat synthetic tests or incomplete re-executions as independent validation of reported empirical estimates.

## Environment

Use Python 3.11 and install the pinned Python dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest api_pilot/tests/test_strict_inference.py api_pilot/tests/test_pilot_runner.py tests/test_cli.py -q
```

Run the public CLI with `python -m survey_audit --help`. The command name describes the scientific task; the underlying module retains the original study-specific arguments.

## Workflow and provenance

| Stage | Relevant entry point | Notes |
| --- | --- | --- |
| Source validation | `prepare-survey` | Requires licensed CSV and codebook; creates local validation metadata |
| Experiment definition | `build-confirmatory-manifest` | Uses stage-specific pre-existing profile/assignment files; do not regenerate or replace a frozen manifest for comparison |
| Inference | `run-forecasts` | API calls with validation, checkpoint/recovery, and provider-specific settings |
| Primary statistical analysis | `analyze-confirmatory` | Checks the expected panel and builds design-aligned estimates; it does not run API inference |
| Presentation | `export-results` | Reads existing analysis outputs to construct result tables |
| Completion audit | `verify-completion` | Checks artifact and experiment completion requirements |

**Keep the exact experiment inputs and versioned prompt bundles fixed** when reproducing a result. Check model identifiers, profile selection, country-label assignment, answer-option order, random seeds, missingness handling, API settings, and checksum bindings before interpreting differences.

## Design and estimands

- `M0`: no country metadata.
- `MO`: an opaque random country label.
- `MR`: a random country label with its independent, uniform assignment mechanism disclosed.
- `MV`: a verified survey-country label with its origin disclosed.

The principal methodological distinction is between a **directional response to a label** (forecast movement relative to country reference distributions) and a **held-out outcome consequence** (proper-score differences evaluated against observed survey answers). Treat confidence intervals and the finite set of evaluated models, countries, and target questions as part of the result rather than as optional annotations.

## Data governance

Do not commit the Joint EVS/WVS microdata or respondent-derived records. The existing `.gitignore` excludes local source-data and generated-state directories and common credential files. Store the survey data and experimental outputs in access-controlled local storage, and use environment variables for API secrets.

## Limits and citation

The current repository has no released raw result corpus or complete fixture bundle. Some integration tests require those unavailable artifacts; a passing unit-test run is **not** a full experiment replication. For the publication's design, sample/panel definitions, exact statistics, and limitations, see [arXiv:2608.06085](https://arxiv.org/abs/2608.06085).
