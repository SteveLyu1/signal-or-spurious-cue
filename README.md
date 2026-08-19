# Signal or Spurious Cue: Experimental Code

Private source repository for the survey-country metadata audit. It contains
the experiment runners, provider adapters, checkpoint/recovery logic,
statistical analysis code, prompt bundles, data-construction utilities,
automation scripts, and tests.

## Excluded material

This repository intentionally excludes author information, API credentials,
the Joint EVS/WVS source data, respondent-derived profiles, frozen task
manifests containing record-level task units, model outputs, checkpoints,
logs, generated reports, paper sources, and LaTeX build files. Those materials
must remain in separately controlled storage.

The code creates generated directories as needed. To reproduce a study run,
obtain Joint EVS/WVS 2017-2022 v5.0 under its distributor's terms, construct
the restricted profiles and manifests with the included tools, and provide
provider credentials through process environment variables only.

## Environment

Python 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest `
  api_pilot\tests\test_strict_inference.py `
  dataset\tests\test_build_pilot_gate.py -q
```

The complete test suite also audits frozen manifests, respondent-derived
profiles, checkpoints, model outputs, and generated reports. Those integration
tests require the separately controlled private fixtures and are not expected
to pass in this source-only repository.

Supported credential variables include `OPENAI_API_KEY`,
`DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `GEMINI_API_KEY`, and `QWEN_API_KEY`.
Comma-separated plural forms are also supported by the runner. Do not create
or commit a plaintext key file.

## Main components

- `api_pilot/survey_pilot_runner.py`: hosted inference, schema validation,
  checkpointing, retries, and provider routing.
- `api_pilot/strict_inference.py`: core statistical estimands and bootstrap
  procedures.
- `api_pilot/build_*.py`: frozen manifest construction.
- `api_pilot/analyze_*.py`: strict and diagnostic analyses.
- `api_pilot/scripts/`: Windows execution, monitoring, and recovery entrypoints.
- `dataset/tools/`: source validation and profile/manifest preparation.
- `api_pilot/tests/` and `dataset/tests/`: unit and pipeline tests.

No public code licence has been selected. The repository must remain private
until a licence and public-release boundary are approved.
