# Signal or Spurious Cue: Experimental Code

This repository contains the implementation for the survey-country metadata audit, including experiment runners, provider adapters, checkpoint/recovery logic, statistical analysis, prompt bundles, data-construction utilities, automation scripts, and tests.

## Data and reproducibility

The Joint EVS/WVS source data, respondent-derived profiles, experiment manifests, model outputs, checkpoints, and generated reports are not included in this repository.

To reproduce a study run, obtain Joint EVS/WVS 2017-2022 v5.0 under its distributor's terms, construct the required profiles and manifests using the included tools, and provide API credentials through process environment variables.

## Environment

Python 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest `
  api_pilot\tests\test_strict_inference.py `
  dataset\tests\test_build_pilot_gate.py -q
```

The complete test suite also audits experiment manifests, respondent-derived
profiles, checkpoints, model outputs, and generated reports. Those integration
tests require additional datasets and generated fixtures that are not included
in this repository.

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

