# Signal or Spurious Cue?

**Research code for** *Signal or Spurious Cue? A Randomized Audit of Survey-Country Metadata in LLM Social Inference*

Yifan Lyu, Xinran Li, Jiaqi Qiao, and Xiujuan Xu (2026)  
Paper: [arXiv:2608.06085](https://arxiv.org/abs/2608.06085)

## Overview

This repository contains Python code for a controlled behavioral audit of how large language models use survey-country metadata when forecasting individual survey answers. The study separates **country-directed changes in predicted probabilities** from **consequences for predictive accuracy** (held-out Brier loss).

The within-record design compares four metadata conditions while holding the respondent evidence and target question fixed:

| Condition | Country metadata supplied to the model |
| --- | --- |
| `M0` | No country metadata |
| `MO` | A randomly assigned country label without its assignment mechanism disclosed |
| `MR` | The same type of random label, explicitly disclosed as uniformly assigned and independent of the respondent |
| `MV` | Verified survey-country metadata, with its provenance disclosed |

The accompanying preprint evaluates five fixed API models, six countries, and seven development-selected targets. Its principal finding is that disclosure did not reliably attenuate random-label uptake in the evaluated panels, whereas verified metadata improved held-out forecasting accuracy on those selected targets. These are **study-specific findings**, not general claims about all models or social-inference tasks.

## Repository contents

| Location | Role |
| --- | --- |
| `survey_audit/` | Descriptively named command-line interface for key workflows |
| `api_pilot/survey_pilot_runner.py` | Versioned forecast generation, API adapters, retries, checkpoints, and prompt validation |
| `api_pilot/strict_inference.py` | Paired estimands, survey-weighted aggregation, bootstrap intervals, and permutation procedures |
| `api_pilot/paired_inference.py` | Research-oriented import alias for the inference module |
| `api_pilot/api_smoke_test.py` | Synthetic API smoke tests (formerly `pilot_runner.py`) |
| `api_pilot/build_*.py` and `api_pilot/run_*.py` | Manifest construction and analysis-stage entry points |
| `api_pilot/analyze_*.py` | Diagnostic, sensitivity, and secondary analyses |
| `api_pilot/prompts/` | Versioned prompt templates defining the experimental conditions |
| `dataset/tools/` | Joint EVS/WVS validation and profile-building utilities |
| `api_pilot/tests/`, `dataset/tests/`, `tests/` | Unit and workflow-contract tests |

Several internal paths retain historical names such as `api_pilot`, `gate_c`, and `pilot`. These are preserved because experiment manifests, archived execution scripts, and recorded provenance refer to them. New public-facing commands use task-oriented names; see [Code organization and naming](docs/CODE_MAP.md).

## Installation and basic checks

Python 3.11 is recommended. From the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest api_pilot/tests/test_strict_inference.py api_pilot/tests/test_pilot_runner.py tests/test_cli.py -q
python -m survey_audit --help
```

The tests above use synthetic or programmatically constructed inputs; they do not reproduce the paper's full empirical findings.

## Named workflows

```bash
python -m survey_audit prepare-survey --help
python -m survey_audit api-smoke-test --help
python -m survey_audit build-confirmatory-manifest
python -m survey_audit run-forecasts --help
python -m survey_audit analyze-confirmatory --help
python -m survey_audit export-results --help
python -m survey_audit verify-completion --help
```

**Do not launch a full experiment solely from the commands above.** Manifest construction and confirmatory analysis require the study's designated local inputs, frozen task definitions, and outputs from preceding steps. Consult [Reproducibility and data requirements](docs/REPRODUCIBILITY.md) before running a study pipeline.

## Data availability and responsible use

The repository does **not** include licensed Joint EVS/WVS microdata, respondent-derived profiles, record-level task manifests, model response logs, checkpoints, or generated result tables. Obtain **Joint EVS/WVS 2017–2022, version 5.0** under the distributor's conditions and prepare local inputs with the included tools. Do not publish respondent-level derived records or API credentials.

API keys must be supplied through environment variables, including `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `GEMINI_API_KEY`, and `QWEN_API_KEY` where applicable. A dry run or a synthetic smoke test is not evidence of replication of the empirical results.

## Citation

If you use this repository or the study methodology, cite the accompanying preprint:

> Lyu, Y., Li, X., Qiao, J., & Xu, X. (2026). *Signal or Spurious Cue? A Randomized Audit of Survey-Country Metadata in LLM Social Inference*. arXiv:2608.06085.

BibTeX:

```bibtex
@misc{lyu2026signalspuriouscue,
  title         = {Signal or Spurious Cue? A Randomized Audit of Survey-Country Metadata in LLM Social Inference},
  author        = {Yifan Lyu and Xinran Li and Jiaqi Qiao and Xiujuan Xu},
  year          = {2026},
  eprint        = {2608.06085},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2608.06085}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable repository citation metadata.

## License

No software license has been assigned to this repository. Public visibility does not automatically authorize copying, modification, or redistribution. The survey data are separately subject to their distributor's terms.
