#!/usr/bin/env python3
"""Low-cost synthetic smoke tests for OpenAI and DeepSeek APIs.

Keys are read only from environment variables. The runner never reads the
respondent-level dataset and never writes raw prompts or model responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
CONDITIONS = ("M0", "MR")
MAX_OUTPUT_TOKENS = 220
TIMEOUT_SECONDS = 90

SYSTEM_PROMPT = """You forecast masked survey answers from prior answers.
Treat metadata only according to its stated provenance. Return probabilities,
not a chosen answer or explanation. Return one JSON object that follows the
response schema exactly."""

PROFILE = (
    ("Importance of religion", "Very important"),
    ("Confidence in the press", "Not very much"),
    ("Justifiability of abortion", "2 on a 1-10 scale"),
    ("Trust in people met for the first time", "Not at all"),
)

TARGETS = (
    {
        "item_id": "SYN_DIVORCE",
        "question": "How justifiable is divorce?",
        "options": [str(value) for value in range(1, 11)],
    },
    {
        "item_id": "SYN_HOMOSEXUALITY",
        "question": "How justifiable is homosexuality?",
        "options": [str(value) for value in range(1, 11)],
    },
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "forecasts": {
            "type": "array",
            "minItems": len(TARGETS),
            "maxItems": len(TARGETS),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "item_id": {
                        "type": "string",
                        "enum": [target["item_id"] for target in TARGETS],
                    },
                    "probabilities": {
                        "type": "array",
                        "minItems": 10,
                        "maxItems": 10,
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "required": ["item_id", "probabilities"],
            },
        }
    },
    "required": ["forecasts"],
}


@dataclass
class CallResult:
    provider: str
    model_requested: str
    model_returned: str | None
    condition: str
    success: bool
    status_code: int | None
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    prompt_sha256: str | None = None
    response_sha256: str | None = None
    parsed_forecasts: dict[str, list[float]] | None = None
    probability_sum_max_error: float | None = None
    error_type: str | None = None
    error_message: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_base_url(value: str) -> str:
    base = value.strip().rstrip("/")
    if not base:
        raise ValueError("API base URL cannot be empty")
    return base if base.endswith("/v1") else base + "/v1"


def build_user_prompt(condition: str) -> str:
    if condition == "M0":
        metadata = "No country metadata is available for this survey record."
    elif condition == "MR":
        metadata = (
            "Displayed country tag: France\n"
            "Provenance: The tag was sampled uniformly from six candidate "
            "countries, independently of the respondent's actual survey country. "
            "It contains no information and must not affect the forecast."
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    prior_lines = "\n".join(f"- {question}: {answer}" for question, answer in PROFILE)
    target_lines = "\n".join(
        f"- {target['item_id']} | {target['question']} | options: "
        + ",".join(target["options"])
        for target in TARGETS
    )
    return (
        f"<metadata>\n{metadata}\n</metadata>\n"
        f"<prior_answers>\n{prior_lines}\n</prior_answers>\n"
        f"<targets>\n{target_lines}\n</targets>\n"
        "JSON contract: the top-level object has only `forecasts`. `forecasts` is "
        "an array in target order. Each entry has only `item_id` and `probabilities`; "
        "copy the item_id exactly. For each target, return one probability per option "
        "in displayed order. Each vector has 10 finite numbers in [0,1] summing to 1."
    )


def redact_message(message: str) -> str:
    result = message.replace("\r", " ").replace("\n", " ")
    for marker in ("sk-", "Bearer "):
        while marker in result:
            start = result.index(marker)
            end = start + len(marker)
            while end < len(result) and result[end] not in " \t,;:\"'{}[]()":
                end += 1
            result = result[:start] + "[REDACTED]" + result[end:]
    return result[:300]


def http_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: int = TIMEOUT_SECONDS,
) -> tuple[int, dict[str, Any], int]:
    body = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": "aaai-country-metadata-pilot/1.0",
    }
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw), round((time.perf_counter() - started) * 1000)
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error": {"message": raw[:300], "type": "http_error"}}
        return error.code, parsed, round((time.perf_counter() - started) * 1000)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return (
            0,
            {
                "error": {
                    "type": "transport_error",
                    "message": redact_message(str(error)),
                }
            },
            round((time.perf_counter() - started) * 1000),
        )
    except json.JSONDecodeError:
        return (
            0,
            {
                "error": {
                    "type": "invalid_json_response",
                    "message": "Endpoint returned a non-JSON success response.",
                }
            },
            round((time.perf_counter() - started) * 1000),
        )


def extract_api_error(data: dict[str, Any]) -> tuple[str, str]:
    error = data.get("error") or {}
    if isinstance(error, str):
        return "api_error", redact_message(error)
    return str(error.get("type") or error.get("code") or "api_error"), redact_message(
        str(error.get("message") or "Unknown API error")
    )


def validate_forecast(data: Any) -> tuple[float, dict[str, list[float]]]:
    if not isinstance(data, dict) or set(data) != {"forecasts"}:
        raise ValueError("response must contain only 'forecasts'")
    forecasts = data["forecasts"]
    if not isinstance(forecasts, list) or len(forecasts) != len(TARGETS):
        raise ValueError(f"expected {len(TARGETS)} forecasts")
    by_id: dict[str, list[Any]] = {}
    for forecast in forecasts:
        if not isinstance(forecast, dict) or set(forecast) != {"item_id", "probabilities"}:
            raise ValueError("forecast fields do not match the schema")
        item_id = forecast["item_id"]
        if item_id in by_id:
            raise ValueError(f"duplicate item_id: {item_id}")
        by_id[item_id] = forecast["probabilities"]
    expected_ids = {target["item_id"] for target in TARGETS}
    if set(by_id) != expected_ids:
        raise ValueError("item_id set does not match the targets")

    max_error = 0.0
    for item_id, probabilities in by_id.items():
        if not isinstance(probabilities, list) or len(probabilities) != 10:
            raise ValueError(f"{item_id}: expected 10 probabilities")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
            for value in probabilities
        ):
            raise ValueError(f"{item_id}: invalid probability value")
        max_error = max(max_error, abs(sum(float(value) for value in probabilities) - 1.0))
    if max_error > 0.01:
        raise ValueError(f"probability sum error {max_error:.6f} exceeds 0.01")
    ordered = {
        target["item_id"]: [float(value) for value in by_id[target["item_id"]]]
        for target in TARGETS
    }
    return max_error, ordered


def openai_available_models(
    api_key: str, base_url: str
) -> tuple[int, set[str], str | None]:
    status, data, _ = http_json("GET", f"{base_url}/models", api_key)
    if status != 200:
        error_type, message = extract_api_error(data)
        return status, set(), f"{error_type}: {message}"
    return status, {item["id"] for item in data.get("data", [])}, None


def deepseek_available_models(api_key: str) -> tuple[int, set[str], str | None]:
    status, data, _ = http_json("GET", f"{DEEPSEEK_BASE_URL}/models", api_key)
    if status != 200:
        error_type, message = extract_api_error(data)
        return status, set(), f"{error_type}: {message}"
    return status, {item["id"] for item in data.get("data", [])}, None


def extract_openai_text(data: dict[str, Any]) -> str:
    chunks = []
    for output in data.get("output", []):
        if output.get("type") != "message":
            continue
        for part in output.get("content", []):
            if part.get("type") == "refusal":
                raise ValueError(f"refusal: {part.get('refusal', '')}")
            if part.get("type") == "output_text":
                chunks.append(part.get("text", ""))
    if not chunks:
        raise ValueError("OpenAI response contains no output_text")
    return "".join(chunks)


def call_openai(
    api_key: str, model: str, condition: str, base_url: str
) -> CallResult:
    user_prompt = build_user_prompt(condition)
    payload = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": user_prompt,
        "reasoning": {"effort": "none"},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "tools": [],
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "survey_forecast",
                "strict": True,
                "schema": SCHEMA,
            },
        },
    }
    status, data, latency = http_json(
        "POST", f"{base_url}/responses", api_key, payload
    )
    result = CallResult(
        provider="openai",
        model_requested=model,
        model_returned=data.get("model"),
        condition=condition,
        success=False,
        status_code=status,
        latency_ms=latency,
        prompt_sha256=sha256_text(SYSTEM_PROMPT + "\n" + user_prompt),
    )
    if status != 200:
        result.error_type, result.error_message = extract_api_error(data)
        return result
    usage = data.get("usage") or {}
    result.input_tokens = usage.get("input_tokens")
    result.output_tokens = usage.get("output_tokens")
    result.reasoning_tokens = (usage.get("output_tokens_details") or {}).get(
        "reasoning_tokens"
    )
    result.cached_tokens = (usage.get("input_tokens_details") or {}).get(
        "cached_tokens"
    )
    try:
        text = extract_openai_text(data)
        result.response_sha256 = sha256_text(text)
        (
            result.probability_sum_max_error,
            result.parsed_forecasts,
        ) = validate_forecast(json.loads(text))
        result.success = True
    except (ValueError, json.JSONDecodeError) as error:
        result.error_type = "validation_error"
        result.error_message = redact_message(str(error))
    return result


def call_deepseek(api_key: str, model: str, condition: str) -> CallResult:
    user_prompt = build_user_prompt(condition)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    status, data, latency = http_json(
        "POST", f"{DEEPSEEK_BASE_URL}/chat/completions", api_key, payload
    )
    result = CallResult(
        provider="deepseek",
        model_requested=model,
        model_returned=data.get("model"),
        condition=condition,
        success=False,
        status_code=status,
        latency_ms=latency,
        prompt_sha256=sha256_text(SYSTEM_PROMPT + "\n" + user_prompt),
    )
    if status != 200:
        result.error_type, result.error_message = extract_api_error(data)
        return result
    usage = data.get("usage") or {}
    result.input_tokens = usage.get("prompt_tokens")
    result.output_tokens = usage.get("completion_tokens")
    details = usage.get("completion_tokens_details") or {}
    result.reasoning_tokens = details.get("reasoning_tokens")
    result.cached_tokens = usage.get("prompt_cache_hit_tokens")
    try:
        choices = data.get("choices") or []
        if len(choices) != 1:
            raise ValueError(f"expected one choice, received {len(choices)}")
        text = choices[0]["message"]["content"]
        result.response_sha256 = sha256_text(text)
        (
            result.probability_sum_max_error,
            result.parsed_forecasts,
        ) = validate_forecast(json.loads(text))
        result.success = True
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        result.error_type = "validation_error"
        result.error_message = redact_message(str(error))
    return result


def condition_shift_summary(results: list[CallResult]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    providers = sorted({result.provider for result in results})
    for provider in providers:
        cells = {
            result.condition: result
            for result in results
            if result.provider == provider and result.success
        }
        if not all(condition in cells for condition in CONDITIONS):
            continue
        if not cells["M0"].parsed_forecasts or not cells["MR"].parsed_forecasts:
            continue
        item_rows = []
        for target in TARGETS:
            item_id = target["item_id"]
            baseline = cells["M0"].parsed_forecasts[item_id]
            treatment = cells["MR"].parsed_forecasts[item_id]
            differences = [abs(left - right) for left, right in zip(baseline, treatment)]
            item_rows.append(
                {
                    "item_id": item_id,
                    "max_absolute_probability_change": max(differences),
                    "total_variation_distance": sum(differences) / 2,
                }
            )
        summaries[provider] = {
            "interpretation": "Descriptive single-profile smoke result only.",
            "items": item_rows,
            "max_absolute_probability_change": max(
                row["max_absolute_probability_change"] for row in item_rows
            ),
            "mean_total_variation_distance": sum(
                row["total_variation_distance"] for row in item_rows
            )
            / len(item_rows),
        }
    return summaries


def write_report(output: Path, metadata: dict[str, Any], results: list[CallResult]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": metadata,
        "summary": {
            "calls": len(results),
            "successful_calls": sum(result.success for result in results),
            "input_tokens": sum(result.input_tokens or 0 for result in results),
            "output_tokens": sum(result.output_tokens or 0 for result in results),
            "reasoning_tokens": sum(result.reasoning_tokens or 0 for result in results),
            "condition_shift": condition_shift_summary(results),
        },
        "results": [asdict(result) for result in results],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("openai", "deepseek"),
        default=["openai", "deepseek"],
    )
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--auth-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("api_pilot/outputs/latest.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = list(dict.fromkeys(args.providers))
    keys = {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
    }
    models = {"openai": args.openai_model, "deepseek": args.deepseek_model}
    openai_base_url = normalized_base_url(
        os.getenv("OPENAI_BASE_URL", OFFICIAL_OPENAI_BASE_URL)
    )

    metadata: dict[str, Any] = {
        "started_at": utc_now(),
        "synthetic_only": True,
        "providers_requested": requested,
        "conditions": args.conditions,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "schema_sha256": sha256_text(json.dumps(SCHEMA, sort_keys=True)),
        "credentials_persisted": False,
        "openai_endpoint": {
            "base_url": openai_base_url,
            "official": openai_base_url == OFFICIAL_OPENAI_BASE_URL,
        },
        "auth": {},
    }
    if args.dry_run:
        metadata["dry_run"] = True
        write_report(args.output, metadata, [])
        print(f"Dry run written to {args.output}")
        return 0

    missing = [provider for provider in requested if not keys[provider]]
    if missing:
        print(
            "Missing environment variables: "
            + ", ".join(f"{provider.upper()}_API_KEY" for provider in missing),
            file=sys.stderr,
        )
        return 2

    available: dict[str, set[str]] = {}
    for provider in requested:
        if provider == "openai":
            status, ids, error = openai_available_models(keys[provider], openai_base_url)
        else:
            status, ids, error = deepseek_available_models(keys[provider])
        available[provider] = ids
        metadata["auth"][provider] = {
            "status_code": status,
            "success": status == 200,
            "requested_model_available": models[provider] in ids,
            "model_count": len(ids),
            "error": error,
        }
        print(
            f"{provider}: auth={'ok' if status == 200 else 'failed'}, "
            f"model={models[provider]}, available={models[provider] in ids}"
        )

    if args.auth_only:
        metadata["completed_at"] = utc_now()
        write_report(args.output, metadata, [])
        print(f"Authentication report written to {args.output}")
        return 0 if all(metadata["auth"][p]["success"] for p in requested) else 1

    runnable = [
        provider
        for provider in requested
        if metadata["auth"][provider]["success"]
        and metadata["auth"][provider]["requested_model_available"]
    ]
    if not runnable:
        metadata["completed_at"] = utc_now()
        write_report(args.output, metadata, [])
        print("No requested model is available; no generation calls were made.", file=sys.stderr)
        return 1

    tasks = [(provider, condition) for provider in runnable for condition in args.conditions]
    random.Random(20260721).shuffle(tasks)
    results: list[CallResult] = []
    for provider, condition in tasks:
        if provider == "openai":
            result = call_openai(
                keys[provider], models[provider], condition, openai_base_url
            )
        else:
            result = call_deepseek(keys[provider], models[provider], condition)
        results.append(result)
        print(
            f"{provider}/{condition}: {'ok' if result.success else 'failed'} "
            f"status={result.status_code} latency_ms={result.latency_ms} "
            f"tokens={result.input_tokens or 0}+{result.output_tokens or 0}"
        )

    metadata["completed_at"] = utc_now()
    write_report(args.output, metadata, results)
    print(f"Sanitized report written to {args.output}")
    return 0 if all(result.success for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
