#!/usr/bin/env python3
"""Run the restricted six-country survey pilot with resumable API calls."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_OPENAI_MODEL = "gpt-5.4-2026-03-05"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_MISTRAL_MODEL = "mistral-medium-3-5"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_QWEN_MODEL = "qwen3.7-max-2026-06-08"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_OUTPUT_TOKENS = 512
TIMEOUT_SECONDS = 120
REQUEST_PROTOCOL_VERSION = "survey-api-v4"
PROVIDERS = ("openai", "deepseek", "mistral", "gemini", "qwen")
# Google and Mistral document 408, 429, transport failures, and 5xx responses
# as transient. Other 4xx responses are configuration/authentication failures
# and should return immediately instead of consuming the retry budget.
RETRIABLE_STATUS_CODES = {0, 408, 429, 500, 502, 503, 504}
COUNTRIES = ("CN", "US", "JO", "FR", "GB", "IT")
COUNTRY_LABELS = {
    "CN": "China",
    "US": "United States",
    "JO": "Jordan",
    "FR": "France",
    "GB": "Great Britain",
    "IT": "Italy",
}
LABEL_TO_COUNTRY = {label: code for code, label in COUNTRY_LABELS.items()}
DEFAULT_PROMPT_BUNDLE_PATH = (
    Path(__file__).resolve().parent / "prompts" / "survey-confirmatory-v3.json"
)
ROOT = Path(__file__).resolve().parents[1]
CELLS = (
    ("M0", "correct"),
    ("MO", "correct"),
    ("MR", "correct"),
    ("MV", "correct"),
    ("M0", "shuffled"),
    ("MR", "shuffled"),
)


def load_prompt_bundle(path: Path) -> dict[str, Any]:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "version",
        "status",
        "system_prompt",
        "user_template",
        "conditions",
        "render",
        "items",
    }
    optional = {"frozen_at", "amendment"}
    unexpected = set(bundle).difference(required | optional)
    missing = required.difference(bundle)
    if missing or unexpected:
        raise ValueError(
            "Prompt bundle fields are invalid; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if "frozen_at" in bundle and not isinstance(bundle["frozen_at"], str):
        raise ValueError("Prompt bundle frozen_at must be a string")
    if "amendment" in bundle and not isinstance(bundle["amendment"], dict):
        raise ValueError("Prompt bundle amendment must be an object")
    if set(bundle["conditions"]) != {"M0", "MO", "MR", "MV"}:
        raise ValueError("Prompt bundle must define M0, MO, MR, and MV")
    if not any(
        marker in bundle["user_template"]
        for marker in ("<<METADATA>>", "<<METADATA_BLOCK>>")
    ):
        raise ValueError(
            "Prompt bundle user template is missing <<METADATA>> or "
            "<<METADATA_BLOCK>>"
        )
    for marker in (
        "<<PRIOR_ANSWERS>>",
        "<<MASKED_TARGETS>>",
        "<<CONTRACT>>",
    ):
        if marker not in bundle["user_template"]:
            raise ValueError(f"Prompt bundle user template is missing {marker}")
    return bundle


PROMPT_BUNDLE = load_prompt_bundle(DEFAULT_PROMPT_BUNDLE_PATH)
SYSTEM_PROMPT = PROMPT_BUNDLE["system_prompt"]
QUESTION_TEXT = {
    item_id: item["question"] for item_id, item in PROMPT_BUNDLE["items"].items()
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_base_url(value: str) -> str:
    base = value.strip().rstrip("/")
    if not base:
        raise ValueError("API base URL cannot be empty")
    return base if base.endswith("/v1") else base + "/v1"


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


KEY_LABELS = {
    "deepseek": "deepseek",
    "gpt": "openai",
    "openai": "openai",
    "mistral": "mistral",
    "gemini": "gemini",
    "qwen": "qwen",
}


def load_key_file(path: Path | None) -> dict[str, list[str]]:
    """Load labelled credentials without persisting them in run artifacts."""
    keys: dict[str, list[str]] = defaultdict(list)
    if path is None or not path.exists():
        return dict(keys)
    waiting_for: str | None = None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        normalized = line.rstrip(":：").strip().lower()
        if normalized in KEY_LABELS:
            waiting_for = KEY_LABELS[normalized]
            continue
        if waiting_for is None:
            continue
        if line.startswith(("http://", "https://")) or any(
            marker in line for marker in ("(", ")", "{", "}", "=", " ")
        ):
            continue
        keys[waiting_for].append(line.strip("\"'"))
        waiting_for = None
    return dict(keys)


def configured_keys(path: Path | None = None) -> dict[str, list[str]]:
    keys = load_key_file(path)
    for provider in PROVIDERS:
        plural = os.getenv(f"{provider.upper()}_API_KEYS", "")
        singular = os.getenv(f"{provider.upper()}_API_KEY", "")
        if provider == "openai":
            plural = os.getenv("OPENAI_API_KEYS", plural)
            singular = os.getenv("OPENAI_API_KEY", singular)
        environment_keys = [
            value.strip()
            for value in re.split(r"[,;\n]", plural)
            if value.strip()
        ]
        if singular.strip():
            environment_keys.insert(0, singular.strip())
        if environment_keys:
            keys[provider] = list(dict.fromkeys(environment_keys))
        elif provider in keys:
            keys[provider] = list(dict.fromkeys(keys[provider]))
    return keys


def choose_key_index(
    available: list[int], cursor: int, previous_result: dict[str, Any] | None
) -> int:
    avoid = None
    if (
        previous_result
        and not previous_result.get("success")
        and previous_result.get("credential_slot")
    ):
        avoid = int(previous_result["credential_slot"]) - 1
    preferred = [index for index in available if index != avoid]
    candidates = preferred or available
    return candidates[cursor % len(candidates)]


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class CheckpointRunLock:
    """Process-scoped exclusive lock for one checkpoint.

    The lock file is intentionally persistent.  The operating system releases
    the byte-range lock if a runner crashes, so a later runner can safely
    overwrite stale PID metadata without deleting a pathname another process
    may still have open.
    """

    def __init__(self, checkpoint_path: Path):
        self.path = checkpoint_path.with_name(checkpoint_path.name + ".runner.lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        self.handle = os.fdopen(descriptor, "r+b", buffering=0)
        self.token = uuid.uuid4().hex
        self._released = False
        try:
            self.handle.seek(0)
            previous_owner = self.handle.read().decode(
                "utf-8", errors="replace"
            ).strip()
        except PermissionError:
            # Windows denies reads that overlap another process's locked byte.
            # The following lock attempt remains the authoritative liveness test.
            previous_owner = "unavailable while locked"
        try:
            if os.fstat(descriptor).st_size == 0:
                self.handle.seek(0)
                self.handle.write(b"\0")
            self.handle.seek(0)
            self._lock_nonblocking(previous_owner)
        except BaseException:
            self.handle.close()
            raise
        metadata = {
            "pid": os.getpid(),
            "process_token": self.token,
            "acquired_at": utc_now(),
            "checkpoint": str(checkpoint_path.resolve()),
        }
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(
            (json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        )
        os.fsync(descriptor)

    def _lock_nonblocking(self, previous_owner: str) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
        except (OSError, BlockingIOError) as error:
            raise RuntimeError(
                "Checkpoint already has an active runner: "
                f"{self.path}; owner={previous_owner or 'unavailable'}"
            ) from error

    def release(self) -> None:
        if self._released:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self._released = True


class CheckpointStore:
    """SQLite WAL checkpoint; JSONL remains the portable audit export."""

    def __init__(self, path: Path, run_signature: str, run_metadata: dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.run_lock = CheckpointRunLock(path)
        try:
            self.connection = sqlite3.connect(path)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS run_meta "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS results ("
                "call_id TEXT PRIMARY KEY, provider TEXT NOT NULL, success INTEGER NOT NULL, "
                "result_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS result_history ("
                "history_id INTEGER PRIMARY KEY AUTOINCREMENT, call_id TEXT NOT NULL, "
                "provider TEXT NOT NULL, result_json TEXT NOT NULL, recorded_at TEXT NOT NULL)"
            )
            previous = self.connection.execute(
                "SELECT value FROM run_meta WHERE key='run_signature'"
            ).fetchone()
            if previous and previous[0] != run_signature:
                result_count = self.connection.execute(
                    "SELECT COUNT(*) FROM results"
                ).fetchone()[0]
                history_count = self.connection.execute(
                    "SELECT COUNT(*) FROM result_history"
                ).fetchone()[0]
                if result_count or history_count:
                    raise ValueError(
                        "Checkpoint belongs to a different frozen run; choose a new output path"
                    )
                # Dry runs may leave an empty checkpoint. A runner-code update can
                # safely rebind that empty shell, but never a checkpoint with calls.
                self.connection.execute(
                    "UPDATE run_meta SET value=? WHERE key='run_signature'",
                    (run_signature,),
                )
                self.connection.execute(
                    "UPDATE run_meta SET value=? WHERE key='run_metadata'",
                    (json.dumps(run_metadata, ensure_ascii=False, sort_keys=True),),
                )
            self.connection.execute(
                "INSERT OR IGNORE INTO run_meta(key, value) VALUES('run_signature', ?)",
                (run_signature,),
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO run_meta(key, value) VALUES('run_metadata', ?)",
                (json.dumps(run_metadata, ensure_ascii=False, sort_keys=True),),
            )
            self.connection.commit()
        except BaseException:
            if hasattr(self, "connection"):
                self.connection.close()
            self.run_lock.release()
            raise

    def import_jsonl(self, path: Path, allowed_call_ids: set[str] | None = None) -> None:
        if not path.exists() or self.connection.execute(
            "SELECT COUNT(*) FROM results"
        ).fetchone()[0]:
            return
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        for index, raw_line in enumerate(lines):
            if not raw_line.strip():
                continue
            try:
                result = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                is_tail = index == len(lines) - 1 and not raw_line.endswith((b"\n", b"\r"))
                if is_tail:
                    print(
                        f"warning: ignored truncated checkpoint tail in {path}",
                        file=sys.stderr,
                    )
                    break
                raise ValueError(f"Invalid checkpoint line {index + 1}: {error}") from error
            if allowed_call_ids is not None and result.get("call_id") not in allowed_call_ids:
                raise ValueError(
                    "JSONL result does not belong to the frozen run: "
                    + str(result.get("call_id"))
                )
            self.record(result)

    def record(self, result: dict[str, Any]) -> None:
        serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        recorded_at = utc_now()
        self.connection.execute(
            "INSERT INTO result_history(call_id, provider, result_json, recorded_at) "
            "VALUES (?, ?, ?, ?)",
            (result["call_id"], result["provider"], serialized, recorded_at),
        )
        self.connection.execute(
            "INSERT INTO results "
            "(call_id, provider, success, result_json, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(call_id) DO UPDATE SET "
            "provider=excluded.provider, success=excluded.success, "
            "result_json=excluded.result_json, updated_at=excluded.updated_at "
            "WHERE results.success=0 OR excluded.success=1",
            (
                result["call_id"],
                result["provider"],
                int(bool(result.get("success"))),
                serialized,
                recorded_at,
            ),
        )
        self.connection.commit()

    def latest_results(self) -> dict[str, dict[str, Any]]:
        return {
            call_id: json.loads(result_json)
            for call_id, result_json in self.connection.execute(
                "SELECT call_id, result_json FROM results"
            )
        }

    def successful_ids(self) -> set[str]:
        return {
            row[0]
            for row in self.connection.execute(
                "SELECT call_id FROM results WHERE success=1"
            )
        }

    def completed_ids(self) -> set[str]:
        """Return valid calls only; invalid outputs remain resumable.

        Failed responses remain in append-only history, while a later
        automation cycle may request the same frozen call again.
        """
        return self.successful_ids()

    def export_jsonl(self, path: Path) -> None:
        rows = [
            json.loads(row[0])
            for row in self.connection.execute(
                "SELECT result_json FROM results ORDER BY provider, call_id"
            )
        ]
        atomic_write_text(
            path,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        )

    def export_history_jsonl(self, path: Path) -> None:
        rows = [
            json.loads(row[0])
            for row in self.connection.execute(
                "SELECT result_json FROM result_history ORDER BY history_id"
            )
        ]
        atomic_write_text(
            path,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        )

    def close(self) -> None:
        try:
            self.connection.close()
        finally:
            self.run_lock.release()


def http_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any], int]:
    body = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": "aaai-country-metadata-pilot/2.0",
    }
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {
                    "error": {
                        "type": "response_decode_error",
                        "message": "HTTP response was not valid JSON",
                    }
                }
            return response.status, parsed, round((time.perf_counter() - started) * 1000)
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error": {"type": "http_error", "message": raw[:300]}}
        retry_after = error.headers.get("Retry-After") if error.headers else None
        if retry_after:
            try:
                parsed["_retry_after_seconds"] = min(float(retry_after), 300.0)
            except ValueError:
                pass
        return error.code, parsed, round((time.perf_counter() - started) * 1000)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return 0, {"error": {"type": "transport_error", "message": redact_message(str(error))}}, round((time.perf_counter() - started) * 1000)


def extract_error(data: dict[str, Any]) -> tuple[str, str]:
    error = data.get("error") or {}
    if isinstance(error, str):
        return "api_error", redact_message(error)
    return str(error.get("type") or error.get("code") or "api_error"), redact_message(
        str(error.get("message") or "Unknown API error")
    )


def is_credential_or_quota_error(result: dict[str, Any]) -> bool:
    """Return true for errors that cannot improve by submitting more calls now."""
    if result.get("status_code") in {401, 402, 403}:
        return True
    error_type = str(result.get("error_type") or "").lower()
    message = str(result.get("error_message") or "").lower()
    markers = (
        "arrearage", "insufficient_quota", "insufficient balance",
        "overdue-payment", "account is in good standing",
        "prepayment credits are depleted", "api key not valid",
        "invalid api key", "invalid_api_key",
    )
    return any(marker in error_type or marker in message for marker in markers)


def extract_openai_text(data: dict[str, Any]) -> str:
    chunks = []
    for output in data.get("output", []):
        if output.get("type") != "message":
            continue
        for part in output.get("content", []):
            if part.get("type") == "refusal":
                raise ValueError("model refusal")
            if part.get("type") == "output_text":
                chunks.append(part.get("text", ""))
    if not chunks:
        raise ValueError("response contains no output_text")
    return "".join(chunks)


def load_profiles(path: Path, max_per_country: int) -> list[dict[str, Any]]:
    profiles = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    selected = []
    counts: Counter[str] = Counter()
    for profile in profiles:
        country = profile["true_country_code"]
        if country not in COUNTRIES or counts[country] >= max_per_country:
            continue
        if any(key in profile for key in ("uniqid", "respondent_id")):
            raise ValueError("Restricted profile unexpectedly contains a source identifier")
        selected.append(profile)
        counts[country] += 1
    if any(counts[country] != max_per_country for country in COUNTRIES):
        raise ValueError(f"Unbalanced profile selection: {dict(counts)}")
    return selected


def target_spec(
    profile: dict[str, Any], bundle: dict[str, Any] = PROMPT_BUNDLE
) -> list[dict[str, Any]]:
    return [
        {
            "item_id": target["item_id"],
            "question": bundle["items"][target["item_id"]]["question"],
            "options": [
                option["answer"]
                for option in profile["target_options"][target["item_id"]]
            ],
            "option_codes": [
                option["answer_code"]
                for option in profile["target_options"][target["item_id"]]
            ],
        }
        for target in profile["targets"]
    ]


def response_schema(targets: list[dict[str, Any]]) -> dict[str, Any]:
    properties = {}
    for target in targets:
        properties[target["item_id"]] = {
            "type": "array",
            "minItems": len(target["options"]),
            "maxItems": len(target["options"]),
            "items": {"type": "number", "minimum": 0, "maximum": 1},
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "forecasts": {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                "required": [target["item_id"] for target in targets],
            }
        },
        "required": ["forecasts"],
    }


def render_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"<<{key}>>", value)
    if "<<" in rendered or ">>" in rendered:
        raise ValueError("Prompt template contains an unresolved placeholder")
    return rendered


def metadata_text(
    profile: dict[str, Any],
    condition: str,
    bundle: dict[str, Any] = PROMPT_BUNDLE,
) -> str:
    if condition in {"M0", "M0_REPEAT"}:
        condition = "M0"
    if condition not in bundle["conditions"]:
        raise ValueError(f"Unknown condition: {condition}")
    return render_template(
        bundle["conditions"][condition],
        {
            "RANDOM_COUNTRY": profile["balanced_random_country_label"],
            "TRUE_COUNTRY": profile["true_country_label"],
            "CANDIDATE_COUNTRIES": ", ".join(
                COUNTRY_LABELS[code] for code in COUNTRIES
            ),
        },
    )


def build_prompt(
    profile: dict[str, Any],
    condition: str,
    evidence_variant: str,
    bundle: dict[str, Any] = PROMPT_BUNDLE,
) -> tuple[str, list[dict[str, Any]]]:
    evidence_key = "evidence" if evidence_variant == "correct" else "shuffled_evidence"
    prior_lines_list = []
    for item in profile[evidence_key]:
        item_id = item["item_id"]
        question = bundle["items"][item_id]["question"]
        if bundle["render"]["include_evidence_code"]:
            line = (
                f"- {item_id}: {question} Recorded answer: "
                f"[{item['answer_code']}] {item['answer']}."
            )
        else:
            line = f"- {question} Answer: {item['answer']}"
        scale_mode = bundle["render"]["evidence_scale_mode"]
        include_scale = scale_mode == "all" or (
            scale_mode == "numeric_only"
            and bundle["items"][item_id].get("numeric_scale", False)
        )
        if include_scale:
            line += f" Scale: {bundle['items'][item_id]['scale']}"
        prior_lines_list.append(line)
    prior_lines = "\n".join(prior_lines_list)

    targets = target_spec(profile, bundle)
    target_lines_list = []
    for target in targets:
        if bundle["render"]["include_target_codes"]:
            options = " | ".join(
                f"[{code}] {label}"
                for code, label in zip(target["option_codes"], target["options"])
            )
            target_lines_list.append(
                f"- {target['item_id']}: {target['question']} "
                f"Response options in order: {options}"
            )
        else:
            target_lines_list.append(
                f"- {target['item_id']}: {target['question']} Options in order: "
                + " | ".join(target["options"])
            )
    target_lines = "\n".join(target_lines_list)
    contract = ", ".join(
        f"{target['item_id']}={len(target['options'])} probabilities"
        for target in targets
    )
    metadata = metadata_text(profile, condition, bundle)
    metadata_block = (
        f"<metadata>\n{metadata}\n</metadata>\n" if metadata else ""
    )
    prompt = render_template(
        bundle["user_template"],
        {
            "METADATA": metadata,
            "METADATA_BLOCK": metadata_block,
            "PRIOR_ANSWERS": prior_lines,
            "MASKED_TARGETS": target_lines,
            "TARGET_COUNT": str(len(targets)),
            "CONTRACT": contract,
        },
    )
    return prompt, targets


def validate_forecast(data: Any, targets: list[dict[str, Any]]) -> tuple[dict[str, list[float]], float]:
    if not isinstance(data, dict) or set(data) != {"forecasts"}:
        raise ValueError("response must contain only forecasts")
    forecasts = data["forecasts"]
    expected = {target["item_id"] for target in targets}
    if not isinstance(forecasts, dict) or set(forecasts) != expected:
        raise ValueError("forecast item IDs do not match targets")
    parsed = {}
    max_error = 0.0
    for target in targets:
        item_id = target["item_id"]
        values = forecasts[item_id]
        if not isinstance(values, list) or len(values) != len(target["options"]):
            raise ValueError(f"{item_id}: wrong probability count")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
            for value in values
        ):
            raise ValueError(f"{item_id}: invalid probability")
        error = abs(sum(float(value) for value in values) - 1)
        if error > 0.01:
            raise ValueError(f"{item_id}: probability sum error {error:.6f}")
        max_error = max(max_error, error)
        parsed[item_id] = [float(value) for value in values]
    return parsed, max_error


def repair_mistral_probability_sums(
    data: Any,
    targets: list[dict[str, Any]],
    maximum_sum_error: float = 0.10,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Normalize otherwise valid Mistral vectors with bounded sum errors."""
    if not isinstance(data, dict) or set(data) != {"forecasts"}:
        raise ValueError("response must contain only forecasts")
    forecasts = data["forecasts"]
    expected = {target["item_id"] for target in targets}
    if not isinstance(forecasts, dict) or set(forecasts) != expected:
        raise ValueError("forecast item IDs do not match targets")
    parsed: dict[str, list[float]] = {}
    original_sums: dict[str, float] = {}
    for target in targets:
        item_id = target["item_id"]
        values = forecasts[item_id]
        if not isinstance(values, list) or len(values) != len(target["options"]):
            raise ValueError(f"{item_id}: wrong probability count")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
            for value in values
        ):
            raise ValueError(f"{item_id}: invalid probability")
        total = math.fsum(float(value) for value in values)
        if total <= 0 or abs(total - 1.0) > maximum_sum_error + 1e-9:
            raise ValueError(f"{item_id}: probability sum error {abs(total - 1.0):.6f}")
        original_sums[item_id] = total
        parsed[item_id] = [float(value) / total for value in values]
    return parsed, original_sums


def make_call_id(
    provider: str,
    model: str,
    profile_id: str,
    condition: str,
    evidence_variant: str,
    prompt_version: str,
    batch_id: str | None = None,
    prompt_digest: str | None = None,
) -> str:
    fields = [provider, model, profile_id, condition, evidence_variant]
    if prompt_version != "survey-pilot-v1":
        fields.insert(2, prompt_version)
    if batch_id is not None:
        profile_index = 3 if prompt_version != "survey-pilot-v1" else 2
        fields.insert(profile_index + 1, batch_id)
    if prompt_digest is not None:
        fields.append(prompt_digest[:12])
    return "/".join(fields)


def base_result(task: dict[str, Any], latency: int, status: int, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "call_id": task["call_id"],
        "provider": task["provider"],
        "model_requested": task["model"],
        "model_returned": data.get("model"),
        "prompt_version": task["prompt_version"],
        "profile_id": task["profile"]["profile_id"],
        "true_country_code": task["profile"]["true_country_code"],
        "condition": task["condition"],
        "evidence_variant": task["evidence_variant"],
        "experiment": task.get("experiment", "core"),
        "source_call_id": task.get("source_call_id"),
        "batch_id": task.get("batch_id"),
        "target_ids": [target["item_id"] for target in task["targets"]],
        "target_option_codes": {
            target["item_id"]: target["option_codes"] for target in task["targets"]
        },
        "displayed_country_code": (
            task["profile"]["true_country_code"]
            if task["condition"] == "MV"
            else LABEL_TO_COUNTRY.get(task["profile"]["balanced_random_country_label"])
            if task["condition"] in {"MO", "MR"}
            else None
        ),
        "success": False,
        "status_code": status,
        "latency_ms": latency,
        "attempts": task["attempt"],
        "prompt_sha256": sha256_text(
            task["system_prompt"] + "\n" + task["prompt"]
        ),
        "response_sha256": None,
        "raw_response_text": None,
        "request_id": data.get("id"),
        "parsed_forecasts": None,
        "probability_sum_max_error": None,
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "cached_tokens": None,
        "error_type": None,
        "error_message": None,
        "retry_after_seconds": data.get("_retry_after_seconds"),
        "terminal_failure": False,
    }


def call_openai(task: dict[str, Any], api_key: str, base_url: str) -> dict[str, Any]:
    schema = response_schema(task["targets"])
    payload = {
        "model": task["model"],
        "instructions": task["system_prompt"],
        "input": task["prompt"],
        "reasoning": {"effort": "none"},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "tools": [],
        "text": {
            "verbosity": "low",
            "format": {"type": "json_schema", "name": "survey_forecast", "strict": True, "schema": schema},
        },
    }
    status, data, latency = http_json("POST", f"{base_url}/responses", api_key, payload)
    result = base_result(task, latency, status, data)
    if status != 200:
        result["error_type"], result["error_message"] = extract_error(data)
        return result
    usage = data.get("usage") or {}
    result["input_tokens"] = usage.get("input_tokens")
    result["output_tokens"] = usage.get("output_tokens")
    result["reasoning_tokens"] = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    result["cached_tokens"] = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    try:
        text = extract_openai_text(data)
        result["raw_response_text"] = text
        result["response_sha256"] = sha256_text(text)
        result["parsed_forecasts"], result["probability_sum_max_error"] = validate_forecast(json.loads(text), task["targets"])
        result["success"] = True
    except (ValueError, json.JSONDecodeError) as error:
        result["error_type"] = "validation_error"
        result["error_message"] = redact_message(str(error))
    return result


def call_deepseek(task: dict[str, Any], api_key: str) -> dict[str, Any]:
    payload = {
        "model": task["model"],
        "messages": [
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": task["prompt"]},
        ],
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    status, data, latency = http_json("POST", f"{DEEPSEEK_BASE_URL}/chat/completions", api_key, payload)
    result = base_result(task, latency, status, data)
    if status != 200:
        result["error_type"], result["error_message"] = extract_error(data)
        return result
    usage = data.get("usage") or {}
    result["input_tokens"] = usage.get("prompt_tokens")
    result["output_tokens"] = usage.get("completion_tokens")
    result["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    result["cached_tokens"] = usage.get("prompt_cache_hit_tokens")
    try:
        choices = data.get("choices") or []
        if len(choices) != 1:
            raise ValueError("response choice count is not one")
        text = choices[0]["message"]["content"]
        result["raw_response_text"] = text
        result["response_sha256"] = sha256_text(text)
        result["parsed_forecasts"], result["probability_sum_max_error"] = validate_forecast(json.loads(text), task["targets"])
        result["success"] = True
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        result["error_type"] = "validation_error"
        result["error_message"] = redact_message(str(error))
    return result


def call_mistral(task: dict[str, Any], api_key: str) -> dict[str, Any]:
    schema = response_schema(task["targets"])
    payload = {
        "model": task["model"],
        "messages": [
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": task["prompt"]},
        ],
        "temperature": 0,
        "reasoning_effort": "none",
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "survey_forecast",
                "schema": schema,
                "strict": True,
            },
        },
        "stream": False,
    }
    status, data, latency = http_json(
        "POST", f"{MISTRAL_BASE_URL}/chat/completions", api_key, payload
    )
    result = base_result(task, latency, status, data)
    if status != 200:
        result["error_type"], result["error_message"] = extract_error(data)
        return result
    usage = data.get("usage") or {}
    result["input_tokens"] = usage.get("prompt_tokens")
    result["output_tokens"] = usage.get("completion_tokens")
    text: str | None = None
    try:
        choices = data.get("choices") or []
        if len(choices) != 1:
            raise ValueError("response choice count is not one")
        text = choices[0]["message"]["content"]
        result["raw_response_text"] = text
        result["response_sha256"] = sha256_text(text)
        result["parsed_forecasts"], result["probability_sum_max_error"] = (
            validate_forecast(json.loads(text), task["targets"])
        )
        result["success"] = True
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        try:
            if text is None:
                raise ValueError("missing response text")
            parsed, original_sums = repair_mistral_probability_sums(
                json.loads(text), task["targets"]
            )
            result["parsed_forecasts"] = parsed
            result["probability_sum_max_error"] = max(
                abs(total - 1.0) for total in original_sums.values()
            )
            result["postprocess_repair"] = {
                "method": "divide_each_probability_by_item_sum",
                "scope": "mistral_http_200_bounded_sum_error",
                "maximum_allowed_sum_error": 0.10,
                "original_sums": original_sums,
                "source_response_sha256": result["response_sha256"],
                "preserves_raw_response_text": True,
            }
            result["success"] = True
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            result["error_type"] = "validation_error"
            result["error_message"] = redact_message(str(error))
    return result


def call_qwen(task: dict[str, Any], api_key: str) -> dict[str, Any]:
    payload = {
        "model": task["model"],
        "messages": [
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": task["prompt"]},
        ],
        "enable_thinking": False,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    status, data, latency = http_json(
        "POST", f"{QWEN_BASE_URL}/chat/completions", api_key, payload
    )
    result = base_result(task, latency, status, data)
    if status != 200:
        result["error_type"], result["error_message"] = extract_error(data)
        return result
    usage = data.get("usage") or {}
    result["input_tokens"] = usage.get("prompt_tokens")
    result["output_tokens"] = usage.get("completion_tokens")
    result["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get(
        "reasoning_tokens"
    )
    try:
        choices = data.get("choices") or []
        if len(choices) != 1:
            raise ValueError("response choice count is not one")
        text = choices[0]["message"]["content"]
        result["raw_response_text"] = text
        result["response_sha256"] = sha256_text(text)
        result["parsed_forecasts"], result["probability_sum_max_error"] = (
            validate_forecast(json.loads(text), task["targets"])
        )
        result["success"] = True
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        result["error_type"] = "validation_error"
        result["error_message"] = redact_message(str(error))
    return result


def call_gemini(task: dict[str, Any], api_key: str) -> dict[str, Any]:
    payload = {
        "systemInstruction": {"parts": [{"text": task["system_prompt"]}]},
        "contents": [{"role": "user", "parts": [{"text": task["prompt"]}]}],
        "generationConfig": {
            "thinkingConfig": {"thinkingLevel": "minimal"},
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
            "responseJsonSchema": response_schema(task["targets"]),
        },
    }
    url = f"{GEMINI_BASE_URL}/models/{task['model']}:generateContent"
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "X-goog-api-key": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "aaai-country-metadata-pilot/3.0",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            status = response.status
            raw = response.read().decode("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {
                    "error": {
                        "type": "response_decode_error",
                        "message": "HTTP response was not valid JSON",
                    }
                }
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"error": {"type": "http_error", "message": raw[:300]}}
        retry_after = error.headers.get("Retry-After") if error.headers else None
        if retry_after:
            try:
                data["_retry_after_seconds"] = min(float(retry_after), 300.0)
            except ValueError:
                pass
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        status = 0
        data = {
            "error": {
                "type": "transport_error",
                "message": redact_message(str(error)),
            }
        }
    latency = round((time.perf_counter() - started) * 1000)
    result = base_result(task, latency, status, data)
    result["model_returned"] = data.get("modelVersion")
    if status != 200:
        result["error_type"], result["error_message"] = extract_error(data)
        return result
    usage = data.get("usageMetadata") or {}
    result["input_tokens"] = usage.get("promptTokenCount")
    result["output_tokens"] = usage.get("candidatesTokenCount")
    result["reasoning_tokens"] = usage.get("thoughtsTokenCount")
    try:
        candidates = data.get("candidates") or []
        if len(candidates) != 1:
            raise ValueError("response candidate count is not one")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        result["raw_response_text"] = text
        result["response_sha256"] = sha256_text(text)
        result["parsed_forecasts"], result["probability_sum_max_error"] = (
            validate_forecast(json.loads(text), task["targets"])
        )
        result["success"] = True
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        result["error_type"] = "validation_error"
        result["error_message"] = redact_message(str(error))
    return result


def run_task(
    task: dict[str, Any],
    api_key: str,
    base_url: str,
    max_attempts: int = 3,
) -> dict[str, Any]:
    attempt_history = []
    for attempt in range(1, max_attempts + 1):
        task["attempt"] = attempt
        try:
            if task["provider"] == "openai":
                result = call_openai(task, api_key, base_url)
            elif task["provider"] == "deepseek":
                result = call_deepseek(task, api_key)
            elif task["provider"] == "mistral":
                result = call_mistral(task, api_key)
            elif task["provider"] == "qwen":
                result = call_qwen(task, api_key)
            else:
                result = call_gemini(task, api_key)
        except Exception as error:  # A paid response must still become an audit row.
            result = base_result(task, 0, 0, {})
            result["error_type"] = "worker_exception"
            result["error_message"] = redact_message(f"{type(error).__name__}: {error}")
        attempt_history.append(
            {
                "attempt": attempt,
                "status_code": result["status_code"],
                "success": result["success"],
                "latency_ms": result["latency_ms"],
                "response_sha256": result["response_sha256"],
                "error_type": result["error_type"],
                "error_message": result["error_message"],
            }
        )
        validation_retry = result.get("error_type") == "validation_error"
        if result["success"] or (
            result["status_code"] not in RETRIABLE_STATUS_CODES
            and not validation_retry
        ):
            result["attempt_history"] = attempt_history
            return result
        if attempt < max_attempts:
            retry_after = result.get("retry_after_seconds")
            delay = (
                min(float(retry_after), 60.0)
                if retry_after is not None
                else min(1.0 * 2 ** (attempt - 1), 60.0)
            )
            time.sleep(delay + random.random() * 0.25)
    result["attempt_history"] = attempt_history
    if result.get("error_type") == "validation_error":
        result["terminal_failure"] = False
    return result


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed = {}
    raw_lines = path.read_bytes().splitlines(keepends=True)
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        try:
            result = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            is_tail = line_number == len(raw_lines) and not raw_line.endswith(
                (b"\n", b"\r")
            )
            if is_tail:
                print(
                    f"warning: ignored truncated checkpoint tail in {path}",
                    file=sys.stderr,
                )
                break
            raise ValueError(f"Invalid checkpoint line {line_number}: {error}") from error
        if result.get("success"):
            completed[result["call_id"]] = result
    return completed


def build_tasks(
    profiles: list[dict[str, Any]],
    providers: list[str],
    models: dict[str, str],
    repeat_m0_profiles_per_country: int = 0,
    bundle: dict[str, Any] = PROMPT_BUNDLE,
    target_batch_size: int = 0,
    target_batch_limit: int = 0,
    cells: tuple[tuple[str, str], ...] | None = None,
    target_batch_ids: tuple[str, ...] | None = None,
    experiment_id: str = "core",
) -> list[dict[str, Any]]:
    tasks = []
    selected_cells = cells or CELLS
    if repeat_m0_profiles_per_country:
        first_by_country: dict[str, list[dict[str, Any]]] = {
            country: [] for country in COUNTRIES
        }
        for profile in profiles:
            bucket = first_by_country[profile["true_country_code"]]
            if len(bucket) < repeat_m0_profiles_per_country:
                bucket.append(profile)
        if any(
            len(first_by_country[country]) != repeat_m0_profiles_per_country
            for country in COUNTRIES
        ):
            raise ValueError("Not enough profiles for the requested repeat control")
        profiles = [
            profile
            for country in COUNTRIES
            for profile in first_by_country[country]
        ]
        selected_cells = (("M0_REPEAT", "correct"),)
    for provider in providers:
        for profile in profiles:
            profile_batches = [profile]
            if target_batch_size:
                profile_batches = []
                for start in range(0, len(profile["targets"]), target_batch_size):
                    batch_targets = profile["targets"][start : start + target_batch_size]
                    batch_profile = dict(profile)
                    batch_profile["targets"] = batch_targets
                    batch_profile["target_options"] = {
                        target["item_id"]: profile["target_options"][target["item_id"]]
                        for target in batch_targets
                    }
                    profile_batches.append(batch_profile)
                if target_batch_limit:
                    profile_batches = profile_batches[:target_batch_limit]
            for batch_index, batch_profile in enumerate(profile_batches, start=1):
                batch_id = f"B{batch_index:02d}" if target_batch_size else None
                if target_batch_ids and batch_id not in target_batch_ids:
                    continue
                for condition, evidence_variant in selected_cells:
                    prompt, targets = build_prompt(
                        batch_profile, condition, evidence_variant, bundle
                    )
                    tasks.append(
                        {
                            "provider": provider,
                            "model": models[provider],
                            "profile": batch_profile,
                            "condition": condition,
                            "evidence_variant": evidence_variant,
                            "batch_id": batch_id,
                            "prompt": prompt,
                            "targets": targets,
                            "prompt_version": bundle["version"],
                            "system_prompt": bundle["system_prompt"],
                            "experiment": (
                                "m0_repeat" if condition == "M0_REPEAT" else experiment_id
                            ),
                            "call_id": make_call_id(
                                provider,
                                models[provider],
                                profile["profile_id"],
                                condition,
                                evidence_variant,
                                bundle["version"],
                                batch_id,
                                sha256_text(bundle["system_prompt"] + "\n" + prompt),
                            ),
                        }
                    )
    random.Random(20260721).shuffle(tasks)
    return tasks


def load_task_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    legacy_required = {
        "version", "status", "experiment_id", "profile_ids", "cells",
        "target_batches", "profile_file_sha256", "prompt_bundle_sha256",
    }
    schema_version = manifest.get("schema_version")
    if schema_version == "survey-task-manifest-v2":
        required = {
            "schema_version", "version", "status", "suite_version",
            "experiment_id", "task_mode", "profile_ids", "cells",
            "target_batches", "task_units", "source", "prompt",
            "provider_models", "runner_configuration", "selection", "expected",
            "immutable_output_path", "immutable_checkpoint_path",
            "analysis_contract",
        }
        optional = {
            "item_to_core_batch", "single_item_batch_to_item",
            "logical_cells", "prospective_amendment", "roster_amendment",
            "profile_target_batches", "runtime_policy",
        }
        unexpected = set(manifest).difference(required | optional)
        missing = required.difference(manifest)
        if missing or unexpected:
            raise ValueError(
                "Version-2 task manifest fields are invalid; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        manifest["profile_file_sha256"] = manifest["source"]["profile_file_sha256"]
        manifest["prompt_bundle_sha256"] = manifest["prompt"]["sha256"]
    elif set(manifest) != legacy_required:
        raise ValueError(
            f"Task manifest fields must be {sorted(legacy_required)}, got {sorted(manifest)}"
        )
    if len(manifest["profile_ids"]) != len(set(manifest["profile_ids"])):
        raise ValueError("Task manifest contains duplicate profile IDs")
    if not manifest["profile_ids"]:
        raise ValueError("Task manifest must contain at least one profile ID")
    cells = []
    for value in manifest["cells"]:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("Each task-manifest cell must be [condition, evidence]")
        condition, evidence = value
        if condition not in {"M0", "MO", "MR", "MV"}:
            raise ValueError(f"Invalid task-manifest condition: {condition}")
        if evidence not in {"correct", "shuffled"}:
            raise ValueError(f"Invalid task-manifest evidence: {evidence}")
        cells.append((condition, evidence))
    if len(cells) != len(set(cells)):
        raise ValueError("Task manifest contains duplicate cells")
    if not cells:
        raise ValueError("Task manifest must contain at least one cell")
    target_batches = manifest["target_batches"]
    if (
        not isinstance(target_batches, list)
        or not target_batches
        or any(
            not isinstance(batch_id, str) or not batch_id
            for batch_id in target_batches
        )
    ):
        raise ValueError("Task manifest must contain non-empty target batch IDs")
    if len(target_batches) != len(set(target_batches)):
        raise ValueError("Task manifest contains duplicate target batches")
    if schema_version == "survey-task-manifest-v2":
        if manifest["task_mode"] not in {"standard", "repeat_identical"}:
            raise ValueError("Invalid version-2 task mode")
        unit_keys = []
        profile_ids = set(manifest["profile_ids"])
        for unit in manifest["task_units"]:
            if set(unit) != {
                "profile_id", "condition", "evidence_variant", "batch_id"
            }:
                raise ValueError("Invalid version-2 task unit fields")
            key = (
                unit["profile_id"], unit["condition"],
                unit["evidence_variant"], unit["batch_id"],
            )
            if unit["profile_id"] not in profile_ids:
                raise ValueError("Task unit references a profile outside profile_ids")
            if (unit["condition"], unit["evidence_variant"]) not in cells:
                raise ValueError("Task unit references a cell outside cells")
            if unit["batch_id"] not in target_batches:
                raise ValueError("Task unit references a batch outside target_batches")
            unit_keys.append(key)
        if not unit_keys or len(unit_keys) != len(set(unit_keys)):
            raise ValueError("Version-2 task units must be non-empty and unique")
        expected = manifest["expected"]
        if expected["calls_per_provider"] != len(unit_keys):
            raise ValueError("Manifest expected per-provider calls do not match task units")
        if expected["providers"] != len(manifest["provider_models"]):
            raise ValueError("Manifest expected provider count does not match model roster")
        if expected["total_calls"] != len(unit_keys) * expected["providers"]:
            raise ValueError("Manifest expected total calls are inconsistent")
        manifest["parsed_task_units"] = set(unit_keys)
        profile_target_batches = manifest.get("profile_target_batches")
        if profile_target_batches is not None:
            if set(profile_target_batches) != set(manifest["profile_ids"]):
                raise ValueError(
                    "Profile-specific target batches do not cover profile_ids"
                )
            for profile_id, batches in profile_target_batches.items():
                if not set(target_batches).issubset(batches):
                    raise ValueError(
                        f"Profile target batches omit selected target_batches: {profile_id}"
                    )
                flattened = [item for batch in batches.values() for item in batch]
                if len(flattened) != len(set(flattened)):
                    raise ValueError(
                        f"Profile target batches contain duplicate items: {profile_id}"
                    )
    manifest["parsed_cells"] = tuple(cells)
    return manifest


def apply_profile_target_batches(
    profiles: list[dict[str, Any]], manifest: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Order each profile's targets according to its frozen batch assignment."""
    if not manifest or not manifest.get("profile_target_batches"):
        return profiles
    assignments = manifest["profile_target_batches"]
    ordered: list[dict[str, Any]] = []
    for profile in profiles:
        by_id = {target["item_id"]: target for target in profile["targets"]}
        batch_map = assignments[profile["profile_id"]]
        target_ids = [
            item_id
            for batch_id in sorted(batch_map)
            for item_id in batch_map[batch_id]
        ]
        if set(target_ids) != set(by_id) or len(target_ids) != len(by_id):
            raise ValueError(
                f"Frozen batch assignment does not match profile targets: "
                f"{profile['profile_id']}"
            )
        value = dict(profile)
        value["targets"] = [by_id[item_id] for item_id in target_ids]
        ordered.append(value)
    return ordered


def apply_task_units(
    tasks: list[dict[str, Any]], manifest: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Apply an exact version-2 task selection and repeat identity contract."""
    if not manifest or "parsed_task_units" not in manifest:
        return tasks
    selected = []
    for task in tasks:
        key = (
            task["profile"]["profile_id"],
            task["condition"],
            task["evidence_variant"],
            task.get("batch_id"),
        )
        if key not in manifest["parsed_task_units"]:
            continue
        task = dict(task)
        if manifest["task_mode"] == "repeat_identical":
            task["source_call_id"] = task["call_id"]
            task["call_id"] += "/ROBUST-REPEAT-01"
        task["experiment"] = manifest["experiment_id"]
        selected.append(task)
    actual = {
        (
            task["profile"]["profile_id"], task["condition"],
            task["evidence_variant"], task.get("batch_id"),
        )
        for task in selected
    }
    missing = manifest["parsed_task_units"].difference(actual)
    if missing:
        raise ValueError(
            "Version-2 task units generated no calls: " + repr(sorted(missing)[:3])
        )
    return selected


def apply_task_manifest(
    profiles: list[dict[str, Any]], manifest: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if manifest is None:
        return profiles
    by_id = {profile["profile_id"]: profile for profile in profiles}
    missing = set(manifest["profile_ids"]).difference(by_id)
    if missing:
        raise ValueError(f"Task manifest references missing profiles: {sorted(missing)[:5]}")
    return [by_id[profile_id] for profile_id in manifest["profile_ids"]]


def execution_gate_error(manifest: dict[str, Any] | None) -> str | None:
    """Return a fail-closed execution-gate error for a frozen v2 task."""
    if not manifest or manifest.get("schema_version") != "survey-task-manifest-v2":
        return None
    contract = manifest.get("analysis_contract") or {}
    gate_file = contract.get("execution_gate_status_file")
    gate_path = contract.get("execution_gate_json_path")
    required = contract.get("execution_gate_required_value")
    if gate_file is None and gate_path is None and required is None:
        return None
    if not gate_file or not gate_path or required is None:
        return "frozen execution-gate contract is incomplete"
    status_path = ROOT / str(gate_file)
    if not status_path.exists():
        return "frozen execution-gate status file is missing"
    try:
        value: Any = json.loads(status_path.read_text(encoding="utf-8"))
        for component in str(gate_path).split("."):
            value = value[component]
    except (OSError, ValueError, KeyError, TypeError):
        return "frozen execution-gate JSON path is absent or invalid"
    if value != required:
        policy = (manifest.get("runtime_policy") or {}).get(
            "scientific_gate_action"
        )
        if policy == "report_only":
            return None
        return "frozen aggregate Gate C status is not open"
    return None


def validate_v2_source_bindings(
    manifest: dict[str, Any], analysis_manifest: Path | None
) -> None:
    """Verify every frozen local source and the explicit analysis-manifest input."""
    if manifest.get("schema_version") != "survey-task-manifest-v2":
        return
    source = manifest["source"]
    hash_fields = {
        "profile_file": "profile_file_sha256",
        "analysis_manifest_file": "analysis_manifest_sha256",
        "label_assignment_file": "label_assignment_sha256",
        "donor_assignment_file": "donor_assignment_sha256",
        "core_manifest_file": "core_manifest_sha256",
        "core_output_file": "core_output_sha256",
    }
    for file_field, hash_field in hash_fields.items():
        if file_field not in source and hash_field not in source:
            continue
        if not source.get(file_field) or not source.get(hash_field):
            raise ValueError(f"Frozen source binding is incomplete: {file_field}")
        source_path = ROOT / source[file_field]
        if not source_path.is_file():
            raise ValueError(f"Frozen source file is missing: {source[file_field]}")
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_hash != source[hash_field]:
            raise ValueError(f"Frozen source hash mismatch: {source[file_field]}")
    frozen_analysis = source.get("analysis_manifest_file")
    if frozen_analysis:
        if analysis_manifest is None:
            raise ValueError("Frozen task manifest requires --analysis-manifest")
        if analysis_manifest.resolve() != (ROOT / frozen_analysis).resolve():
            raise ValueError("Analysis manifest path does not match the frozen task manifest")


def validate_v2_immutable_paths(
    manifest: dict[str, Any], output: Path, checkpoint: Path | None
) -> None:
    """Reject output or checkpoint drift from a version-2 task definition."""
    if manifest.get("schema_version") != "survey-task-manifest-v2":
        return
    expected_output = ROOT / manifest["immutable_output_path"]
    if output.resolve() != expected_output.resolve():
        raise ValueError("Output path does not match the frozen task manifest")
    expected_checkpoint = ROOT / manifest["immutable_checkpoint_path"]
    actual_checkpoint = checkpoint or output.with_suffix(".checkpoint.sqlite3")
    if actual_checkpoint.resolve() != expected_checkpoint.resolve():
        raise ValueError("Checkpoint path does not match the frozen task manifest")


def add_balanced_core_repeats(
    tasks: list[dict[str, Any]], providers: list[str], calls_per_cell: int
) -> list[dict[str, Any]]:
    """Add prompt-identical repeats, matched across models and balanced by cell."""
    if calls_per_cell <= 0:
        return tasks
    repeats = []
    for provider in providers:
        provider_tasks = [
            task for task in tasks
            if task["provider"] == provider and task.get("experiment") == "core"
        ]
        for condition, evidence_variant in CELLS:
            bucket = [
                task for task in provider_tasks
                if task["condition"] == condition
                and task["evidence_variant"] == evidence_variant
            ]
            bucket.sort(
                key=lambda task: sha256_text(
                    "balanced-core-repeat\x1f"
                    + "\x1f".join(
                        (
                            task["profile"]["profile_id"],
                            task["condition"],
                            task["evidence_variant"],
                            task.get("batch_id") or "all-targets",
                            task["prompt_version"],
                        )
                    )
                )
            )
            if len(bucket) < calls_per_cell:
                raise ValueError(
                    f"Not enough {provider} {condition}/{evidence_variant} calls "
                    f"for {calls_per_cell} repeats"
                )
            for original in bucket[:calls_per_cell]:
                repeated = dict(original)
                repeated["experiment"] = "repeat"
                repeated["source_call_id"] = original["call_id"]
                repeated["call_id"] = original["call_id"] + "/REPEAT-01"
                repeats.append(repeated)
    combined = [*tasks, *repeats]
    random.Random(20260723).shuffle(combined)
    return combined


def limit_tasks_balanced(
    tasks: list[dict[str, Any]], providers: list[str], calls_per_provider: int
) -> list[dict[str, Any]]:
    """Select the same provider-neutral, cell-balanced work for every model."""
    if calls_per_provider <= 0:
        return tasks
    limited = []
    for provider in providers:
        provider_tasks = [task for task in tasks if task["provider"] == provider]
        buckets = []
        actual_cells = sorted(
            {
                (task["condition"], task["evidence_variant"])
                for task in provider_tasks
            }
        )
        for condition, evidence_variant in actual_cells:
            bucket = [
                task
                for task in provider_tasks
                if task["condition"] == condition
                and task["evidence_variant"] == evidence_variant
            ]
            bucket.sort(
                key=lambda task: sha256_text(
                    "small-pilot-selection\x1f"
                    + "\x1f".join(
                        (
                            task["profile"]["profile_id"],
                            task["condition"],
                            task["evidence_variant"],
                            task.get("batch_id") or "all-targets",
                            task["prompt_version"],
                        )
                    )
                )
            )
            buckets.append(bucket)
        for index in range(calls_per_provider):
            bucket = buckets[index % len(buckets)]
            offset = index // len(buckets)
            if offset < len(bucket):
                limited.append(bucket[offset])
    return limited


def build_run_metadata(
    args: argparse.Namespace,
    profiles: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    base_url: str,
    bundle: dict[str, Any],
    models: dict[str, str],
) -> dict[str, Any]:
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    actual_cells = sorted(
        {(task["condition"], task["evidence_variant"]) for task in tasks}
    )
    frozen = {
        "restricted_respondent_derived_data": True,
        "source_identifiers_in_prompts": False,
        "profile_file_sha256": hashlib.sha256(args.profiles.read_bytes()).hexdigest(),
        "profile_count": len(profiles),
        "profiles_per_country": dict(
            Counter(profile["true_country_code"] for profile in profiles)
        ),
        "target_batch_size": args.target_batch_size,
        "target_batch_limit": args.target_batch_limit,
        "reverse_target_options": bool(args.reverse_target_options),
        "task_manifest": str(args.task_manifest) if args.task_manifest else None,
        "task_manifest_sha256": (
            hashlib.sha256(args.task_manifest.read_bytes()).hexdigest()
            if args.task_manifest else None
        ),
        "analysis_manifest": str(args.analysis_manifest) if args.analysis_manifest else None,
        "analysis_manifest_sha256": (
            hashlib.sha256(args.analysis_manifest.read_bytes()).hexdigest()
            if args.analysis_manifest else None
        ),
        "cells": actual_cells,
        "models": {provider: models[provider] for provider in args.providers},
        "openai_endpoint": {"base_url": base_url, "official": base_url == OFFICIAL_OPENAI_BASE_URL},
        "provider_endpoints": {
            "openai": base_url,
            "deepseek": DEEPSEEK_BASE_URL,
            "mistral": MISTRAL_BASE_URL,
            "gemini": GEMINI_BASE_URL,
            "qwen": QWEN_BASE_URL,
        },
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "request_protocol_version": REQUEST_PROTOCOL_VERSION,
        "runner_sha256": runner_sha256,
        "max_attempts": args.max_attempts,
        "validation_retry_policy": (
            "schema-constrained retries through max_attempts; invalid latest "
            "responses remain pending across automation cycles"
        ),
        "workers_per_key": args.workers_per_key,
        "workers_per_provider": args.workers_per_provider,
        "provider_worker_overrides": provider_worker_overrides(args.provider_workers),
        "planned_calls": len(tasks),
        "max_calls_per_provider": args.max_calls_per_provider,
        "repeat_m0_profiles_per_country": args.repeat_m0_profiles_per_country,
        "repeat_core_calls_per_cell": args.repeat_core_calls_per_cell,
        "prompt_bundle": str(args.prompt_bundle),
        "prompt_version": bundle["version"],
        "prompt_bundle_sha256": hashlib.sha256(
            args.prompt_bundle.read_bytes()
        ).hexdigest(),
        "system_prompt_sha256": sha256_text(bundle["system_prompt"]),
        "credentials_persisted": False,
    }
    signature_payload = {
        "frozen": frozen,
        "call_ids": sorted(task["call_id"] for task in tasks),
    }
    frozen["run_signature"] = sha256_text(
        json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)
    )
    return {"created_at": utc_now(), **frozen}


def checkpoint_is_empty(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        connection = sqlite3.connect(path)
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        result_count = (
            connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            if "results" in tables else 0
        )
        history_count = (
            connection.execute("SELECT COUNT(*) FROM result_history").fetchone()[0]
            if "result_history" in tables else 0
        )
        connection.close()
    except (OSError, sqlite3.Error):
        return False
    return result_count == 0 and history_count == 0


OPERATIONAL_MIGRATION_FIELDS = {
    "created_at",
    "run_signature",
    "runner_sha256",
    "validation_retry_policy",
    "operational_migrations",
    "workers_per_key",
    "workers_per_provider",
    "provider_worker_overrides",
}


def operationally_compatible_run(
    previous: dict[str, Any], current: dict[str, Any]
) -> bool:
    """Allow a checkpoint migration only when scientific inputs are unchanged."""
    previous_frozen = {
        key: value for key, value in previous.items()
        if key not in OPERATIONAL_MIGRATION_FIELDS
    }
    current_frozen = {
        key: value for key, value in current.items()
        if key not in OPERATIONAL_MIGRATION_FIELDS
    }
    return json.dumps(
        previous_frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) == json.dumps(
        current_frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def migrate_checkpoint_signature(
    checkpoint_path: Path,
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Record an auditable operational-only runner migration."""
    if not operationally_compatible_run(previous, current):
        raise ValueError(
            "Run metadata differs in scientific fields; use a new output path"
        )
    migration = {
        "migrated_at": utc_now(),
        "previous_run_signature": previous.get("run_signature"),
        "previous_runner_sha256": previous.get("runner_sha256"),
        "new_runner_sha256": current.get("runner_sha256"),
        "reason": "operational retry and unattended-supervisor resilience only",
    }
    migrated = dict(current)
    migrated["operational_migrations"] = [
        *(previous.get("operational_migrations") or []), migration,
    ]
    with sqlite3.connect(checkpoint_path) as connection:
        stored = connection.execute(
            "SELECT value FROM run_meta WHERE key='run_metadata'"
        ).fetchone()
        if stored is None:
            raise ValueError("Checkpoint has no frozen run metadata")
        stored_metadata = json.loads(stored[0])
        if not operationally_compatible_run(stored_metadata, current):
            raise ValueError("Checkpoint scientific metadata does not match")
        connection.execute(
            "UPDATE run_meta SET value=? WHERE key='run_signature'",
            (current["run_signature"],),
        )
        connection.execute(
            "UPDATE run_meta SET value=? WHERE key='run_metadata'",
            (json.dumps(migrated, ensure_ascii=False, sort_keys=True),),
        )
        connection.commit()
    return migrated


def write_run_metadata(
    path: Path, metadata: dict[str, Any], checkpoint_path: Path
) -> None:
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("run_signature") != metadata["run_signature"]:
            if not checkpoint_is_empty(checkpoint_path):
                metadata = migrate_checkpoint_signature(
                    checkpoint_path, previous, metadata
                )
            atomic_write_text(
                path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
            )
        return
    atomic_write_text(path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def duration_text(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class ProgressTracker:
    def __init__(
        self,
        tasks: list[dict[str, Any]],
        results: dict[str, dict[str, Any]],
        models: dict[str, str],
        key_counts: dict[str, int],
        worker_counts: dict[str, int],
        run_label: str = "inference",
        checkpoint_path: Path | None = None,
        output_path: Path | None = None,
        resume_command: str | None = None,
    ):
        self.tasks = tasks
        self.models = models
        self.key_counts = key_counts
        self.worker_counts = worker_counts
        self.run_label = run_label
        self.checkpoint_path = checkpoint_path
        self.output_path = output_path
        self.resume_command = resume_command
        self.results = dict(results)
        self.planned_provider = Counter(task["provider"] for task in tasks)
        self.planned_cell = Counter(
            (task["condition"], task["evidence_variant"]) for task in tasks
        )
        self.planned_provider_cell = Counter(
            (task["provider"], task["condition"], task["evidence_variant"])
            for task in tasks
        )
        self.planned_experiment = Counter(
            task.get("experiment", "core") for task in tasks
        )
        self.cells = sorted(self.planned_cell)
        self.success_provider: Counter[str] = Counter()
        self.terminal_provider: Counter[str] = Counter()
        self.retryable_failure_provider: Counter[str] = Counter()
        self.success_cell: Counter[tuple[str, str]] = Counter()
        self.terminal_cell: Counter[tuple[str, str]] = Counter()
        self.success_provider_cell: Counter[tuple[str, str, str]] = Counter()
        self.terminal_provider_cell: Counter[tuple[str, str, str]] = Counter()
        self.success_experiment: Counter[str] = Counter()
        self.terminal_experiment: Counter[str] = Counter()
        for result in self.results.values():
            self._add_result_count(result)
        self.started = time.monotonic()
        self.started_at = utc_now()
        self.new_results: Counter[str] = Counter()
        self.new_successes: Counter[str] = Counter()
        self.blocked: dict[str, str] = {}

    def _add_result_count(self, result: dict[str, Any]) -> None:
        provider = result["provider"]
        if result.get("success"):
            self.success_provider[provider] += 1
            self.success_cell[(result["condition"], result["evidence_variant"])] += 1
            self.success_provider_cell[
                (provider, result["condition"], result["evidence_variant"])
            ] += 1
            self.success_experiment[result.get("experiment", "core")] += 1
        elif result.get("terminal_failure"):
            self.terminal_provider[provider] += 1
            self.terminal_cell[(result["condition"], result["evidence_variant"])] += 1
            self.terminal_provider_cell[
                (provider, result["condition"], result["evidence_variant"])
            ] += 1
            self.terminal_experiment[result.get("experiment", "core")] += 1
        else:
            self.retryable_failure_provider[provider] += 1

    def _remove_result_count(self, result: dict[str, Any]) -> None:
        provider = result["provider"]
        if result.get("success"):
            self.success_provider[provider] -= 1
            self.success_cell[(result["condition"], result["evidence_variant"])] -= 1
            self.success_provider_cell[
                (provider, result["condition"], result["evidence_variant"])
            ] -= 1
            self.success_experiment[result.get("experiment", "core")] -= 1
        elif result.get("terminal_failure"):
            self.terminal_provider[provider] -= 1
            self.terminal_cell[(result["condition"], result["evidence_variant"])] -= 1
            self.terminal_provider_cell[
                (provider, result["condition"], result["evidence_variant"])
            ] -= 1
            self.terminal_experiment[result.get("experiment", "core")] -= 1
        else:
            self.retryable_failure_provider[provider] -= 1

    def update(self, result: dict[str, Any]) -> None:
        previous = self.results.get(result["call_id"])
        if previous is not None:
            self._remove_result_count(previous)
        self.results[result["call_id"]] = result
        self._add_result_count(result)
        self.new_results[result["provider"]] += 1
        if result.get("success"):
            self.new_successes[result["provider"]] += 1

    def render(
        self,
        active: Counter[str] | None = None,
        stopped: bool = False,
        planned: bool = False,
    ) -> str:
        active = active or Counter()
        elapsed = max(time.monotonic() - self.started, 0.001)
        total_success = sum(self.success_provider.values())
        total_terminal = sum(self.terminal_provider.values())
        total_completed = total_success + total_terminal
        total_pending = max(len(self.tasks) - total_completed, 0)
        run_status = (
            "PLANNED"
            if planned
            else "RUNNING"
            if not stopped
            else "COMPLETE"
            if total_pending == 0
            and total_terminal == 0
            else "COMPLETE_WITH_INVALID_OUTPUTS"
            if total_pending == 0
            else "BLOCKED"
            if self.blocked
            else "PAUSED"
        )
        lines = [
            "# Experiment Inference Progress",
            "",
            "> This file is generated atomically by `survey_pilot_runner.py`. Do not edit while a run is active.",
            "",
            f"- Last update (UTC): `{utc_now()}`",
            f"- Run started (UTC): `{self.started_at}`",
            f"- Elapsed: `{duration_text(elapsed)}`",
            f"- Planned calls: **{len(self.tasks):,}**",
            f"- Successful checkpoints: **{total_success:,}**",
            f"- Terminal invalid outputs: **{total_terminal:,}**",
            f"- Run status: **{run_status}**",
            f"- Stage: **{self.run_label}**",
            f"- Process ID: **{os.getpid()}**",
            f"- SQLite checkpoint: `{self.checkpoint_path}`",
            f"- JSONL audit export: `{self.output_path}`",
            "",
            "## Model Progress",
            "",
            "| Provider | Model | Keys | Workers | Planned | Valid | Terminal invalid | Retryable latest | Pending | Complete | Success/min | ETA | Status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        provider_etas = []
        for provider in self.models:
            provider_planned = self.planned_provider[provider]
            success = self.success_provider[provider]
            terminal = self.terminal_provider[provider]
            retryable = self.retryable_failure_provider[provider]
            completed = success + terminal
            pending = max(provider_planned - completed, 0)
            rate = self.new_successes[provider] / elapsed * 60
            eta = pending / rate * 60 if rate > 0 and provider not in self.blocked else None
            if eta is not None:
                provider_etas.append(eta)
            status = (
                "DONE"
                if pending == 0 and terminal == 0
                else "DONE_INVALID"
                if pending == 0
                else f"BLOCKED: {self.blocked[provider]}"
                if provider in self.blocked
                else "PLANNED"
                if planned
                else "PAUSED"
                if stopped
                else "RUNNING"
                if active[provider]
                else "PENDING"
            )
            progress = 100 * completed / provider_planned if provider_planned else 100.0
            lines.append(
                f"| {provider} | `{self.models[provider]}` | {self.key_counts.get(provider, 0)} "
                f"| {self.worker_counts.get(provider, 0)} | {provider_planned:,} | {success:,} "
                f"| {terminal:,} | {retryable:,} "
                f"| {pending:,} | {progress:.1f}% | {rate:.1f} | {duration_text(eta)} | {status} |"
            )
        pending_providers = [
            provider
            for provider in self.models
            if self.planned_provider[provider]
            > self.success_provider[provider] + self.terminal_provider[provider]
        ]
        if any(provider in self.blocked for provider in pending_providers):
            total_eta = "BLOCKED"
        elif any(self.new_successes[provider] == 0 for provider in pending_providers):
            total_eta = "-- (warming up)"
        elif provider_etas:
            total_eta = duration_text(max(provider_etas))
        else:
            total_eta = "0s" if total_pending == 0 else "--"
        lines.extend(
            [
                "",
                f"Estimated total remaining time: **{total_eta}**",
                "",
        "## Experimental Cell Progress",
                "",
                "| Condition | Evidence | Planned | Valid | Terminal invalid | Pending |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for cell in self.cells:
            cell_planned = self.planned_cell[cell]
            success = self.success_cell[cell]
            terminal = self.terminal_cell[cell]
            lines.append(
                f"| {cell[0]} | {cell[1]} | {cell_planned:,} | {success:,} "
                f"| {terminal:,} | {max(cell_planned-success-terminal, 0):,} |"
        )
        lines.extend(
            [
                "",
                "## Experiment Workstream Progress",
                "",
                "| Workstream | Planned | Valid | Terminal invalid | Pending |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for experiment in sorted(self.planned_experiment):
            stream_planned = self.planned_experiment[experiment]
            valid = self.success_experiment[experiment]
            terminal = self.terminal_experiment[experiment]
            lines.append(
                f"| {experiment} | {stream_planned:,} | {valid:,} | {terminal:,} "
                f"| {max(stream_planned-valid-terminal, 0):,} |"
            )
        lines.extend(
            [
                "",
                "## Model by Cell Progress",
                "",
                "| Provider | Condition | Evidence | Planned | Valid | Terminal invalid | Pending |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for provider in self.models:
            for condition, evidence in self.cells:
                key = (provider, condition, evidence)
                cell_planned = self.planned_provider_cell[key]
                success = self.success_provider_cell[key]
                terminal = self.terminal_provider_cell[key]
                if cell_planned:
                    lines.append(
                        f"| {provider} | {condition} | {evidence} | {cell_planned:,} "
                        f"| {success:,} | {terminal:,} "
                        f"| {max(cell_planned-success-terminal, 0):,} |"
                    )
        lines.extend(
            [
                "",
                "## Recovery Semantics",
                "",
                "- A call is measured only after a valid response or terminal invalid outcome is committed to the SQLite WAL checkpoint.",
                "- Operational failures remain pending for resume; validation failures are terminal after the frozen retry policy.",
                "- Provider circuit breakers stop new work for an exhausted credential while other providers continue.",
                "- Total ETA is the maximum provider ETA because model queues run in parallel.",
                "- Resume by running the exact frozen command again; successful call IDs are skipped.",
                "",
            ]
        )
        if self.resume_command:
            lines.extend(
                [
                    "## Resume Command",
                    "",
                    "```powershell",
                    self.resume_command,
                    "```",
                    "",
                ]
            )
        return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=Path("dataset/derived/pilot_profiles_6x12.jsonl"))
    parser.add_argument(
        "--prompt-bundle",
        type=Path,
        default=DEFAULT_PROMPT_BUNDLE_PATH,
    )
    parser.add_argument("--max-profiles-per-country", type=int, choices=range(1, 121), default=2)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=["openai", "deepseek"],
    )
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--mistral-model", default=DEFAULT_MISTRAL_MODEL)
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--workers-per-key", type=int, choices=range(1, 33), default=4)
    parser.add_argument(
        "--provider-workers",
        action="append",
        default=[],
        metavar="PROVIDER=COUNT",
        help="Operational per-provider worker override, for example gemini=1.",
    )
    parser.add_argument(
        "--workers-per-provider",
        type=int,
        choices=range(0, 65),
        default=0,
        help="Compatibility override; 0 uses workers-per-key times key count.",
    )
    parser.add_argument("--max-attempts", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--quota-stop-threshold", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--target-batch-size", type=int, choices=range(0, 21), default=5)
    parser.add_argument("--target-batch-limit", type=int, choices=range(0, 21), default=0)
    parser.add_argument(
        "--task-manifest",
        type=Path,
        help="Frozen profile/cell/batch selection for an auxiliary experiment.",
    )
    parser.add_argument(
        "--analysis-manifest",
        type=Path,
        help="Frozen item-stratum manifest included in the run signature.",
    )
    parser.add_argument("--reverse-target-options", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("api_pilot/restricted_outputs/survey_pilot.jsonl"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--progress-doc", type=Path, default=Path("api_pilot/EXPERIMENT_PROGRESS.md"))
    parser.add_argument("--run-label", default="inference")
    parser.add_argument(
        "--resume-command",
        help="Sanitized command shown in the progress document; never include credentials.",
    )
    parser.add_argument("--progress-every", type=int, choices=range(1, 1001), default=12)
    parser.add_argument("--key-file", type=Path, default=Path("api_pilot/apikey.txt"))
    parser.add_argument(
        "--allow-custom-openai-base-url",
        action="store_true",
        help="Acknowledge an explicitly configured OpenAI-compatible base URL.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-calls-per-provider",
        type=int,
        choices=range(0, 100001),
        default=0,
        help="Limit each provider to a deterministic, condition-balanced task subset.",
    )
    parser.add_argument(
        "--stop-after-successes-per-provider",
        type=int,
        choices=range(0, 100001),
        default=0,
        help=(
            "Operational phase boundary. Stop submitting for each provider after "
            "this many frozen calls have succeeded; excluded from the run signature."
        ),
    )
    parser.add_argument(
        "--repeat-m0-profiles-per-country",
        type=int,
        choices=range(0, 13),
        default=0,
    )
    parser.add_argument(
        "--repeat-core-calls-per-cell",
        type=int,
        choices=range(0, 1001),
        default=0,
        help="Add this many prompt-identical repeats per provider and core cell.",
    )
    parser.add_argument("--acknowledge-respondent-data-processing", action="store_true")
    return parser.parse_args()


def provider_worker_overrides(values: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for value in values:
        provider, separator, count_text = value.partition("=")
        provider = provider.strip().lower()
        if separator != "=" or provider not in PROVIDERS:
            raise ValueError(f"Invalid --provider-workers value: {value}")
        try:
            count = int(count_text)
        except ValueError as error:
            raise ValueError(f"Invalid --provider-workers count: {value}") from error
        if not 1 <= count <= 64:
            raise ValueError(f"Provider worker count must be 1..64: {value}")
        overrides[provider] = count
    return overrides


def main() -> int:
    args = parse_args()
    worker_overrides = provider_worker_overrides(args.provider_workers)
    providers = list(dict.fromkeys(args.providers))
    models = {
        "openai": args.openai_model,
        "deepseek": args.deepseek_model,
        "mistral": args.mistral_model,
        "gemini": args.gemini_model,
        "qwen": args.qwen_model,
    }
    base_url = normalized_base_url(os.getenv("OPENAI_BASE_URL", OFFICIAL_OPENAI_BASE_URL))
    if (
        "openai" in providers
        and base_url != OFFICIAL_OPENAI_BASE_URL
        and not args.allow_custom_openai_base_url
    ):
        print(
            "Generation blocked: OPENAI_BASE_URL differs from the default base URL. "
            "Unset it or add --allow-custom-openai-base-url to acknowledge the configured route.",
            file=sys.stderr,
        )
        return 2
    bundle = load_prompt_bundle(args.prompt_bundle)
    profiles = load_profiles(args.profiles, args.max_profiles_per_country)
    task_manifest = load_task_manifest(args.task_manifest)
    if task_manifest:
        profile_hash = hashlib.sha256(args.profiles.read_bytes()).hexdigest()
        prompt_hash = hashlib.sha256(args.prompt_bundle.read_bytes()).hexdigest()
        if task_manifest["profile_file_sha256"] != profile_hash:
            raise ValueError("Task manifest profile hash does not match --profiles")
        if task_manifest["prompt_bundle_sha256"] != prompt_hash:
            raise ValueError("Task manifest prompt hash does not match --prompt-bundle")
        if task_manifest.get("schema_version") == "survey-task-manifest-v2":
            validate_v2_source_bindings(task_manifest, args.analysis_manifest)
            manifest_models = task_manifest["provider_models"]
            if set(providers) != set(manifest_models):
                raise ValueError("Providers do not match the frozen task-manifest roster")
            if any(models[provider] != model for provider, model in manifest_models.items()):
                raise ValueError("Model IDs do not match the frozen task-manifest roster")
            runner_config = task_manifest["runner_configuration"]
            if args.target_batch_size != runner_config["target_batch_size"]:
                raise ValueError("Target batch size does not match the frozen task manifest")
            if bool(args.reverse_target_options) != bool(
                runner_config["reverse_target_options"]
            ):
                raise ValueError("Option-order setting does not match the frozen task manifest")
            validate_v2_immutable_paths(
                task_manifest, args.output, args.checkpoint
            )
            if any((
                args.target_batch_limit,
                args.max_calls_per_provider,
                args.repeat_m0_profiles_per_country,
                args.repeat_core_calls_per_cell,
            )):
                raise ValueError(
                    "Version-2 task manifests forbid post-manifest task limiting or repeats"
                )
        profiles = apply_task_manifest(profiles, task_manifest)
        profiles = apply_profile_target_batches(profiles, task_manifest)
    if args.reverse_target_options:
        profiles = [dict(profile) for profile in profiles]
        for profile in profiles:
            profile["target_options"] = {
                item_id: list(reversed(options))
                for item_id, options in profile["target_options"].items()
            }
    tasks = build_tasks(
        profiles,
        providers,
        models,
        args.repeat_m0_profiles_per_country,
        bundle,
        args.target_batch_size,
        args.target_batch_limit,
        task_manifest["parsed_cells"] if task_manifest else None,
        tuple(task_manifest["target_batches"]) if task_manifest else None,
        task_manifest["experiment_id"] if task_manifest else "core",
    )
    if task_manifest:
        generated_batches = {task.get("batch_id") for task in tasks}
        missing_batches = set(task_manifest["target_batches"]).difference(
            generated_batches
        )
        if missing_batches:
            raise ValueError(
                "Task manifest target batches generated no tasks: "
                + ", ".join(sorted(missing_batches))
            )
        tasks = apply_task_units(tasks, task_manifest)
        expected_per_provider = task_manifest.get("expected", {}).get(
            "calls_per_provider"
        )
        if expected_per_provider is not None:
            actual_counts = Counter(task["provider"] for task in tasks)
            if any(
                actual_counts[provider] != expected_per_provider
                for provider in providers
            ):
                raise ValueError(
                    "Generated calls per provider do not match the frozen task manifest"
                )
    if not tasks:
        raise ValueError("Frozen task selection generated zero tasks")
    tasks = add_balanced_core_repeats(
        tasks, providers, args.repeat_core_calls_per_cell
    )
    tasks = limit_tasks_balanced(tasks, providers, args.max_calls_per_provider)
    metadata_path = args.output.with_suffix(".run.json")
    run_metadata = build_run_metadata(args, profiles, tasks, base_url, bundle, models)
    checkpoint_path = args.checkpoint or args.output.with_suffix(".checkpoint.sqlite3")
    write_run_metadata(metadata_path, run_metadata, checkpoint_path)
    store = CheckpointStore(checkpoint_path, run_metadata["run_signature"], run_metadata)
    planned_ids = {task["call_id"] for task in tasks}
    store.import_jsonl(args.output, planned_ids)
    successful = store.successful_ids()
    completed = store.completed_ids()
    completed_planned = planned_ids.intersection(completed)
    pending = [task for task in tasks if task["call_id"] not in completed]
    print(
        f"profiles={len(profiles)} planned={len(tasks)} completed={len(completed_planned)} "
        f"successful={len(planned_ids.intersection(successful))} "
        f"pending={len(pending)} endpoint_official={base_url == OFFICIAL_OPENAI_BASE_URL}"
    )
    if args.dry_run:
        key_pool = configured_keys(args.key_file)
        worker_counts = {
            provider: worker_overrides.get(provider)
            or args.workers_per_provider
            or args.workers_per_key * max(len(key_pool.get(provider, [])), 1)
            for provider in providers
        }
        tracker = ProgressTracker(
            tasks,
            store.latest_results(),
            {provider: models[provider] for provider in providers},
            {provider: len(key_pool.get(provider, [])) for provider in providers},
            worker_counts,
            args.run_label,
            checkpoint_path,
            args.output,
            args.resume_command,
        )
        atomic_write_text(args.progress_doc, tracker.render(planned=True))
        store.close()
        return 0
    gate_error = execution_gate_error(task_manifest)
    if gate_error:
        print(f"Generation blocked: {gate_error}.", file=sys.stderr)
        store.close()
        return 2
    if not args.acknowledge_respondent_data_processing:
        print("Generation blocked: add --acknowledge-respondent-data-processing after reviewing data and provider terms.", file=sys.stderr)
        store.close()
        return 2
    keys = configured_keys(args.key_file)
    missing = [provider for provider in providers if not keys.get(provider)]
    if missing:
        print("Missing environment keys for: " + ", ".join(missing), file=sys.stderr)
        store.close()
        return 2
    worker_counts = {
        provider: worker_overrides.get(provider)
        or args.workers_per_provider
        or args.workers_per_key * len(keys[provider])
        for provider in providers
    }
    key_worker_limits = {
        provider: (
            max(1, math.ceil(worker_counts[provider] / len(keys[provider])))
            if provider in worker_overrides or args.workers_per_provider
            else args.workers_per_key
        )
        for provider in providers
    }
    prior_results = store.latest_results()
    successful_by_provider = Counter(
        row["provider"]
        for call_id, row in prior_results.items()
        if call_id in planned_ids and row.get("success")
    )
    phase_cap = args.stop_after_successes_per_provider
    if phase_cap and any(
        successful_by_provider[provider] > phase_cap for provider in providers
    ):
        store.close()
        raise ValueError(
            "A provider checkpoint is already beyond the requested phase boundary"
        )
    tracker = ProgressTracker(
        tasks,
        prior_results,
        {provider: models[provider] for provider in providers},
        {provider: len(keys[provider]) for provider in providers},
        worker_counts,
        args.run_label,
        checkpoint_path,
        args.output,
        args.resume_command,
    )
    atomic_write_text(args.progress_doc, tracker.render())
    succeeded = 0
    failed = 0
    queues = {
        provider: deque(task for task in pending if task["provider"] == provider)
        for provider in providers
    }
    executors = {
        provider: concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_counts[provider],
            thread_name_prefix=f"survey-{provider}",
        )
        for provider in providers
    }
    futures: dict[concurrent.futures.Future, tuple[dict[str, Any], str, int]] = {}
    key_cursor = Counter()
    key_429_streak: Counter[tuple[str, int]] = Counter()
    blocked_keys: dict[tuple[str, int], str] = {}
    active = Counter()
    active_key = Counter()
    stopping = False

    def unblocked_key_indexes(provider: str) -> list[int]:
        return [
            index
            for index in range(len(keys[provider]))
            if (provider, index) not in blocked_keys
        ]

    def available_key_indexes(provider: str) -> list[int]:
        return [
            index
            for index in unblocked_key_indexes(provider)
            if active_key[(provider, index)] < key_worker_limits[provider]
        ]

    def submit_next(provider: str) -> bool:
        available = available_key_indexes(provider)
        if (
            not queues[provider]
            or not available
            or stopping
            or (
                phase_cap > 0
                and successful_by_provider[provider] + active[provider]
                >= phase_cap
            )
        ):
            return False
        task = queues[provider][0]
        previous = prior_results.get(task["call_id"])
        key_index = choose_key_index(available, key_cursor[provider], previous)
        key_cursor[provider] += 1
        task = queues[provider].popleft()
        future = executors[provider].submit(
            run_task,
            task,
            keys[provider][key_index],
            base_url,
            args.max_attempts,
        )
        futures[future] = (task, provider, key_index)
        active[provider] += 1
        active_key[(provider, key_index)] += 1
        return True

    for provider in providers:
        for _ in range(worker_counts[provider]):
            if not submit_next(provider):
                break

    processed = 0
    try:
        while futures:
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                task, provider, key_index = futures.pop(future)
                active[provider] -= 1
                active_key[(provider, key_index)] -= 1
                try:
                    result = future.result()
                except Exception as error:
                    task["attempt"] = 0
                    result = base_result(task, 0, 0, {})
                    result["error_type"] = "future_exception"
                    result["error_message"] = redact_message(
                        f"{type(error).__name__}: {error}"
                    )
                result["credential_slot"] = key_index + 1
                store.record(result)
                tracker.update(result)
                processed += 1
                if result["success"]:
                    succeeded += 1
                    successful_by_provider[provider] += 1
                    key_429_streak[(provider, key_index)] = 0
                else:
                    failed += 1
                    status = result["status_code"]
                    key_was_blocked = False
                    if is_credential_or_quota_error(result):
                        blocked_keys[(provider, key_index)] = (
                            f"credential/quota: {result.get('error_type') or status}"
                        )
                        key_was_blocked = True
                    elif status == 429:
                        attempts_429 = sum(
                            attempt.get("status_code") == 429
                            for attempt in result.get("attempt_history", [])
                        )
                        key_429_streak[(provider, key_index)] += max(attempts_429, 1)
                        if key_429_streak[(provider, key_index)] >= args.quota_stop_threshold:
                            blocked_keys[(provider, key_index)] = "repeated HTTP 429 quota/rate limit"
                            key_was_blocked = True
                    else:
                        key_429_streak[(provider, key_index)] = 0
                    if key_was_blocked and unblocked_key_indexes(provider):
                        queues[provider].appendleft(task)
                unblocked = unblocked_key_indexes(provider)
                if not unblocked:
                    tracker.blocked[provider] = "; ".join(
                        sorted(
                            reason
                            for (blocked_provider, _), reason in blocked_keys.items()
                            if blocked_provider == provider
                        )
                    )
                while active[provider] < worker_counts[provider] and submit_next(provider):
                    pass
                if (
                    processed % args.progress_every == 0
                    or not result["success"]
                    or provider in tracker.blocked
                ):
                    atomic_write_text(args.progress_doc, tracker.render(active, stopping))
                    print(
                        f"progress={processed}/{len(pending)} new_success={succeeded} "
                        f"new_failed={failed} provider={provider}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        stopping = True
        print("Interrupt received; stopping new submissions and saving in-flight results", file=sys.stderr)
        for future, (task, provider, key_index) in list(futures.items()):
            if future.cancel():
                active[provider] -= 1
                active_key[(provider, key_index)] -= 1
                continue
            try:
                result = future.result()
            except Exception as error:
                task["attempt"] = 0
                result = base_result(task, 0, 0, {})
                result["error_type"] = "future_exception"
                result["error_message"] = redact_message(
                    f"{type(error).__name__}: {error}"
                )
            result["credential_slot"] = key_index + 1
            store.record(result)
            tracker.update(result)
            active[provider] -= 1
            active_key[(provider, key_index)] -= 1
            succeeded += int(bool(result.get("success")))
            failed += int(not result.get("success"))
    finally:
        for executor in executors.values():
            executor.shutdown(wait=not stopping, cancel_futures=stopping)
        store.export_jsonl(args.output)
        store.export_history_jsonl(
            args.output.with_name(args.output.stem + ".history.jsonl")
        )
        atomic_write_text(args.progress_doc, tracker.render(active, stopped=True))
        remaining_after_run = len(planned_ids.difference(store.completed_ids()))
        terminal_failures_after_run = sum(
            1
            for call_id, result in store.latest_results().items()
            if call_id in planned_ids and result.get("terminal_failure")
        )
        store.close()
    print(
        f"finished new_success={succeeded} new_failed={failed} "
        f"remaining={remaining_after_run} terminal_invalid={terminal_failures_after_run} "
        f"phase_cap={args.stop_after_successes_per_provider} "
        f"checkpoint={checkpoint_path} "
        f"progress={args.progress_doc}"
    )
    phase_cap_reached = (
        args.stop_after_successes_per_provider > 0
        and all(
            successful_by_provider[provider]
            >= args.stop_after_successes_per_provider
            for provider in providers
        )
    )
    return (
        130
        if stopping
        else 0
        if remaining_after_run == 0 and terminal_failures_after_run == 0
        else 3
        if phase_cap_reached and terminal_failures_after_run == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
