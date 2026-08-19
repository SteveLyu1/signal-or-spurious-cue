#!/usr/bin/env python3
"""Probe Gemini model capacity with synthetic text and no survey-derived data."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_gemini_key(path: Path) -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import survey_pilot_runner as runner

    keys = runner.configured_keys(path).get("gemini") or []
    if not keys:
        raise ValueError("No Gemini key is configured")
    return keys[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, default=Path("api_pilot/apikey.txt"))
    parser.add_argument("--model", default="gemini-3.5-flash")
    args = parser.parse_args()
    key = load_gemini_key(args.key_file)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{args.model}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": "Return exactly the word READY."}]}],
        "generationConfig": {"maxOutputTokens": 8, "temperature": 0},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "X-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            status = response.status
            response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        body = error.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(body)
            message = str((data.get("error") or {}).get("message") or "")
        except json.JSONDecodeError:
            message = ""
        category = "quota" if status == 429 else "capacity" if status == 503 else "api"
        print(f"status={status} category={category} message={message[:180]}")
        return 10 if status in {429, 503} else 2
    except Exception as error:
        print(f"status=0 category=transport error={type(error).__name__}")
        return 3
    print(f"status={status} category=ready model={args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
