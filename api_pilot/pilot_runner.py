#!/usr/bin/env python3
"""Legacy alias for the synthetic provider smoke-test runner.

Prefer ``python -m survey_audit api-smoke-test`` or
``api_pilot.api_smoke_test`` for new scripts.
"""

from __future__ import annotations

import sys

if __package__:
    from . import api_smoke_test as _implementation
else:  # Historical ``python api_pilot/pilot_runner.py`` execution.
    import api_smoke_test as _implementation

# Alias the module object so existing imports and monkeypatch targets behave
# exactly like the renamed implementation.
sys.modules[__name__] = _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
