from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

# ── State ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
_loaded = False


# ── .env loading ─────────────────────────────────────────────────────────────
# Read KEY=VALUE lines from .env into os.environ (existing variables win)
def load_dotenv(path: str | Path = REPO_ROOT / ".env") -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    os.environ.setdefault("REPO_ROOT", str(REPO_ROOT))
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), os.path.expanduser(value))


# ── Config expansion ─────────────────────────────────────────────────────────
# Replace ${VAR} in every string of a nested config; fail on unset variables
def expand_config(value: Any) -> Any:
    load_dotenv()
    if isinstance(value, dict):
        return {k: expand_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_config(v) for v in value]
    if isinstance(value, str) and "$" in value:
        out = os.path.expandvars(value)
        missing = re.findall(r"\$\{?(\w+)\}?", out)
        if missing:
            raise SystemExit(f"{', '.join(missing)} not set: copy .env.example to .env at the repository root and fill it in.")
        return out
    return value
