"""Load prompt template files shipped inside this package."""
from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def prompts_dir() -> Path:
    return _PROMPTS_DIR


def load_prompt(name: str) -> str:
    """Read a prompt file (e.g. ``intake_system.txt``) as text."""
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")
