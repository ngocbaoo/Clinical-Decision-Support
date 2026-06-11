"""
XML-backed LLM prompt store.

Every system prompt the models receive lives in src/prompts/*.xml — one <prompt> file with
one or more named <section> elements — instead of being inlined in code. This keeps wording
out of logic (easy to iterate / translate / A-B test) and gives one place to review every
instruction sent to an LLM.

Conventions:
  - Prompt bodies are wrapped in CDATA, so JSON examples, quotes and Vietnamese punctuation
    need no XML escaping.
  - Placeholders use mustache syntax {{NAME}} and are filled by str.replace (NOT str.format),
    so the literal { } braces inside JSON examples are left untouched.

Usage:
    from prompts import load_prompt
    SYSTEM_PROMPT = load_prompt("generation")
    DIRECTIVE = load_prompt("generation", section="scoring_directive")
    JUDGE = load_prompt("judge")
"""

import sys
from pathlib import Path
from xml.etree import ElementTree as ET

PROMPTS_DIR = Path(__file__).resolve().parent
_cache: dict[tuple[str, str], str] = {}


def load_prompt(name: str, section: str = "system", **subs) -> str:
    """Return the text of <section> in src/prompts/<name>.xml, with {{NAME}} substituted.

    Parsed prompts are cached (pre-substitution); substitution is applied per call.
    """
    key = (name, section)
    text = _cache.get(key)
    if text is None:
        path = PROMPTS_DIR / f"{name}.xml"
        try:
            root = ET.parse(path).getroot()
        except FileNotFoundError as err:
            raise FileNotFoundError(f"prompt file not found: {path}") from err
        el = root.find(section)
        if el is None or el.text is None:
            raise KeyError(f"prompt '{name}' has no <{section}> section ({path})")
        text = el.text.strip("\n")
        _cache[key] = text
    for k, v in subs.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text
