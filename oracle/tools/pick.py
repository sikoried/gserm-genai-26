"""List-pick tool: choose one item from a small candidate set by a criterion.

Pure and offline (0 tokens). For "which of these is the largest / earliest /
longest …" style quiz questions. Candidates are passed as a comma-separated
string so the tool signature stays simple for the router.
"""
from __future__ import annotations

import re

_CRITERIA = ("first", "last", "longest", "shortest",
             "alphabetical_first", "alphabetical_last", "largest", "smallest")


def _number(item: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", item)
    return float(m.group(0)) if m else None


def list_pick(items: str, criterion: str) -> str:
    """Pick one item from a comma-separated list by a selection criterion.

    Criteria: "first", "last", "longest", "shortest", "alphabetical_first",
    "alphabetical_last", "largest" / "smallest" (by the number found in each
    item). Returns an error string for an unknown criterion or empty list.

    Args:
        items: Comma-separated candidates, e.g. "Mercury, Venus, Earth".
        criterion: How to choose — see the list above.
    """
    candidates = [s.strip() for s in (items or "").split(",") if s.strip()]
    crit = (criterion or "").strip().lower()
    if not candidates:
        return "No candidates given."
    if crit not in _CRITERIA:
        return f"Unknown criterion {criterion!r}. Use one of: {', '.join(_CRITERIA)}."

    if crit == "first":
        return candidates[0]
    if crit == "last":
        return candidates[-1]
    if crit == "longest":
        return max(candidates, key=len)
    if crit == "shortest":
        return min(candidates, key=len)
    if crit == "alphabetical_first":
        return min(candidates, key=str.lower)
    if crit == "alphabetical_last":
        return max(candidates, key=str.lower)
    # largest / smallest by embedded number
    numbered = [(c, _number(c)) for c in candidates]
    numbered = [(c, n) for c, n in numbered if n is not None]
    if not numbered:
        return "No numbers found in the candidates to compare."
    chooser = max if crit == "largest" else min
    return chooser(numbered, key=lambda cn: cn[1])[0]
