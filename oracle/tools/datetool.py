"""Date tool: weekdays, differences, and date arithmetic.

Pure and offline (0 tokens). Handles the common "on which weekday…", "how many
days between…", and "N years/days before/after…" trivia. Dates are ISO
(``YYYY-MM-DD``).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

_OPERATIONS = ("today", "weekday", "difference", "shift")


def _parse(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def date_tool(operation: str, date: str = "", date2: str = "",
              days: int = 0, years: int = 0) -> str:
    """Compute weekdays, date differences, or shifted dates.

    Operations:
      - "today": current date (ISO), ignores other args.
      - "weekday": the weekday name of `date`.
      - "difference": absolute number of days between `date` and `date2`.
      - "shift": `date` shifted by `days` and/or `years` (use negatives to go back).

    Returns an error string for an unknown operation or unparseable date.

    Args:
        operation: One of "today", "weekday", "difference", "shift".
        date: The primary date as YYYY-MM-DD (not needed for "today").
        date2: The second date as YYYY-MM-DD (only for "difference").
        days: Day offset for "shift" (may be negative).
        years: Year offset for "shift" (may be negative).
    """
    op = (operation or "").strip().lower()
    if op not in _OPERATIONS:
        return f"Unknown operation {operation!r}. Use one of: {', '.join(_OPERATIONS)}."
    try:
        if op == "today":
            return datetime.now().date().isoformat()
        if op == "weekday":
            return _parse(date).strftime("%A")
        if op == "difference":
            return str(abs((_parse(date) - _parse(date2)).days))
        # shift
        d = _parse(date)
        try:
            d = d.replace(year=d.year + years)
        except ValueError:  # e.g. Feb 29 -> non-leap year
            d = d.replace(year=d.year + years, day=28)
        return (d + timedelta(days=days)).isoformat()
    except (ValueError, TypeError) as exc:
        return f"Could not compute date: {exc}"
