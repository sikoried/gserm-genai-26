"""Unit-conversion tool: length, mass, and temperature.

Pure and offline (0 tokens). Length/mass go through a base unit (metre / gram);
temperature is special-cased because it is affine, not a simple scale.
"""
from __future__ import annotations

# Factor to the base unit (metres for length, grams for mass).
_LENGTH = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
           "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344}
_MASS = {"mg": 0.001, "g": 1.0, "kg": 1000.0, "t": 1_000_000.0,
         "oz": 28.349523125, "lb": 453.59237}
_TEMP = {"c", "f", "k"}


def _to_celsius(value: float, unit: str) -> float:
    return {"c": value, "f": (value - 32) / 1.8, "k": value - 273.15}[unit]


def _from_celsius(value: float, unit: str) -> float:
    return {"c": value, "f": value * 1.8 + 32, "k": value + 273.15}[unit]


def _fmt(value: float) -> str:
    rounded = round(value, 6)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:g}"


def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between common units of length, mass, or temperature.

    Supported units — length: mm, cm, m, km, in, ft, yd, mi; mass: mg, g, kg, t,
    oz, lb; temperature: c, f, k. The two units must be the same kind. Returns an
    error string for unknown or mismatched units.

    Args:
        value: The numeric quantity to convert.
        from_unit: The source unit (e.g. "km", "lb", "c").
        to_unit: The target unit (e.g. "mi", "kg", "f").
    """
    src = (from_unit or "").strip().lower()
    dst = (to_unit or "").strip().lower()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return f"Not a number: {value!r}."

    for table in (_LENGTH, _MASS):
        if src in table and dst in table:
            return f"{_fmt(value * table[src] / table[dst])} {dst}"
    if src in _TEMP and dst in _TEMP:
        return f"{_fmt(_from_celsius(_to_celsius(value, src), dst))} {dst}"

    if src not in _LENGTH and src not in _MASS and src not in _TEMP:
        return f"Unknown unit {from_unit!r}."
    if dst not in _LENGTH and dst not in _MASS and dst not in _TEMP:
        return f"Unknown unit {to_unit!r}."
    return f"Cannot convert {from_unit!r} to {to_unit!r} — different kinds of unit."
