"""Unit tests for the bar-quiz tools — pure logic, no network, no smolagents.

Each tool gets a typical case and an empty/edge case (per the acceptance criteria
in tools.md).
"""
from oracle.tools.calculator import calculator
from oracle.tools.convert import unit_convert
from oracle.tools.datetool import date_tool
from oracle.tools.pick import list_pick
from oracle.tools.wiki_lookup import lookup


# --- calculator ---------------------------------------------------------------

def test_calculator_typical():
    assert calculator("12 * (3 + 4)") == "84"
    assert calculator("2 ** 10") == "1024"
    assert calculator("10 / 4") == "2.5"


def test_calculator_edge_cases():
    assert calculator("") == "No expression given."
    assert "Could not evaluate" in calculator("__import__('os')")  # no names allowed
    assert "Could not evaluate" in calculator("1/0")               # div by zero


def test_calculator_extended_functions():
    assert calculator("sqrt(144)") == "12"
    assert calculator("factorial(6)") == "720"
    assert calculator("comb(49, 6)") == "13983816"
    assert calculator("gcd(48, 36)") == "12"
    assert calculator("log10(1000)") == "3"
    assert calculator("max(3, 7, 5)") == "7"
    assert calculator("round(pi, 2)") == "3.14"


def test_calculator_rejects_unwhitelisted_calls_and_names():
    assert "Could not evaluate" in calculator("open('x')")        # not whitelisted
    assert "Could not evaluate" in calculator("os.system('ls')")  # attribute access
    assert "Could not evaluate" in calculator("foo")              # unknown name


# --- date_tool ----------------------------------------------------------------

def test_date_tool_typical():
    assert date_tool("weekday", date="1969-07-20") == "Sunday"
    assert date_tool("difference", date="2000-01-01", date2="2000-01-31") == "30"
    assert date_tool("shift", date="2020-01-01", years=5, days=-1) == "2024-12-31"


def test_date_tool_edge_cases():
    assert "Unknown operation" in date_tool("frobnicate")
    assert "Could not compute" in date_tool("weekday", date="not-a-date")


# --- unit_convert -------------------------------------------------------------

def test_unit_convert_typical():
    assert unit_convert(1, "mi", "km") == "1.60934 km"
    assert unit_convert(100, "c", "f") == "212 f"
    assert unit_convert(1000, "g", "kg") == "1 kg"


def test_unit_convert_edge_cases():
    assert "different kinds of unit" in unit_convert(1, "kg", "m")
    assert "Unknown unit" in unit_convert(1, "smoot", "m")


# --- list_pick ----------------------------------------------------------------

def test_list_pick_typical():
    assert list_pick("Mercury, Venus, Earth", "alphabetical_first") == "Earth"
    assert list_pick("cat, hippopotamus, dog", "longest") == "hippopotamus"
    assert list_pick("year 1990, year 1066, year 1815", "smallest") == "year 1066"


def test_list_pick_edge_cases():
    assert list_pick("", "first") == "No candidates given."
    assert "Unknown criterion" in list_pick("a, b", "weirdest")
    assert "No numbers found" in list_pick("alpha, beta", "largest")


# --- wiki_lookup (pure helper, fake chunks) -----------------------------------

class _Chunk:
    def __init__(self, title, text):
        self.title = title
        self.text = text


def test_wiki_lookup_typical():
    chunks = [_Chunk("France", "France is a country in Europe."),
              _Chunk("Germany", "Germany is in Europe.")]
    out = lookup("france", chunks)  # case-insensitive exact match
    assert out.startswith("France")
    assert "country in Europe" in out


def test_wiki_lookup_edge_cases():
    assert lookup("", []) == "No title given."
    assert "No article titled" in lookup("Atlantis", [_Chunk("France", "...")])
