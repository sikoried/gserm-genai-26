"""Calculator tool: evaluate a safe arithmetic/math expression.

Pure and offline — no LLM, no network — so it reports 0 tokens to the trace.
A restricted AST evaluator keeps ``eval`` out of the picture: only numbers, the
arithmetic operators, a **whitelist** of ``math`` functions, and the constants
``pi`` / ``e`` are allowed. No attribute access, no arbitrary names, no calls to
anything outside the whitelist.
"""
from __future__ import annotations

import ast
import math
import operator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Whitelisted callables (advanced maths). Variadic ones (min/max/sum/gcd/lcm)
# accept several arguments.
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "floor": math.floor, "ceil": math.ceil,
    "log": math.log, "log10": math.log10, "ln": math.log, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "radians": math.radians, "degrees": math.degrees,
    "factorial": math.factorial, "gcd": math.gcd, "lcm": math.lcm,
    "comb": math.comb, "nCr": math.comb, "perm": math.perm, "nPr": math.perm,
    "min": min, "max": max, "sum": lambda *a: sum(a),
    "pow": math.pow, "hypot": math.hypot,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):  # numbers only
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric literals are allowed")
        return node.value
    if isinstance(node, ast.Name):  # a whitelisted constant
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"unknown name {node.id!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("only whitelisted functions may be called")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic or math expression and return the numeric result.

    Supports + - * / // % ** and parentheses, plus a whitelist of functions —
    sqrt, abs, round, floor, ceil, log, log10, ln, exp, sin/cos/tan (+ inverses),
    radians, degrees, factorial, gcd, lcm, comb/nCr, perm/nPr, min, max, sum, pow,
    hypot — and the constants pi, e, tau. Use it whenever a quiz answer reduces to
    a computation. Returns an error string on an invalid expression.

    Args:
        expression: An expression, e.g. "12 * (3 + 4)", "sqrt(2) ** 2", or "comb(49, 6)".
    """
    expr = (expression or "").strip()
    if not expr:
        return "No expression given."
    try:
        result = _eval(ast.parse(expr, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError,
            OverflowError, RecursionError) as exc:
        return f"Could not evaluate {expr!r}: {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    elif isinstance(result, float):
        result = round(result, 10)
    return str(result)
