"""Calculator tool: evaluate a safe arithmetic expression.

Pure and offline — no LLM, no network — so it reports 0 tokens to the trace.
A restricted AST evaluator (no names, calls, or attribute access) keeps `eval`
out of the picture.
"""
from __future__ import annotations

import ast
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


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):  # numbers only
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric literals are allowed")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the numeric result.

    Supports + - * / // % ** and parentheses over numbers only. Use this for
    quiz questions that reduce to a small computation (sums, products,
    percentages, powers). Returns an error string on an invalid expression.

    Args:
        expression: An arithmetic expression, e.g. "12 * (3 + 4)" or "2 ** 10".
    """
    expr = (expression or "").strip()
    if not expr:
        return "No expression given."
    try:
        result = _eval(ast.parse(expr, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError) as exc:
        return f"Could not evaluate {expr!r}: {exc}"
    # Render whole floats without a trailing ".0".
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)
