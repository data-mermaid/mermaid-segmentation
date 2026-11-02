"""Per-pixel concept expression language for the CBM demos and evaluation.

Atoms resolve to concept probability maps from the model. Operators are
evaluated per-pixel with numpy: ``*`` multiplies, ``+``/``-`` add or subtract
(clamped to ``[0, 1]``), and ``max(a, b, ...)`` is the per-pixel maximum.
The sentinel ``@classes`` is handled outside this module (see ``demo/video_demo.py``).

This module is the canonical home for the DSL; ``demo/concept_expr.py`` re-exports
everything defined here so the demo scripts and their tests keep working.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from mermaidseg.dataset_reconciliation.concepts import parse_concept_rank

CLASSES_SENTINEL = "@classes"

_FUNCTIONS = frozenset({"max"})

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<number>\d+(?:\.\d+)?)
        | (?P<ident>[A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z0-9_]+)?)
        | (?P<lparen>\()
        | (?P<rparen>\))
        | (?P<comma>,)
        | (?P<op>[+\-*])
    )
    """,
    re.VERBOSE,
)


class _Kind(Enum):
    NUMBER = auto()
    IDENT = auto()
    OP = auto()
    FUNC = auto()


@dataclass(frozen=True)
class _Token:
    kind: _Kind
    value: str
    arity: int = 0


@dataclass
class _ParenFrame:
    """Tracks one open parenthesis while converting infix to RPN."""

    is_func: bool
    commas: int = 0
    has_content: bool = False


class ConceptExpressionError(ValueError):
    """Raised when an expression cannot be parsed or evaluated."""


class ConceptResolver:
    """Map expression atoms to concept channel indices."""

    def __init__(self, concept_names: Sequence[str]) -> None:
        self.concept_names = list(concept_names)
        self._by_name: dict[str, int] = {name: idx for idx, name in enumerate(self.concept_names)}

    def resolve(self, atom: str) -> int:
        if atom == CLASSES_SENTINEL:
            raise ConceptExpressionError(
                f"{CLASSES_SENTINEL!r} is a top-level sentinel and cannot appear inside an expression."
            )
        if ":" in atom:
            rank, value = atom.split(":", 1)
            channel_name = f"{rank}__{value}"
        else:
            channel_name = atom
        try:
            return self._by_name[channel_name]
        except KeyError as exc:
            raise ConceptExpressionError(
                f"Unknown concept atom {atom!r} (resolved to channel {channel_name!r}). "
                f"Known channels include: {', '.join(self.concept_names[:8])}..."
            ) from exc

    @classmethod
    def from_concept_names(cls, concept_names: Sequence[str]) -> ConceptResolver:
        return cls(concept_names)


def is_classes_sentinel(expression: str) -> bool:
    return expression.strip() == CLASSES_SENTINEL


def tokenize(expression: str) -> list[_Token]:
    pos = 0
    tokens: list[_Token] = []
    while pos < len(expression):
        match = _TOKEN_RE.match(expression, pos)
        if match is None:
            snippet = expression[pos : pos + 20]
            raise ConceptExpressionError(f"Unexpected token near {snippet!r} in {expression!r}")
        if match.group("number") is not None:
            tokens.append(_Token(_Kind.NUMBER, match.group("number")))
        elif match.group("ident") is not None:
            tokens.append(_Token(_Kind.IDENT, match.group("ident")))
        elif match.group("lparen") is not None:
            tokens.append(_Token(_Kind.OP, "("))
        elif match.group("rparen") is not None:
            tokens.append(_Token(_Kind.OP, ")"))
        elif match.group("comma") is not None:
            tokens.append(_Token(_Kind.OP, ","))
        elif match.group("op") is not None:
            tokens.append(_Token(_Kind.OP, match.group("op")))
        pos = match.end()
    return tokens


def _precedence(op: str) -> int:
    if op in ("+", "-"):
        return 1
    if op == "*":
        return 2
    return 0


def _mark_content(frames: list[_ParenFrame]) -> None:
    if frames:
        frames[-1].has_content = True


def _to_rpn(tokens: Sequence[_Token]) -> list[_Token]:
    output: list[_Token] = []
    stack: list[_Token] = []
    frames: list[_ParenFrame] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.kind == _Kind.IDENT and i + 1 < len(tokens) and tokens[i + 1].value == "(":
            if token.value not in _FUNCTIONS:
                raise ConceptExpressionError(f"Unknown function {token.value!r}")
            stack.append(_Token(_Kind.FUNC, token.value))
            i += 1
            continue
        if token.kind in (_Kind.NUMBER, _Kind.IDENT):
            output.append(token)
            _mark_content(frames)
            i += 1
            continue
        if token.value == ",":
            if not frames or not frames[-1].is_func:
                raise ConceptExpressionError("Misplaced comma")
            if not frames[-1].has_content:
                raise ConceptExpressionError("Empty function argument")
            while stack and stack[-1].value != "(":
                output.append(stack.pop())
            if not stack or stack[-1].value != "(":
                raise ConceptExpressionError("Mismatched parentheses")
            frames[-1].commas += 1
            frames[-1].has_content = False
            i += 1
            continue
        if token.value == "(":
            is_func = bool(stack) and stack[-1].kind == _Kind.FUNC
            stack.append(token)
            frames.append(_ParenFrame(is_func=is_func))
            i += 1
            continue
        if token.value == ")":
            while stack and stack[-1].value != "(":
                output.append(stack.pop())
            if not stack or not frames:
                raise ConceptExpressionError("Mismatched parentheses")
            stack.pop()
            frame = frames.pop()
            if frame.is_func:
                if not frame.has_content:
                    raise ConceptExpressionError("Empty function argument")
                if not stack or stack[-1].kind != _Kind.FUNC:
                    raise ConceptExpressionError("Mismatched parentheses")
                func = stack.pop()
                output.append(_Token(_Kind.FUNC, func.value, arity=frame.commas + 1))
                _mark_content(frames)
            i += 1
            continue
        while (
            stack
            and stack[-1].value != "("
            and stack[-1].kind != _Kind.FUNC
            and _precedence(stack[-1].value) >= _precedence(token.value)
        ):
            output.append(stack.pop())
        stack.append(token)
        i += 1
    while stack:
        op = stack.pop()
        if op.value == "(" or op.kind == _Kind.FUNC:
            raise ConceptExpressionError("Mismatched parentheses")
        output.append(op)
    return output


def parse(expression: str) -> list[_Token]:
    tokens = tokenize(expression.strip())
    if not tokens:
        raise ConceptExpressionError("Empty expression")
    return _to_rpn(tokens)


def _apply_binary(op: str, left: NDArray[np.float32], right: NDArray[np.float32]) -> NDArray[np.float32]:
    if op == "+":
        return np.clip(left + right, 0.0, 1.0)
    if op == "-":
        return np.clip(left - right, 0.0, 1.0)
    if op == "*":
        return np.clip(left * right, 0.0, 1.0)
    raise ConceptExpressionError(f"Unknown operator {op!r}")


def evaluate(
    expression: str,
    concept_probs: NDArray[np.float32],
    resolver: ConceptResolver | Mapping[str, int] | None = None,
) -> NDArray[np.float32]:
    """Evaluate an expression to a per-pixel map in ``[0, 1]``.

    Args:
        expression: Infix expression string.
        concept_probs: Shape ``(C, H, W)`` concept activations in ``[0, 1]``.
        resolver: Optional resolver or prebuilt atom->channel mapping.
    """
    if is_classes_sentinel(expression):
        raise ConceptExpressionError(
            f"Use is_classes_sentinel() and render classes separately for {CLASSES_SENTINEL!r}."
        )

    if resolver is None:
        raise ConceptExpressionError("ConceptResolver is required for evaluation.")

    resolve_fn: Callable[[str], int]
    if isinstance(resolver, ConceptResolver):
        resolve_fn = resolver.resolve
    else:
        mapping = resolver

        def resolve_fn(atom: str) -> int:
            if atom not in mapping:
                raise ConceptExpressionError(f"Unknown concept atom {atom!r}")
            return mapping[atom]

    rpn = parse(expression)
    stack: list[NDArray[np.float32]] = []
    for token in rpn:
        if token.kind == _Kind.NUMBER:
            value = float(token.value)
            stack.append(np.full(concept_probs.shape[1:], value, dtype=np.float32))
            continue
        if token.kind == _Kind.IDENT:
            channel_idx = resolve_fn(token.value)
            stack.append(concept_probs[channel_idx].astype(np.float32, copy=False))
            continue
        if token.kind == _Kind.FUNC:
            if token.value not in _FUNCTIONS:
                raise ConceptExpressionError(f"Unknown function {token.value!r}")
            if token.arity < 1 or len(stack) < token.arity:
                raise ConceptExpressionError(f"Not enough operands for function {token.value!r}")
            args = [stack.pop() for _ in range(token.arity)]
            args.reverse()
            reduced = np.maximum.reduce(args)
            stack.append(np.clip(reduced, 0.0, 1.0).astype(np.float32, copy=False))
            continue
        if len(stack) < 2:
            raise ConceptExpressionError(f"Not enough operands for operator {token.value!r}")
        right = stack.pop()
        left = stack.pop()
        stack.append(_apply_binary(token.value, left, right))
    if len(stack) != 1:
        raise ConceptExpressionError(f"Invalid expression {expression!r}")
    return stack[0]


def concept_names_from_id2name(concept_id2name: Mapping[int, str]) -> list[str]:
    return [name for _, name in sorted(concept_id2name.items(), key=lambda kv: int(kv[0]))]


def parse_concept_channel_name(name: str) -> tuple[str | None, str]:
    """Public wrapper around ``parse_concept_rank`` for tests and tooling."""
    return parse_concept_rank(name)
