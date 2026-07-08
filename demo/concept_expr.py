"""Per-pixel concept expression language for the CBM video demo.

Atoms resolve to concept probability maps from the model. Operators are
evaluated per-pixel with numpy. The sentinel ``@classes`` is handled outside
this module (see ``video_demo.py``).
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

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<number>\d+(?:\.\d+)?)
        | (?P<ident>[A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z0-9_]+)?)
        | (?P<lparen>\()
        | (?P<rparen>\))
        | (?P<op>[+\-*])
    )
    """,
    re.VERBOSE,
)


class _Kind(Enum):
    NUMBER = auto()
    IDENT = auto()
    OP = auto()


@dataclass(frozen=True)
class _Token:
    kind: _Kind
    value: str


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


def _to_rpn(tokens: Sequence[_Token]) -> list[_Token]:
    output: list[_Token] = []
    stack: list[str] = []
    for token in tokens:
        if token.kind in (_Kind.NUMBER, _Kind.IDENT):
            output.append(token)
            continue
        if token.value == "(":
            stack.append(token.value)
            continue
        if token.value == ")":
            while stack and stack[-1] != "(":
                output.append(_Token(_Kind.OP, stack.pop()))
            if not stack:
                raise ConceptExpressionError("Mismatched parentheses")
            stack.pop()
            continue
        while stack and stack[-1] != "(" and _precedence(stack[-1]) >= _precedence(token.value):
            output.append(_Token(_Kind.OP, stack.pop()))
        stack.append(token.value)
    while stack:
        op = stack.pop()
        if op == "(":
            raise ConceptExpressionError("Mismatched parentheses")
        output.append(_Token(_Kind.OP, op))
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
