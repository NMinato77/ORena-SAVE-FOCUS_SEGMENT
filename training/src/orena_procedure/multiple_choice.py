"""Question-only parsing for SEGMENT multiple-choice questions.

The parser is deliberately independent of answer/reference metadata.  It
extracts the semicolon-separated option list from the official question
templates and can canonicalize a model response against that list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SELECT_RE = re.compile(
    r"\bPlease\s+select\s+"
    r"(?P<mode>none\s*,\s*one\s+or\s+multiple\s+answers|"
    r"one\s+or\s+multiple\s+answers|one\s+answer)\s*:\s*"
    r"(?P<options>.+)\Z",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MultipleChoiceSpec:
    """The finite option set and selection semantics in a question."""

    options: tuple[str, ...]
    allows_multiple: bool
    allows_none: bool


def _normalize_whitespace(value: str) -> str:
    return " ".join(str(value).strip().split())


def _option_key(value: str) -> str:
    return _normalize_whitespace(value).casefold()


def extract_multiple_choice_spec(question: str) -> MultipleChoiceSpec:
    """Extract choices using only the question text.

    The current official SEGMENT templates use one of three ``Please
    select ...:`` clauses followed by semicolon-separated choices.  Unknown
    templates fail closed instead of guessing where the options begin.
    """

    normalized_question = _normalize_whitespace(question)
    match = _SELECT_RE.search(normalized_question)
    if match is None:
        raise ValueError("question has no supported multiple-choice selection clause")

    options = tuple(_normalize_whitespace(option) for option in match.group("options").split(";"))
    if len(options) < 2 or any(not option for option in options):
        raise ValueError("multiple-choice question must contain at least two non-empty options")
    keys = tuple(_option_key(option) for option in options)
    if len(set(keys)) != len(keys):
        raise ValueError("multiple-choice options must be unique")

    mode = _normalize_whitespace(match.group("mode")).casefold()
    allows_none = mode.startswith("none,")
    allows_multiple = allows_none or mode == "one or multiple answers"
    return MultipleChoiceSpec(
        options=options,
        allows_multiple=allows_multiple,
        allows_none=allows_none,
    )


def extract_multiple_choice_options(question: str) -> tuple[str, ...]:
    """Return the canonical option text in the order listed by the question."""

    return extract_multiple_choice_spec(question).options


def multiple_choice_prompt_instruction(question: str) -> str | None:
    """Build a strict output instruction when the question contains choices.

    ``None`` means that the question is not one of the supported
    multiple-choice templates.  Callers that know the official format is
    ``multiple_choice`` should call :func:`extract_multiple_choice_spec`
    directly and fail closed on ``ValueError``.
    """

    try:
        spec = extract_multiple_choice_spec(question)
    except ValueError:
        return None

    choices = "; ".join(spec.options)
    if not spec.allows_multiple:
        return (
            "Multiple-choice: output exactly one option text from this list: "
            f"{choices}. Do not output an option number, label, or explanation."
        )
    if spec.allows_none:
        return (
            "Multiple-choice: output one or more option texts from this list, "
            f"separated by commas and kept in listed order: {choices}. "
            "Output none when no option applies. Do not output option numbers, "
            "labels, or explanation."
        )
    return (
        "Multiple-choice: output one or more option texts from this list, "
        f"separated by commas and kept in listed order: {choices}. "
        "Do not output option numbers, labels, or explanation."
    )


def canonicalize_multiple_choice_answer(question: str, prediction: str) -> str | None:
    """Canonicalize an answer against question-derived choices.

    Only whitespace and case are normalized before matching.  Explanations,
    unknown options, duplicate options, and invalid empty answers return
    ``None``.  Multiple options are emitted in the order given by the
    question, making this suitable for an internal exact-match proxy.
    """

    spec = extract_multiple_choice_spec(question)
    option_by_key = {_option_key(option): option for option in spec.options}
    value = _normalize_whitespace(prediction)
    if spec.allows_none and value.casefold() == "none":
        return "none"
    if not value:
        return None

    pieces = tuple(_normalize_whitespace(piece) for piece in value.split(","))
    if any(not piece for piece in pieces):
        return None
    if not spec.allows_multiple and len(pieces) != 1:
        return None
    keys = tuple(_option_key(piece) for piece in pieces)
    if len(set(keys)) != len(keys) or any(key not in option_by_key for key in keys):
        return None
    selected = {key for key in keys}
    return ", ".join(option for option in spec.options if _option_key(option) in selected)
