"""Question-aware answer contracts and deterministic answer reduction.

SEGMENT requests do not expose the reference-side ``answer_format`` at
inference time.  This module therefore supports two explicit modes:

* question-only mode, used by prompts and submission inference;
* official-format mode, used by offline evaluation when the reference format
  is legitimately available.

The reducer removes only harmless serialization differences for deterministic
formats.  It does not turn explanations into answers and does not attempt to
judge open-ended or matching answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from focus.data.formats import FOClass, Time, get_format_class

from orena_procedure.multiple_choice import (
    canonicalize_multiple_choice_answer,
    multiple_choice_prompt_instruction,
)
from orena_procedure.segment_routing import (
    FO_CLASS_NAMES,
    infer_answer_format,
    infer_time_subtype,
)

DETERMINISTIC_FORMATS = frozenset({"binary", "fo_class", "number", "percentage", "time"})
JUDGE_FORMATS = frozenset({"open_ended", "matching", "multiple_choice"})
SUPPORTED_FORMATS = DETERMINISTIC_FORMATS | JUDGE_FORMATS


@dataclass(frozen=True)
class ReducedAnswer:
    """Reduced prediction and diagnostics for one question."""

    raw: str
    content: str
    answer_format: str
    format_valid: bool
    judge_required: bool
    internal_exact_match_available: bool
    validation_mode: str
    canonical_value: Any = None
    error: str | None = None
    time_subtype: str | None = None
    request_window_valid: bool | None = None
    request_window_violation: bool | None = None


def _normalize_whitespace(value: str) -> str:
    return " ".join(str(value).replace("\x00", "").strip().split())


def _strip_outer_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] in "'\"`" and value[-1] == value[0]:
        return value[1:-1].strip()
    return value


def _strip_serialization_wrapper(value: str) -> str:
    """Remove one harmless answer wrapper without extracting explanations."""

    value = _normalize_whitespace(value)
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
        if value.lower().startswith("text "):
            value = value[5:].strip()
    value = _strip_outer_quotes(value)
    return re.sub(r"^(?:final\s+)?answer\s*:\s*", "", value, count=1, flags=re.I)


def _strip_scalar_punctuation(value: str) -> str:
    value = _strip_serialization_wrapper(value)
    while value.endswith((".", "。")):
        value = value[:-1].rstrip()
    return value


def generic_canonicalize(prediction: str) -> str:
    """Remove only answer-serialization wrappers without inferring a type.

    This is the format-aware experiment's F1 canonicalizer.  It deliberately
    does not inspect a reference-side ``answer_format`` or route from the
    question.  Type-specific parsing is applied only later, offline, by the
    official evaluator for diagnostic stratification.
    """

    value = _strip_scalar_punctuation(str(prediction))
    # ``The answer is ...`` is a serialization wrapper only in this generic
    # mode; the stricter question-aware reducer keeps its historical behavior.
    return re.sub(r"^(?:final\s+)?(?:the\s+)?answer\s+is\s*", "", value, count=1, flags=re.I)


def _canonical_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return format(value, ".15g")


def _canonical_time(value: tuple[timedelta, ...]) -> str:
    result: list[str] = []
    for item in value:
        seconds = int(item.total_seconds())
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        result.append(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
    return ", ".join(result)


def _canonical_fo_classes(value: frozenset[str]) -> str:
    if value == frozenset({FOClass.NONE}):
        return "none"
    known = {name.casefold(): name for name in FO_CLASS_NAMES}
    return ", ".join(known.get(name.casefold(), name) for name in FO_CLASS_NAMES if name in value)


def _formatter(answer_format: str, time_threshold_seconds: float) -> Any:
    if answer_format == "time":
        return Time(threshold_seconds=time_threshold_seconds)
    return get_format_class(answer_format)()


def _fallback_binary_candidate(value: str) -> str | None:
    yes = re.findall(r"\byes\b", value, re.IGNORECASE)
    no = re.findall(r"\bno\b", value, re.IGNORECASE)
    if bool(yes) == bool(no):
        return None
    return "yes" if yes else "no"


def _fallback_number_candidate(value: str) -> str | None:
    if re.search(r"\d+\.\d+", value):
        return None
    candidates = re.findall(r"(?<!\d)\d+(?!\d)", value)
    if len(candidates) != 1:
        return None
    return str(int(candidates[0]))


def _fallback_percentage_candidate(value: str) -> str | None:
    candidates = re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?=\s*%|\b)", value)
    if len(candidates) != 1:
        return None
    return candidates[0]


def _fallback_time_candidate(question: str, value: str) -> str | None:
    candidates = re.findall(r"(?<!\d)(\d{1,2}:\d{2}:\d{2})(?!\d)", value)
    subtype = infer_time_subtype(question)
    expected_count = 2 if subtype == "multi_point" else 1
    if len(candidates) != expected_count:
        return None
    normalized = []
    for candidate in candidates:
        hours, minutes, seconds = candidate.split(":")
        normalized.append(f"{int(hours):02d}:{minutes}:{seconds}")
    return ", ".join(normalized)


def _fallback_fo_class_candidate(value: str) -> str | None:
    none_matches = re.findall(r"\bnone\b", value, re.IGNORECASE)
    names_by_key = {name.casefold(): name for name in FO_CLASS_NAMES}
    alternatives = "|".join(re.escape(name) for name in sorted(FO_CLASS_NAMES, key=len, reverse=True))
    class_matches = [
        names_by_key[match.group(0).casefold()]
        for match in re.finditer(rf"\b(?:{alternatives})\b", value, re.IGNORECASE)
    ]
    if none_matches and class_matches:
        return None
    if len(class_matches) != len(set(class_matches)):
        return None
    if len(none_matches) > 1:
        return None
    if class_matches:
        return ", ".join(name for name in FO_CLASS_NAMES if name in class_matches)
    if len(none_matches) == 1:
        return "none"
    return None


def _conservative_fallback_candidate(
    answer_format: str,
    question: str,
    raw: str,
    time_threshold_seconds: float,
) -> tuple[str, Any] | None:
    """Extract one unambiguous scalar from a failed strict parse."""

    value = _normalize_whitespace(raw)
    if answer_format == "binary":
        candidate = _fallback_binary_candidate(value)
    elif answer_format == "number":
        candidate = _fallback_number_candidate(value)
    elif answer_format == "percentage":
        candidate = _fallback_percentage_candidate(value)
    elif answer_format == "time":
        candidate = _fallback_time_candidate(question, value)
    elif answer_format == "fo_class":
        candidate = _fallback_fo_class_candidate(value)
    else:
        return None
    if candidate is None:
        return None
    try:
        parsed = _formatter(answer_format, time_threshold_seconds).read(candidate)
    except (TypeError, ValueError, KeyError):
        return None
    return candidate, parsed


def _clock_text(seconds: float) -> str:
    """Render request bounds using the official whole-second clock format."""

    total = max(0, round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _time_instruction(
    question: str,
    request_start_seconds: float | None,
    request_end_seconds: float | None,
) -> str:
    subtype = infer_time_subtype(question)
    if subtype == "duration":
        return (
            "Return the total duration, not a timestamp on the procedure timeline. "
            "Output exactly one value as HH:MM:SS. Do not output an explanation."
        )
    if subtype == "single_point":
        instruction = (
            "Return exactly one timestamp in the original source-procedure timeline as "
            "HH:MM:SS. Do not return elapsed time from the beginning of this clip. "
            "Do not output an explanation."
        )
    elif subtype == "multi_point":
        instruction = (
            "Return all requested timestamps in chronological order as HH:MM:SS, "
            "separated by commas, one timestamp per requested event instance. "
            "Use the original source-procedure timeline, not elapsed clip time. "
            "Do not output an explanation."
        )
    else:
        instruction = (
            "Return the requested time value as HH:MM:SS. For event times, use the "
            "original source-procedure timeline, not elapsed clip time. Do not output "
            "an explanation."
        )
    instruction += " Example: 1 minute 43 seconds is 00:01:43, not 01:43:00."
    if (
        subtype != "duration"
        and request_start_seconds is not None
        and request_end_seconds is not None
    ):
        instruction += (
            f" The timestamp must be between {_clock_text(request_start_seconds)} and "
            f"{_clock_text(request_end_seconds)} inclusive."
        )
    return instruction


def format_instruction(
    question: str,
    answer_format: str | None = None,
    *,
    request_start_seconds: float | None = None,
    request_end_seconds: float | None = None,
) -> str:
    """Return a strict output instruction using question text when needed.

    ``answer_format`` is an optional audit/evaluation hint.  Submission
    callers should omit it so that the instruction remains answer-free.
    """

    fmt = str(answer_format) if answer_format else infer_answer_format(question)
    if fmt == "multiple_choice":
        instruction = multiple_choice_prompt_instruction(question)
        if instruction is None:
            raise ValueError("multiple-choice question has no supported option list")
        return instruction
    if fmt == "binary":
        return "Output exactly yes or no. Do not output an explanation."
    if fmt == "number":
        return "Output one non-negative integer only. Do not output an explanation."
    if fmt == "percentage":
        return "Output one numeric percentage only; the percent sign is optional."
    if fmt == "time":
        return _time_instruction(question, request_start_seconds, request_end_seconds)
    if fmt == "fo_class":
        names = ", ".join(FO_CLASS_NAMES)
        return f"Output only comma-separated foreign-object class names from: {names}; use none when absent."
    if fmt in {"open_ended", "matching"}:
        return "Output only the concise answer required by the question, without explanation."
    return "Answer only the question, without explanation."


def reduce_answer(
    question: str,
    prediction: str,
    answer_format: str | None = None,
    time_threshold_seconds: float = 5.0,
    request_start_seconds: float | None = None,
    request_end_seconds: float | None = None,
) -> ReducedAnswer:
    """Reduce and validate a prediction without using its reference answer."""

    raw = str(prediction)
    fmt = str(answer_format) if answer_format else infer_answer_format(question)
    clean = _strip_serialization_wrapper(raw)

    if fmt == "multiple_choice":
        try:
            canonical = canonicalize_multiple_choice_answer(question, clean)
        except (TypeError, ValueError) as exc:
            return ReducedAnswer(
                raw=raw,
                content=clean,
                answer_format=fmt,
                format_valid=False,
                judge_required=True,
                internal_exact_match_available=True,
                validation_mode="question_only_exact_match_proxy",
                error=str(exc),
            )
        if canonical is None:
            return ReducedAnswer(
                raw=raw,
                content=clean,
                answer_format=fmt,
                format_valid=False,
                judge_required=True,
                internal_exact_match_available=True,
                validation_mode="question_only_exact_match_proxy",
                error="prediction is not a canonical question-derived option",
            )
        return ReducedAnswer(
            raw=raw,
            content=canonical,
            answer_format=fmt,
            format_valid=True,
            judge_required=True,
            internal_exact_match_available=True,
            validation_mode="question_only_exact_match_proxy",
            canonical_value=canonical,
        )

    if fmt in DETERMINISTIC_FORMATS:
        candidate = _strip_scalar_punctuation(raw)
        time_subtype = infer_time_subtype(question) if fmt == "time" else None
        try:
            formatter = _formatter(fmt, time_threshold_seconds)
            parsed = formatter.read(candidate)
            if fmt == "binary":
                content = "yes" if parsed else "no"
            elif fmt == "number":
                content = str(parsed)
            elif fmt == "percentage":
                content = _canonical_number(float(parsed))
            elif fmt == "fo_class":
                content = _canonical_fo_classes(parsed)
            else:
                content = _canonical_time(parsed)
            request_window_valid: bool | None = None
            if (
                fmt == "time"
                and time_subtype != "duration"
                and request_start_seconds is not None
                and request_end_seconds is not None
            ):
                values = [item.total_seconds() for item in parsed]
                request_window_valid = bool(
                    values
                    and all(
                        float(request_start_seconds) <= value <= float(request_end_seconds)
                        for value in values
                    )
                )
            return ReducedAnswer(
                raw=raw,
                content=content,
                answer_format=fmt,
                format_valid=True,
                judge_required=False,
                internal_exact_match_available=False,
                validation_mode="official_formatter",
                canonical_value=parsed,
                time_subtype=time_subtype,
                request_window_valid=request_window_valid,
                request_window_violation=(
                    None if request_window_valid is None else not request_window_valid
                ),
            )
        except (TypeError, ValueError, KeyError) as exc:
            fallback = _conservative_fallback_candidate(
                fmt,
                question,
                raw,
                time_threshold_seconds,
            )
            if fallback is not None:
                _fallback_text, fallback_value = fallback
                if fmt == "binary":
                    fallback_content = "yes" if fallback_value else "no"
                elif fmt == "number":
                    fallback_content = str(fallback_value)
                elif fmt == "percentage":
                    fallback_content = _canonical_number(float(fallback_value))
                elif fmt == "fo_class":
                    fallback_content = _canonical_fo_classes(fallback_value)
                else:
                    fallback_content = _canonical_time(fallback_value)
                request_window_valid: bool | None = None
                if (
                    fmt == "time"
                    and time_subtype != "duration"
                    and request_start_seconds is not None
                    and request_end_seconds is not None
                ):
                    values = [item.total_seconds() for item in fallback_value]
                    request_window_valid = bool(
                        values
                        and all(
                            float(request_start_seconds) <= value <= float(request_end_seconds)
                            for value in values
                        )
                    )
                return ReducedAnswer(
                    raw=raw,
                    content=fallback_content,
                    answer_format=fmt,
                    format_valid=True,
                    judge_required=False,
                    internal_exact_match_available=False,
                    validation_mode="conservative_fallback",
                    canonical_value=fallback_value,
                    time_subtype=time_subtype,
                    request_window_valid=request_window_valid,
                    request_window_violation=(
                        None if request_window_valid is None else not request_window_valid
                    ),
                )
            return ReducedAnswer(
                raw=raw,
                content=candidate,
                answer_format=fmt,
                format_valid=False,
                judge_required=False,
                internal_exact_match_available=False,
                validation_mode="official_formatter",
                error=str(exc),
                time_subtype=time_subtype,
            )

    if fmt == "open_ended":
        valid = len(clean) <= 300
        return ReducedAnswer(
            raw=raw,
            content=clean,
            answer_format=fmt,
            format_valid=valid,
            judge_required=True,
            internal_exact_match_available=False,
            validation_mode="length_only",
            canonical_value=clean if valid else None,
            error=None if valid else "open-ended answer exceeds 300 characters",
        )

    if fmt == "matching":
        valid = len(clean) <= 300
        return ReducedAnswer(
            raw=raw,
            content=clean,
            answer_format=fmt,
            format_valid=valid,
            judge_required=True,
            internal_exact_match_available=False,
            validation_mode="length_only_without_reference_pattern",
            canonical_value=clean if valid else None,
            error=None if valid else "matching answer exceeds 300 characters",
        )

    return ReducedAnswer(
        raw=raw,
        content=clean,
        answer_format=fmt,
        format_valid=False,
        judge_required=False,
        internal_exact_match_available=False,
        validation_mode="unsupported_format",
        error=f"unsupported answer format: {fmt}",
    )


def compare_reduced_answers(
    question: str,
    reference: str,
    prediction: str,
    answer_format: str,
    time_threshold_seconds: float = 5.0,
    request_start_seconds: float | None = None,
    request_end_seconds: float | None = None,
) -> tuple[ReducedAnswer, ReducedAnswer, bool | None]:
    """Compare a candidate with a reference for deterministic/proxy formats.

    ``None`` is returned for open-ended and matching formats because they need
    an LLM judge (or a reference-specific matching pattern).
    """

    reference_reduced = reduce_answer(
        question,
        reference,
        answer_format,
        time_threshold_seconds,
        request_start_seconds,
        request_end_seconds,
    )
    prediction_reduced = reduce_answer(
        question,
        prediction,
        answer_format,
        time_threshold_seconds,
        request_start_seconds,
        request_end_seconds,
    )
    if answer_format in DETERMINISTIC_FORMATS:
        formatter = _formatter(answer_format, time_threshold_seconds)
        correct = (
            reference_reduced.format_valid
            and prediction_reduced.format_valid
            and formatter.compare(
                reference_reduced.canonical_value,
                prediction_reduced.canonical_value,
            )
        )
        return reference_reduced, prediction_reduced, bool(correct)
    if answer_format == "multiple_choice":
        correct = (
            reference_reduced.format_valid
            and prediction_reduced.format_valid
            and reference_reduced.canonical_value == prediction_reduced.canonical_value
        )
        return reference_reduced, prediction_reduced, bool(correct)
    return reference_reduced, prediction_reduced, None
