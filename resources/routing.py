"""Answer-format hints inferred from the question only."""

from __future__ import annotations

import re
from dataclasses import dataclass

FO_CLASS_NAMES = (
    "Sponge", "Clip", "Specimen Bag", "Silicone Loop", "External Drain",
    "Needle", "Gallstone", "Specimen", "Mesh", "Absorbable Hemostatic Agent",
)


_BINARY_CUE = r"yes\s+or\s+no|answer\s+with\s+yes|answer\s+with\s+'yes'|answer\s+with\s+\"yes\""
_NUMBER_CUE = r"provide\s+(?:a|the)\s+number|single\s+integer|how\s+many|number\s+of|maximum\s+number|total\s+count"
_FO_CLASS_CUE = r"class\s+name|class(?:es)?(?:\s+(?:appear|are|is))|foreign\s+object\s+class"
_TIME_CUE = (
    r"hh:mm:ss|\bat\s+what\s+time\b|\bat\s+which\s+time\b|\bwhat\s+time\b|"
    r"\btimestamp\b|\btime\s+points?\b|\bwhen\s+(?:was|were|did)\s+.*\b"
    r"(?:inserted|retrieved|removed|visible|appeared|occurred)\b"
)
_MULTI_VALUE_NUMERIC_CUE = (
    r"\btwo\s+(?:non-negative\s+)?integers?\b|\btwo\s+numbers?\b|"
    r"\bP\s*,\s*D\b|\bproximal\s*,\s*distal\b|"
    r"\bproximal\s*:\s*.*?\bdistal\s*:"
)
_EXPLICIT_REASONING_CUE = (
    r"\b(?:why|explain|because|reason|explanation)\b|"
    r"\bshort\s+description\b|\bbrief\s+explanation\b|\bvisual\s+evidence\b"
)
_BINARY_SECONDARY_DETAIL_CUE = (
    r"\bspecify\b|\bif\s+(?:the\s+answer\s+is\s+)?['\"]?yes\b|"
    r"\bprovide\s+(?:the\s+)?(?:class|object)\b|\bwhere\b|\bwhich\b|"
    r"\battribute\b|\bdetail\b"
)
_NUMBER_NON_SCALAR_CUE = (
    r"\b(?:describe|sequence)\b|\bmethod\s*=|\bstructure:\b|"
    r"\bordered\s+step\s+list\b|\bwhat\s+.*\s+and\s+what\b"
)
_FO_EXPLANATION_CUE = (
    r"\b(?:why|explain|because|reason|explanation|complication)\b|"
    r"\bbrief\s+explanation\b"
)
_FO_SECOND_FIELD_CUE = (
    r"\bstructure:\b|\bobject:\b|\btwo\s+fields\b|"
    r"\band\s+what\s+object\b"
)
_TIME_EVENT_DESCRIPTION_CUE = (
    r"\b(?:short\s+description|trigger|operative\s+event|event|"
    r"what\s+happened|reason)\b|\bwithout\s+additional\s+explanation\b"
)


def _has_both(question: str, first: str, second: str) -> bool:
    return _has(question, first) and _has(question, second)


def _has_conservative_open_ended_guard(question: str) -> bool:
    """Return whether explicit response structure requires free text.

    These guards are intentionally question-only and conservative.  They are
    ordered independently from the scalar routing rules so ``how many`` does
    not hide an explicit two-value or structured response contract.
    """

    return any(
        (
            _has_both(question, _NUMBER_CUE, _MULTI_VALUE_NUMERIC_CUE),
            _has_both(question, _BINARY_CUE, _EXPLICIT_REASONING_CUE),
            _has_both(question, _BINARY_CUE, _BINARY_SECONDARY_DETAIL_CUE),
            _has_both(question, _NUMBER_CUE, _NUMBER_NON_SCALAR_CUE),
            _has_both(question, _FO_CLASS_CUE, _FO_EXPLANATION_CUE),
            _has_both(question, _FO_CLASS_CUE, _FO_SECOND_FIELD_CUE),
            _has_both(question, _TIME_CUE, _TIME_EVENT_DESCRIPTION_CUE),
        )
    )


def _has(question: str, pattern: str) -> bool:
    return bool(re.search(pattern, str(question), re.IGNORECASE))


def infer_answer_format(question: str) -> str:
    q = str(question)
    if _has(q, r"\bPlease\s+select\s+(?:none\s*,\s*)?(?:one\s+answer|one\s+or\s+multiple\s+answers)\s*:"):
        return "multiple_choice"
    if _has(q, r"\bpercent(?:age)?\b|%"):
        return "percentage"
    if _has_conservative_open_ended_guard(q):
        return "open_ended"
    if _has(q, _BINARY_CUE):
        return "binary"
    if _has(q, _NUMBER_CUE):
        return "number"
    if _has(q, r"hh:mm:ss|\bat what time\b|\bat which time\b|\bwhat time\b|\btimestamp\b|\btime points?\b|\bwhen (?:was|were|did) .*\b(?:inserted|retrieved|removed|visible|appeared|occurred)\b"):
        return "time"
    if _has(q, r"chronological order|in what order|order .* first appear|first to last"):
        return "open_ended"
    if _has(q, r"class name|class(?:es)?(?: appear| are| is)|foreign object class"):
        return "fo_class"
    return "open_ended"


def infer_time_subtype(question: str) -> str:
    q = str(question)
    if _has(
        q,
        r"\bfor how long\b|\bhow long\b|\btotal duration\b|\bduration\b|"
        r"\btime intervals?\b|\badding together\b|\bhow many (?:seconds|minutes)\b|"
        r"\bhow much time (?:passes|passed|elapses|elapsed)\b|"
        r"\btime (?:passes|passed|elapses|elapsed) between\b|\btime elapsed\b",
    ):
        return "duration"
    if _has(q, r"\btime points\b|\btimepoints\b|\beach individual\b|\bmultiple timestamps?\b|\bfrom when to when\b|\bfirst and last\b|\bstart and end\b"):
        return "multi_point"
    if _has(q, r"\bat what time\b|\bwhat time\b|\bwhen (?:was|were|did|does|is|are)\b|\btimestamp\b|\btime point\b"):
        return "single_point"
    return "unknown"


# The semantic route below is the question-only subset of the SEG002c router.
# It is kept in the submission package so inference never imports development
# metadata or the training-side source tree.
_COMPLEX_PRIMARY = frozenset(
    {
        "causal_consequence_reasoning",
        "functional_reasoning",
        "multi_step_reasoning",
        "fo_usage_purpose",
    }
)
_TEMPORAL_PRIMARY = frozenset(
    {
        "temporal_localization",
        "temporal_ordering",
        "fo_interaction_recognition",
        "instance_matching",
    }
)

_STATIC_D_HIRES_FINE = frozenset(
    {
        "object_attributes",
        "object_identity_presence",
    }
)

_FO_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bspecimen\s+bags?\b", "Specimen Bag"),
    (r"\bsilicone\s+loops?\b", "Silicone Loop"),
    (r"\bexternal\s+drains?\b", "External Drain"),
    (r"\babsorbable\s+hemostatic\s+agents?\b", "Absorbable Hemostatic Agent"),
    (r"\bgallstones?\b", "Gallstone"),
    (r"\bsponges?\b", "Sponge"),
    (r"\bclips?\b", "Clip"),
    (r"\bneedles?\b", "Needle"),
    (r"\bspecimens?\b", "Specimen"),
    (r"\bmeshes?\b", "Mesh"),
)
_ACTUAL_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")
_TIME_RE = re.compile(r"\b(?:\d{1,2}:\d{2}:\d{2}|hh:mm:ss)\b", re.IGNORECASE)
_ORDINAL_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|last|earliest|latest|initial|final|"
    r"1st|2nd|3rd|4th|5th)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


@dataclass(frozen=True)
class Routing:
    """Question-only semantic and sampling route for one request."""

    route_coarse: str
    route_medium: str
    route_fine: str
    scope_class: str
    sampling_policy: str
    target_object: str
    secondary_object: str
    event_type: str
    temporal_relation: str
    aggregation_type: str
    routing_confidence: str
    parser_basis: str
    notes: str


def extract_objects(question: str) -> tuple[str, ...]:
    """Return explicit foreign-object classes in question order."""

    matches: list[tuple[int, int, str]] = []
    lowered = str(question).casefold()
    for pattern, canonical in _FO_PATTERNS:
        for match in re.finditer(pattern, lowered, re.IGNORECASE):
            matches.append((match.start(), -(match.end() - match.start()), canonical))
    matches.sort()
    result: list[str] = []
    for _, _, canonical in matches:
        if canonical not in result:
            result.append(canonical)
    return tuple(result)


def normalized_template(question: str) -> str:
    """Canonicalize common slots for deterministic route diagnostics."""

    value = str(question).casefold()
    value = _TIME_RE.sub("<time>", value)
    value = _ORDINAL_RE.sub("<ordinal>", value)
    for pattern, _ in _FO_PATTERNS:
        value = re.sub(pattern, "<fo>", value, flags=re.IGNORECASE)
    value = _NUMBER_RE.sub("<number>", value)
    value = re.sub(r"[\"'“”]", "", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value


def _infer_primary(question: str, answer_format: str) -> str:
    q = str(question)
    fmt = str(answer_format)
    if _has(q, r"\bwhy\b|\bpurpose\b|\bfunction\b|\breason for\b|what happens to"):
        return "functional_reasoning"
    if _has(q, r"remain intra-abdominal|left intra-abdominal|should not be removed|remain in the body"):
        return "causal_consequence_reasoning"
    if _has(q, r"chronological order|in what order|order of|before .* already"):
        return "temporal_ordering"
    if _has(q, r"quadrant|image center|camera view|relative central position"):
        return "spatial_localization_camera"
    if _has(q, r"anatomical structure|where.*located|encircling|around what structure|which organ"):
        return "spatial_localization_situs"
    if _has(q, r"\bcolor\b|predominant(?:ly)?"):
        return "object_attributes"
    if _has(q, r"re-?appear|re-?enter|also appear|leave the field"):
        return "instance_matching"
    if fmt == "percentage" or _has(q, r"for how long|how much time|total duration"):
        return "duration_estimation"
    if fmt == "number" and _has(q, r"leave.*return|re-enter|retriev"):
        return "event_aggregation"
    if fmt == "number" or _has(q, r"how many|number of|maximum number|total count"):
        return "object_aggregation"
    if fmt == "time" or _has(q, r"first visible|last visible|when .* inserted|when .* retrieved"):
        return "temporal_localization"
    if _has(q, r"insert\w*|retriev\w*|co-?occur|same time|being used|sutured"):
        return "fo_interaction_recognition"
    if fmt == "fo_class" or _has(q, r"foreign object|\bSponge\b|\bClip\b|\bNeedle\b"):
        return "object_identification"
    return ""


def _event_type(question: str) -> str:
    q = str(question)
    if _has(q, r"re-?appear|re-?enter|leave the field|leave the surgical view"):
        return "reappearance"
    if _has(q, r"insert\w*|created|placed|appl(?:y|ied)|deployed|introduced"):
        return "insertion"
    if _has(q, r"retriev\w*|removed|taken out|should not be removed"):
        return "retrieval"
    if _has(q, r"co-?occur|same time|at the same time|simultaneous"):
        return "cooccurrence"
    if _has(q, r"first\s+appear|last\s+visible|becomes visible|appearing"):
        return "appearance"
    if _has(q, r"used|purpose|encircl|sutured|suturing|being clipped|cover"):
        return "manipulation"
    return "none"


def _temporal_relation(question: str) -> str:
    q = str(question)
    terms: list[str] = []
    for name, pattern in (
        ("before", r"\bbefore\b"),
        ("after", r"\bafter\b"),
        ("between", r"\bbetween\b"),
        ("during", r"\bduring\b|\bwhile\b"),
        ("at_anchor", r"\bat\s+(?:timepoint|time\s+point)|\bframe\s+\d{1,2}:\d{2}:\d{2}"),
        ("first", r"\bfirst\b|\bearliest\b|\b1st\b"),
        ("last", r"\blast\b|\blatest\b|\bfinal\b"),
        ("same_time", r"same time|at the same time|co-?occur"),
        ("ordering", r"chronological order|in what order|order of"),
    ):
        if _has(q, pattern):
            terms.append(name)
    return ";".join(terms) if terms else "none"


def _aggregation_type(question: str, answer_format: str, primary: str) -> str:
    q = str(question)
    fmt = str(answer_format)
    if fmt == "percentage" or _has(q, r"\bpercent(?:age)?\b|\bin %\b|%"):
        return "percentage"
    if primary == "duration_estimation" or _has(q, r"for how long|how much time|total duration"):
        return "duration"
    if fmt == "number" or _has(q, r"how many|number of|count|maximum|total count"):
        return "count"
    if primary == "temporal_ordering" or _has(q, r"chronological order|in what order"):
        return "ordering"
    if _has(q, r"any frame|any foreign object|which classes appear|what types"):
        return "presence_or_inventory"
    if fmt == "time":
        return "timestamp"
    return "none"


def _is_explicit_restriction(question: str) -> bool:
    q = str(question)
    return bool(
        _ACTUAL_TIME_RE.search(q)
        or _has(q, r"\bbefore\b|\bafter\b|\bbetween\b|\bwhile\b|\buntil\b|\bfrom\b.+\bto\b|at timepoint|at time point")
    )


def _fine_route(question: str, answer_format: str, primary: str, event: str, aggregation: str) -> str:
    q = str(question)
    fmt = str(answer_format)
    if fmt == "percentage" or aggregation == "percentage":
        return "occupancy_percentage"
    if primary in _COMPLEX_PRIMARY:
        return "clinical_reasoning"
    if primary == "spatial_localization_camera":
        return "spatial_camera"
    if primary == "spatial_localization_situs":
        return "spatial_situs"
    if primary == "object_attributes":
        return "object_attributes"
    if primary == "instance_matching" or event == "reappearance":
        return "instance_matching"
    if _has(q, r"\bcolor\b|attribute"):
        return "object_attributes"
    if _has(q, r"anatomical structure|where.*located|encircling"):
        return "spatial_situs"
    if _has(q, r"quadrant|image center|camera view"):
        return "spatial_camera"
    if primary == "event_aggregation":
        return "event_count"
    if primary == "object_aggregation":
        return "object_count"
    if primary == "duration_estimation":
        return "duration"
    if primary == "temporal_ordering":
        return "temporal_ordering"
    if primary == "fo_interaction_recognition":
        return "event_occurrence"
    if primary == "temporal_localization":
        return "event_localization"
    if primary == "object_identification" and event in {"cooccurrence", "manipulation"}:
        return "event_occurrence"
    if primary == "object_identification" and event in {"insertion", "retrieval", "appearance"}:
        return "event_object_identity"
    if primary == "object_identification":
        return "object_identity_presence"
    if aggregation == "count":
        return "object_count"
    if aggregation == "duration":
        return "duration"
    if aggregation == "ordering":
        return "temporal_ordering"
    if event in {"insertion", "retrieval", "cooccurrence", "manipulation"}:
        return "event_occurrence"
    if fmt == "time":
        return "event_localization"
    return "general_other"


def _medium_route(fine: str) -> str:
    if fine in {"object_attributes", "object_identity_presence", "general_other"}:
        return "static_object_visual"
    if fine in {"spatial_camera", "spatial_situs"}:
        return "spatial_localization"
    if fine in {"object_count", "event_count", "occupancy_percentage", "duration"}:
        return "global_aggregation"
    if fine == "instance_matching":
        return "instance_tracking_matching"
    if fine in {"event_occurrence", "event_localization", "event_object_identity"}:
        return "event_localization_interaction"
    if fine == "temporal_ordering":
        return "temporal_relation_ordering"
    if fine == "clinical_reasoning":
        return "clinical_reasoning"
    return "static_object_visual"


def _scope_and_sampling(question: str, primary: str, medium: str, fine: str) -> tuple[str, str, str]:
    q = str(question)
    if primary in _COMPLEX_PRIMARY:
        return "5_complex_uncertain", "multi-interval", "complex capability or causal dependency"
    explicit = _is_explicit_restriction(q)
    global_wording = _has(
        q,
        r"whole video|throughout the video|all time intervals|longest.*duration|classes appear|what types.*seen|for most of the time|populated with|longest continuous time",
    )
    if medium == "global_aggregation" and not explicit:
        return "1_global_coverage_required", "uniform-global", "aggregate over the segment"
    if fine == "temporal_ordering" and not explicit:
        return "1_global_coverage_required", "uniform-global", "global chronological scan"
    if global_wording and medium in {"static_object_visual", "spatial_localization"} and not explicit:
        return "1_global_coverage_required", "uniform-global", "question explicitly quantifies the video"
    if explicit and not _has(q, r"\bfirst\s+appear|\blast\s+visible"):
        if _ACTUAL_TIME_RE.search(q) or _has(q, r"timepoint|time point|at frame"):
            return "3_question_defined_restriction", "anchor-centered", "explicit timestamp anchor"
        return "3_question_defined_restriction", "interval-restricted", "explicit before/after/between restriction"
    if medium in {"event_localization_interaction", "instance_tracking_matching", "temporal_relation_ordering"}:
        return "2_local_event_search", "coarse-to-fine", "global coarse scan followed by local dense search"
    if medium == "spatial_localization" and _has(q, r"first|last|moment|timepoint"):
        return "2_local_event_search", "coarse-to-fine", "spatial evidence tied to an event boundary"
    if medium == "static_object_visual":
        return "4_sparse_representative_sufficient", "sparse-global", "representative frames likely sufficient"
    return "5_complex_uncertain", "unknown-general", "no safe deterministic scope reduction"


def route_question(question: str) -> Routing:
    """Return a semantic route from question text only.

    This deliberately has no official capability, answer-format metadata, or
    answer-bearing argument.  It is the only route input used by inference.
    """

    q = str(question)
    answer_format = infer_answer_format(q)
    primary = _infer_primary(q, answer_format)
    objects = extract_objects(q)
    event = _event_type(q)
    relation = _temporal_relation(q)
    aggregation = _aggregation_type(q, answer_format, primary)
    fine = _fine_route(q, answer_format, primary, event, aggregation)
    medium = _medium_route(fine)
    if medium == "clinical_reasoning":
        coarse = "complex_reasoning"
    elif medium == "global_aggregation":
        coarse = "global_coverage_aggregation"
    elif medium in {"event_localization_interaction", "instance_tracking_matching", "temporal_relation_ordering"}:
        coarse = "local_event_temporal"
    else:
        coarse = "static_visual_perception"
    scope, sampling, scope_note = _scope_and_sampling(q, primary, medium, fine)
    basis = "question_wording"
    strong_rule = fine != "general_other" and (bool(primary) or answer_format in {"time", "number", "percentage"})
    confidence = "high" if strong_rule and fine != "clinical_reasoning" else "medium"
    if fine == "general_other":
        confidence = "low"
    notes = scope_note
    if len(objects) > 2:
        notes += "; multiple explicit FO classes"
    if primary in _COMPLEX_PRIMARY:
        notes += "; specialist scope should be validated against examples"
    return Routing(
        route_coarse=coarse,
        route_medium=medium,
        route_fine=fine,
        scope_class=scope,
        sampling_policy=sampling,
        target_object=objects[0] if objects else "",
        secondary_object=objects[1] if len(objects) > 1 else "",
        event_type=event,
        temporal_relation=relation,
        aggregation_type=aggregation,
        routing_confidence=confidence,
        parser_basis=basis,
        notes=notes,
    )


def is_static_d_hires(route: Routing) -> bool:
    """Return whether the conservative SEG002c D condition applies."""

    return (
        route.route_fine in _STATIC_D_HIRES_FINE
        and route.sampling_policy == "sparse-global"
        and route.routing_confidence == "high"
    )
