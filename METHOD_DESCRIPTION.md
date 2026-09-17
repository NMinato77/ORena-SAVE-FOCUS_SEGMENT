# SEGMENT method description

The submission candidate uses an offline Qwen3-VL-8B base snapshot together
with the H2 question-only router. Q2 uses a locally packaged CPU SigLIP text
encoder and frozen classifier. The startup-only fallback chain is H2 → Q1-R5
→ Q0 rule router; prediction confidence, disagreement, voting, and
per-question fallback are not used.

Training used FOCUS challenge data and public pretrained model assets; detailed
checkpoint and router provenance is recorded in the public
[`release provenance manifest`](docs/release_provenance_manifest.json).

The operational routes are:

- Route A: Aggregation specialist with R0.
- Route B: General specialist with R0.
- Route C: General specialist with R1.

R0 uses the clip-relative fixed-1-fps timeline, at most 240 frames, and
`max_pixels=50176`. R1 selects the endpoint-preserving half of the R0 timeline
and uses `max_pixels=100352`. Prompt timestamps use the absolute
source-procedure timeline (`request.start_time + clip_frame_index / fps`).
The frozen evidence instruction is P2 and the timestamp serialization is T0.

Final General and Aggregation LMV specialist packs are runtime-bound. In
final-candidate mode, missing or invalid specialist assets produce an explicit
hard failure before base-model loading or forward execution. No temporary or
historical weight is used as an implicit replacement.

The official definitions input is read and hash-audited, but its full text is
not injected into the request-only prompt. Generated text is passed through
the question-only format reducer before the official response is written.
