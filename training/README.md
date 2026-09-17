# Final SEG010 training package

This directory is the public training implementation for the final SEG010
General and Aggregation specialists. The historical experiment filenames are
kept in provenance outside this public entry point; the supported public
entry point is `train.py`.

These historical-named modules are retained as internal dependencies of the
final trainer; `training/train.py` is the supported public entry point.

The state packs and locked manifests use Git LFS. Run `git lfs pull` from the
repository root after cloning.

## Requirements

Install the root inference requirements plus the training additions in an
environment with a CUDA-enabled PyTorch build:

```bash
pip install --requirement requirements.txt
pip install --requirement requirements-training.txt
```

`PyNvVideoCodec` must be imported before `torch` or CUDA model initialization.
The public trainer preserves this import-order contract. Use four visible
GPUs; the trainer fails closed for another world size.

## Inputs

The exact final stream manifests are under `artifacts/final_manifests/`.
They contain QA metadata and `${FOCUS_DATA_ROOT}/...` video-path placeholders;
the videos themselves must be supplied from the official challenge data/cache.
The scope and schedule definitions are under `artifacts/scope/`.

The canonical zero-output union initialization is under
`weights/canonical_union/`. The final General and Aggregation state packs are
under `weights/`.

Set `FOCUS_DATA_ROOT` and `SEGMENT_CACHE_ROOT` when the external data/cache
are not mounted at `/data/focus` and `/cache`.

## Final recipes

From the repository root:

```bash
torchrun --standalone --nproc_per_node=4 training/train.py \
  --recipe general \
  --output-dir outputs/SEG010_FINAL_GENERAL

torchrun --standalone --nproc_per_node=4 training/train.py \
  --recipe aggregation \
  --output-dir outputs/SEG010_FINAL_AGGREGATION
```

The recipes use the final SEG010 contract:

- Qwen3-VL-8B-Instruct at the pinned revision, BF16
- P2 evidence instruction and T0 absolute source-procedure timestamps
- AdamW, no scheduler, W0 unweighted answer+EOT mean NLL for training
- language LoRA LR `1e-4`, vision LoRA LR `1e-5`, merger LR `5e-5`
- global batch 8, four ranks, microbatch 1 per rank, accumulation 2
- PF-C main-thread decode with depth-one background PIL/processor prefetch
- complete recovery checkpoints every 250 optimizer updates
- independent stage-boundary and final checkpoints

The evaluation primary used to select the final recipe is answer-content NLL
with equal weighting over the capability-by-dataset cells. This is distinct
from the training loss definition.

For a short smoke run, add `--max-updates 1`. A normal run omits that option.
Use `--resume` with a complete checkpoint directory to resume; the trainer
restores optimizer, schedule, cursor, stage, and per-rank RNG state.

Use separate output directories when running both recipes, as shown above.
The trainer's default output directory remains `outputs/SEG010_FINAL/`; output
directories are intentionally ignored by the public Git repository.
