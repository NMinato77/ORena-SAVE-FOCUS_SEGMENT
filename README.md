# ORena SAVE FOCUS — SEGMENT Track

This repository contains the public SEGMENT-track submission and the
reproducibility package for the final SEG010 method submitted to the
[ORena SAVE FOCUS Challenge — SEGMENT Track](https://segment.orena-focus-challenge.org/).

The submission path is kept at the repository root: it contains the Docker
entrypoint, inference code, routing, format reduction, configuration, and
official sample fixtures. The `training/` tree contains the final General and
Aggregation training entry point, its source dependencies, locked manifests,
training configuration, and the final specialist state packs.

## Repository layout

```text
.
├── inference.py                 # Docker inference entry point
├── resources/                   # submission-time code and small router assets
├── training/
│   ├── train.py                # final trainer: --recipe general|aggregation
│   ├── src/orena_procedure/    # source dependencies of the trainer
│   ├── artifacts/              # locked scope and path-portable manifests
│   ├── configs/                # final training and prompt specifications
│   └── weights/                # canonical init and specialist state packs
├── docs/                       # reproducibility notes
└── test/                       # official sample fixtures
```

The Qwen base model and the SigLIP encoder are public pretrained assets and
are intentionally not duplicated in Git. The final specialist weights are
included in the `training/weights/` release assets and are also copied into
the Docker build context by `prepare_assets.sh`.

The specialist packs and locked training manifests are Git LFS assets. After
cloning, run `git lfs install` and `git lfs pull` before training or building
the submission image.

## External data and model-cache layout

The repository does not redistribute the challenge videos. Download the
official HeiCo/LapChole data according to the challenge instructions and make
the following external layout available before training:

```text
/data/focus/
└── derived/procedure_5fps_cache/   # decoded 5-fps video cache used by manifests

/cache/
├── huggingface/                    # Qwen Hugging Face cache
└── models/
    └── siglip-base-patch16-384/    # local SigLIP snapshot
```

`FOCUS_DATA_ROOT` may be set to another data location. The checked-in
manifests use `${FOCUS_DATA_ROOT}/...` placeholders and are resolved by
`training/train.py` at runtime. `SEGMENT_CACHE_ROOT` may be set when the
standard `/cache` mount is unavailable; `SEGMENT_WORKSPACE_ROOT` controls the
fallback workspace used by `prepare_assets.sh`.

## Download the public pretrained assets

Use the exact Qwen revision used by the final training and inference code:

```bash
export HF_HOME=/cache/huggingface
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
```

Download the SigLIP encoder into the path expected by the training and H2
router code:

```bash
hf download google/siglip-base-patch16-384 \
  --local-dir /cache/models/siglip-base-patch16-384
```

The exact file hashes for the SigLIP snapshot and the small Q2 classifier are
recorded in `resources/q2_siglip_manifest.json`.

## Training

Training requires four visible CUDA devices and is launched with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=4 training/train.py --recipe general
torchrun --standalone --nproc_per_node=4 training/train.py --recipe aggregation
```

See [`training/README.md`](training/README.md) for dependencies, the final
recipe, resume behavior, and output locations. Training outputs are generated
locally and are not copied into the Docker image.

## Submission image

For the offline submission image, run `do_build.sh`. It automatically invokes
`prepare_assets.sh` to copy the locally available Qwen and SigLIP assets and
the released specialist packs into `resources/` before building the image. The
Docker image remains an inference-only artifact and does not execute training
code.

The submission runs without network access and uses a single GPU, as required
by the SEGMENT track. No development-only absolute host paths are required at
inference time.

## Local release checks

The CPU-only release contract checks can be run with `pytest` from the
repository root:

```bash
pytest -q tests/test_public_release.py
```

These checks validate the build chain, portable training manifests, asset
staging, recorded hashes, shell syntax, and the absence of private absolute
paths. They do not build the Docker image or require a GPU.
