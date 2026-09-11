FROM --platform=linux/amd64 pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime AS segment-algorithm-amd64

# torch 2.11.0 + torchvision are pre-installed in the base image (CUDA 12.8).
# requirements.txt only lists additional dependencies.
#
# This base image is the recommended default, not a hard requirement. It is chosen
# because it is the ONE PyTorch build that runs on both evaluation GPUs unmodified:
# an NVIDIA RTX PRO 6000 Blackwell (compute capability sm_120) and an NVIDIA L40S
# (sm_89). Its CUDA 12.8 kernels cover sm_75/80/86/90/100/120 — Blackwell natively
# and the L40S via its sm_86 kernels (a CUDA binary runs on any later minor
# revision of the same major capability). So one image serves whichever GPU you
# select on the platform, with no rebuild.
#
# If you change it, know the two limits it steers around: CUDA 12.4/12.6 wheels
# stop at sm_90 and abort on Blackwell ("not compatible with the current PyTorch
# installation"); CUDA 13.0 wheels cover both GPUs but need a host driver >= 580 —
# the RTX PRO 6000 fleet has it (595), the L40S fleet does not. If your submission
# targets ONLY the RTX PRO 6000 (offered on the SEGMENT and PROCEDURE tracks), a
# newer CUDA/torch is a valid choice — see the guard below for the one line to relax.

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

# ── System packages ───────────────────────────────────────────────────────────
# The slim -runtime base image ships neither the shared libraries that
# opencv-python (a dependency of orena-focus) needs at import time, nor any
# compiler toolchain. Install system packages HERE, while the build still runs
# as root — everything below runs as the unprivileged 'user'. If one of your
# pip dependencies compiles from source (e.g. flash-attn), add build-essential.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

# If one of your dependencies compiles CUDA extensions during the build (e.g.
# flash-attn from source), uncomment this so it emits kernels for both evaluation
# GPUs rather than only for the card in your build machine:
# ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"

# --break-system-packages: this base image's Python is Ubuntu's system Python,
# which is marked externally-managed (PEP 668), so pip refuses to install without
# it. Nothing is actually broken — `--user` still installs into the unprivileged
# user's own ~/.local, leaving the system packages untouched.
RUN python -m pip install \
    --user \
    --break-system-packages \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# ── Guard: is torch still the recommended CUDA 12.8 build? ────────────────────
# This enforces the recommendation above so an ACCIDENTAL swap cannot reach the
# platform unnoticed. If one of your dependencies needs a torch version this image
# does not have, pip quietly replaces the CUDA 12.8 build with the default PyPI
# wheel (built against CUDA 13.0). That wheel needs a host driver >= 580.65.06; the
# L40S fleet runs 570.211.01, so it cannot start there. Failing the build now beats
# failing every job after you submit.
#
# Both halves matter. The CUDA version is what the driver floor depends on, and it
# is NOT visible in the architecture list: the CUDA 13.0 wheel advertises the very
# same sm_ list as the 12.8 one. The arch list is still checked because it is what
# an older CUDA 12.4/12.6 wheel gets wrong. get_arch_list() returns [] with no GPU
# visible — a docker build never has one — so the flags compiled into the wheel are
# read directly.
#
# Targeting the RTX PRO 6000 only, and want a newer CUDA? Then this default no
# longer applies to you: drop the `cuda.startswith('12.8')` assert below (keep the
# sm_120 arch assert, so a build with no Blackwell kernels is still caught).
RUN python -c "import torch; \
flags = torch._C._cuda_getArchFlags() or ''; \
cuda = torch.version.cuda or ''; \
print('torch', torch.__version__, '| cuda', cuda, '|', flags); \
assert cuda.startswith('12.8'), 'torch was replaced by a CUDA ' + cuda + ' build; the L40S driver cannot run it'; \
assert 'sm_120' in flags and 'sm_86' in flags, 'torch build misses a required GPU architecture: ' + repr(flags)"

# ── Model definition + weights ────────────────────────────────────────────────
# resources/ is populated by prepare_assets.sh before this build. It contains
# the complete local Qwen3-VL snapshot and the frozen question-router assets.
# No model hub access is possible or required when the image starts.
#
# ECR layer-size limit: a single image layer must not exceed 50 GB, and each
# COPY instruction produces exactly one layer. A checkpoint large enough to push
# one COPY past that limit must be split into chunks and copied with several COPY
# instructions (one layer each), then reassembled at runtime — e.g. pre-split
# with `split -b 45G weights.pt resources/weights.part-` and:
#     COPY --chown=user:user resources/weights.part-aa /opt/app/resources/
#     COPY --chown=user:user resources/weights.part-ab /opt/app/resources/
#     COPY --chown=user:user resources/weights.part-ac /opt/app/resources/
# reassembling with `cat resources/weights.part-* > weights.pt` before load.
COPY --chown=user:user resources/ /opt/app/resources/

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
