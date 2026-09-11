#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)
RESOURCES_DIR="${SCRIPT_DIR}/resources"
MODEL_DEST="${RESOURCES_DIR}/qwen3-vl-8b"
Q2_ENCODER_DEST="${RESOURCES_DIR}/q2_siglip_base_patch16_384"
Q2_ROUTER_DEST="${RESOURCES_DIR}/q2_siglip_text_r5.joblib"
SPECIALISTS_DEST="${RESOURCES_DIR}/specialists"
GENERAL_SPECIALIST_DEST="${SPECIALISTS_DEST}/general_final_union_adapted_weights.safetensors"
AGG_SPECIALIST_DEST="${SPECIALISTS_DEST}/agg_final_union_adapted_weights.safetensors"

MODEL_RELATIVE_PATH="huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
Q2_ENCODER_RELATIVE_PATH="models/siglip-base-patch16-384"
Q2_ROUTER_RELATIVE_PATH="outputs/SEG010_cross_timescale_21way_v1/question_router_v1/models/q2_siglip_text_r5.joblib"
GENERAL_SPECIALIST_RELATIVE_PATH="outputs/SEG010_FINAL_v4/checkpoints/GENERAL_FINAL/union_adapted_weights.safetensors"
AGG_SPECIALIST_RELATIVE_PATH="outputs/SEG010_AGG_FINAL_v4/checkpoints/AGG_FINAL/union_adapted_weights.safetensors"

# The DevContainer mounts host-side ``orena/cache`` as /cache and the
# repository as /workspace. When this script runs directly on the host,
# neither container path exists; the sibling cache and repository-relative
# checkpoint do. Explicit environment variables remain the highest priority.
if [[ -n "${QWEN_MODEL_PATH:-}" ]]; then
  MODEL_SOURCE="$QWEN_MODEL_PATH"
elif [[ -d "/cache/${MODEL_RELATIVE_PATH}" ]]; then
  MODEL_SOURCE="/cache/${MODEL_RELATIVE_PATH}"
else
  MODEL_SOURCE="${REPO_ROOT}/../cache/${MODEL_RELATIVE_PATH}"
fi
if [[ -n "${Q2_ENCODER_PATH:-}" ]]; then
  Q2_ENCODER_SOURCE="$Q2_ENCODER_PATH"
elif [[ -d "/cache/${Q2_ENCODER_RELATIVE_PATH}" ]]; then
  Q2_ENCODER_SOURCE="/cache/${Q2_ENCODER_RELATIVE_PATH}"
else
  Q2_ENCODER_SOURCE="${REPO_ROOT}/../cache/${Q2_ENCODER_RELATIVE_PATH}"
fi
if [[ -n "${Q2_ROUTER_PATH:-}" ]]; then
  Q2_ROUTER_SOURCE="$Q2_ROUTER_PATH"
else
  Q2_ROUTER_SOURCE="${REPO_ROOT}/${Q2_ROUTER_RELATIVE_PATH}"
fi
if [[ -n "${GENERAL_FINAL_STATE_PATH:-}" ]]; then
  GENERAL_SPECIALIST_SOURCE="$GENERAL_FINAL_STATE_PATH"
else
  GENERAL_SPECIALIST_SOURCE="${REPO_ROOT}/${GENERAL_SPECIALIST_RELATIVE_PATH}"
fi
if [[ -n "${AGG_FINAL_STATE_PATH:-}" ]]; then
  AGG_SPECIALIST_SOURCE="$AGG_FINAL_STATE_PATH"
else
  AGG_SPECIALIST_SOURCE="${REPO_ROOT}/${AGG_SPECIALIST_RELATIVE_PATH}"
fi
EXPECTED_Q2_ROUTER_SHA256="a335c568991db95b7633dd9fb9c23eef2ee276ae51485592781056c1917c9158"

required_model_files=(
  config.json generation_config.json preprocessor_config.json
  video_preprocessor_config.json tokenizer_config.json tokenizer.json
  merges.txt vocab.json chat_template.json model.safetensors.index.json
  model-00001-of-00004.safetensors model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors model-00004-of-00004.safetensors
)

if [[ ! -d "$MODEL_SOURCE" ]]; then
  echo "ERROR: Qwen model snapshot not found: $MODEL_SOURCE" >&2
  echo "Set QWEN_MODEL_PATH to the local snapshot directory." >&2
  exit 1
fi
if [[ ! -d "$Q2_ENCODER_SOURCE" ]]; then
  echo "ERROR: Q2 SigLIP encoder not found: $Q2_ENCODER_SOURCE" >&2
  exit 1
fi
if [[ ! -f "$Q2_ROUTER_SOURCE" ]]; then
  echo "ERROR: Q2 classifier not found: $Q2_ROUTER_SOURCE" >&2
  exit 1
fi
if [[ ! -f "$GENERAL_SPECIALIST_SOURCE" ]]; then
  echo "ERROR: final General specialist state not found: $GENERAL_SPECIALIST_SOURCE" >&2
  echo "Set GENERAL_FINAL_STATE_PATH to the validated union_adapted_weights.safetensors file." >&2
  exit 1
fi
if [[ ! -f "$AGG_SPECIALIST_SOURCE" ]]; then
  echo "ERROR: final Aggregation specialist state not found: $AGG_SPECIALIST_SOURCE" >&2
  echo "Set AGG_FINAL_STATE_PATH to the validated union_adapted_weights.safetensors file." >&2
  exit 1
fi
for filename in "${required_model_files[@]}"; do
  if [[ ! -f "$MODEL_SOURCE/$filename" ]]; then
    echo "ERROR: missing model asset: $MODEL_SOURCE/$filename" >&2
    exit 1
  fi
done

actual_q2_router_sha256=$(sha256sum "$Q2_ROUTER_SOURCE" | awk '{print $1}')
if [[ "$actual_q2_router_sha256" != "$EXPECTED_Q2_ROUTER_SHA256" ]]; then
  echo "ERROR: Q2 classifier SHA256 mismatch: $actual_q2_router_sha256" >&2
  exit 1
fi

mkdir -p "$MODEL_DEST"
copy_required=true
if [[ -f "$RESOURCES_DIR/.asset_stamp" ]]; then
  copy_required=false
  for filename in "${required_model_files[@]}"; do
    if [[ ! -f "$MODEL_DEST/$filename" || "$(stat -Lc '%s' "$MODEL_SOURCE/$filename")" != "$(stat -c '%s' "$MODEL_DEST/$filename")" ]]; then
      copy_required=true
      break
    fi
  done
fi

if [[ "$copy_required" == true ]]; then
  echo "=+= Copying local Qwen3-VL snapshot into Docker build context"
  for filename in "${required_model_files[@]}"; do
    cp -L "$MODEL_SOURCE/$filename" "$MODEL_DEST/$filename"
  done
fi
q2_required_files=(
  .gitattributes README.md config.json model.safetensors preprocessor_config.json
  special_tokens_map.json spiece.model tokenizer.json tokenizer_config.json
)
mkdir -p "$Q2_ENCODER_DEST"
for filename in "${q2_required_files[@]}"; do
  if [[ ! -f "$Q2_ENCODER_SOURCE/$filename" ]]; then
    echo "ERROR: missing Q2 encoder asset: $Q2_ENCODER_SOURCE/$filename" >&2
    exit 1
  fi
  if [[ ! -f "$Q2_ENCODER_DEST/$filename" || "$(stat -Lc '%s' "$Q2_ENCODER_SOURCE/$filename")" != "$(stat -Lc '%s' "$Q2_ENCODER_DEST/$filename")" ]]; then
    echo "=+= Copying Q2 SigLIP asset $filename"
    cp -L "$Q2_ENCODER_SOURCE/$filename" "$Q2_ENCODER_DEST/$filename"
  fi
done
if [[ ! -f "$Q2_ROUTER_DEST" || "$(sha256sum "$Q2_ROUTER_DEST" | awk '{print $1}')" != "$EXPECTED_Q2_ROUTER_SHA256" ]]; then
  echo "=+= Copying Q2 SigLIP classifier"
  cp -L "$Q2_ROUTER_SOURCE" "$Q2_ROUTER_DEST"
fi

mkdir -p "$SPECIALISTS_DEST"
for pair in \
  "$GENERAL_SPECIALIST_SOURCE|$GENERAL_SPECIALIST_DEST" \
  "$AGG_SPECIALIST_SOURCE|$AGG_SPECIALIST_DEST"; do
  source_path="${pair%%|*}"
  destination_path="${pair#*|}"
  if [[ ! -f "$destination_path" ]] || ! cmp -s "$source_path" "$destination_path"; then
    echo "=+= Copying final specialist state $(basename "$destination_path")"
    cp -L "$source_path" "$destination_path"
  fi
  if ! cmp -s "$source_path" "$destination_path"; then
    echo "ERROR: final specialist state copy mismatch: $destination_path" >&2
    exit 1
  fi
done

stamp="model_revision=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b model_source=$MODEL_SOURCE"
if [[ ! -f "$RESOURCES_DIR/.asset_stamp" ]] || [[ "$(<"$RESOURCES_DIR/.asset_stamp")" != "$stamp" ]]; then
  printf '%s\n' "$stamp" > "$RESOURCES_DIR/.asset_stamp"
fi

echo "=+= Assets ready"
echo "    model: $MODEL_DEST ($(du -sh "$MODEL_DEST" | awk '{print $1}'))"
echo "    Q2 encoder: $Q2_ENCODER_DEST ($(du -sh "$Q2_ENCODER_DEST" | awk '{print $1}'))"
echo "    Q2 classifier SHA256: $(sha256sum "$Q2_ROUTER_DEST" | awk '{print $1}')"
echo "    final General specialist SHA256: $(sha256sum "$GENERAL_SPECIALIST_DEST" | awk '{print $1}')"
echo "    final Aggregation specialist SHA256: $(sha256sum "$AGG_SPECIALIST_DEST" | awk '{print $1}')"
