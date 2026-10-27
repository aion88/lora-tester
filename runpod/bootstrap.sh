#!/usr/bin/env bash
# RunPod ComfyUI bootstrap - same ComfyUI tag as your local install
# Tested on: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 (and compatible)
# Usage: curl -sL https://.../bootstrap.sh | bash   OR   scp this file to pod and `bash bootstrap.sh`
#
# Environment knobs (export before running if needed):
#   COMFY_TAG   - ComfyUI git tag to check out (default: v0.19.4, match local)
#   WORKSPACE   - install root (default: /workspace)
#   HF_TOKEN    - HuggingFace token for gated models (required for BFL Flux.2-dev)
#   SKIP_MODELS - set to 1 to skip weights download (custom_nodes + venv only)

set -euo pipefail

COMFY_TAG="${COMFY_TAG:-v0.19.4}"
WORKSPACE="${WORKSPACE:-/workspace}"
COMFY_DIR="$WORKSPACE/ComfyUI"
MODELS_DIR="$WORKSPACE/models"
HF_TOKEN="${HF_TOKEN:-}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Phase 1: Clone ComfyUI $COMFY_TAG ==="
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
if [ ! -d "$COMFY_DIR" ]; then
  git clone https://github.com/comfyanonymous/ComfyUI.git
fi
cd "$COMFY_DIR"
git fetch --tags
git checkout "$COMFY_TAG"

log "=== Phase 2: venv + PyTorch CUDA ==="
if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip

# CUDA PyTorch (pin to stable for pod reproducibility; adjust cu124 → cu128 per pod image)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

log "=== Phase 3: custom_nodes (same set as local) ==="
mkdir -p custom_nodes
cd custom_nodes

for repo in \
  "https://github.com/ltdrdata/ComfyUI-Manager.git" \
  "https://github.com/city96/ComfyUI-GGUF.git" \
  "https://github.com/kijai/ComfyUI-KJNodes.git" \
  "https://github.com/rgthree/rgthree-comfy.git" \
  "https://github.com/Lightricks/ComfyUI-LTXVideo.git" \
  "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git" \
  "https://github.com/ShmuelRonen/ComfyUI-NanoBanano.git" \
  "https://github.com/cubiq/PuLID_ComfyUI.git"
do
  name=$(basename "$repo" .git)
  if [ ! -d "$name" ]; then
    git clone --depth 1 "$repo"
    [ -f "$name/requirements.txt" ] && pip install -r "$name/requirements.txt" || true
  fi
done

cd "$COMFY_DIR"

if [ "${SKIP_MODELS:-0}" = "1" ]; then
  log "=== Phase 4: SKIPPED (SKIP_MODELS=1) ==="
  log "Done. Start: cd $COMFY_DIR && source venv/bin/activate && python main.py --listen 0.0.0.0 --port 8188"
  exit 0
fi

log "=== Phase 4: Download production-tier models ==="
# HF CLI for gated + fast parallel downloads
pip install -U "huggingface_hub[cli]"
# huggingface-cli reads HF_TOKEN from the environment. Never pass it as an
# argument (it would show in `ps`) and never store it in a git credential helper.
if [ -n "$HF_TOKEN" ]; then
  export HF_TOKEN
fi

mkdir -p "$MODELS_DIR"/{diffusion_models,text_encoders,vae,loras,checkpoints,controlnet,upscale_models,clip_vision}

# Flux.2-dev fp8 - PRODUCTION tier (33 GB diffusion + 16.8 GB text encoder)
log "Downloading Flux.2-dev fp8 diffusion model (33 GB)..."
huggingface-cli download Comfy-Org/flux2-dev \
  split_files/diffusion_models/flux2_dev_fp8mixed.safetensors \
  --local-dir "$MODELS_DIR/.hf-dl" --local-dir-use-symlinks False
mv "$MODELS_DIR/.hf-dl/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors" "$MODELS_DIR/diffusion_models/"

log "Downloading Mistral text encoder fp8 (16.8 GB)..."
huggingface-cli download Comfy-Org/flux2-dev \
  split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors \
  --local-dir "$MODELS_DIR/.hf-dl" --local-dir-use-symlinks False
mv "$MODELS_DIR/.hf-dl/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors" "$MODELS_DIR/text_encoders/"

log "Downloading Flux.2 VAE (0.3 GB)..."
huggingface-cli download Comfy-Org/flux2-dev \
  split_files/vae/flux2-vae.safetensors \
  --local-dir "$MODELS_DIR/.hf-dl" --local-dir-use-symlinks False
mv "$MODELS_DIR/.hf-dl/split_files/vae/flux2-vae.safetensors" "$MODELS_DIR/vae/"

# OPTIONAL additional models - uncomment what you need
# Z-Image Turbo (fast alternative, works on Mac too; mostly useful for parity testing)
# huggingface-cli download Comfy-Org/z_image_turbo split_files/diffusion_models/z_image_turbo_bf16.safetensors --local-dir "$MODELS_DIR/.hf-dl" --local-dir-use-symlinks False
# huggingface-cli download Comfy-Org/z_image_turbo split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors --local-dir "$MODELS_DIR/.hf-dl" --local-dir-use-symlinks False

# LTX-2.3 video (if doing video)
# huggingface-cli download Kijai/LTX2.3_comfy --local-dir "$MODELS_DIR/.hf-dl/ltx23" --local-dir-use-symlinks False

rm -rf "$MODELS_DIR/.hf-dl"

log "=== Phase 5: extra_model_paths.yaml ==="
cp "$COMFY_DIR/extra_model_paths.yaml.example" "$COMFY_DIR/extra_model_paths.yaml" 2>/dev/null || true
cat > "$COMFY_DIR/extra_model_paths.yaml" <<YAML
runpod-shared:
  base_path: $MODELS_DIR/
  is_default: true
  checkpoints: checkpoints/
  diffusion_models: diffusion_models/
  text_encoders: text_encoders/
  vae: vae/
  loras: loras/
  controlnet: controlnet/
  upscale_models: upscale_models/
  clip_vision: clip_vision/
YAML

log "=== Done ==="
log "Start: cd $COMFY_DIR && source venv/bin/activate && python main.py --listen 0.0.0.0 --port 8188"
log "Expose 8188 via RunPod TCP/HTTP proxy; set COMFY_URL=https://<pod-id>-8188.proxy.runpod.net locally"
