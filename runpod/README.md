# RunPod ComfyUI bootstrap

Helper for running the same ComfyUI workflows on a RunPod pod that you run locally.

## What it does

`bootstrap.sh` spins up ComfyUI v0.19.4 on a fresh RunPod pod with:

- Same tag as your local install
- Same custom_nodes list (Manager, GGUF, KJNodes, rgthree, LTXVideo, IPAdapter-plus, NanoBanano, PuLID)
- Production-tier models: `flux2_dev_fp8mixed.safetensors` (33 GB), `mistral_3_small_flux2_fp8.safetensors` (16.8 GB), `flux2-vae.safetensors`

## Quick start

Recommended pod image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` with ≥40 GB VRAM (A100, H100, or 4090 will work for Flux.2-dev fp8).

```bash
# From your machine, copy the script to the pod:
scp runpod/bootstrap.sh <pod>:/workspace/

# On the pod:
export HF_TOKEN="<your token>"   # read from the environment only; needed for the gated Comfy-Org/flux2-dev mirror
bash /workspace/bootstrap.sh

# Start ComfyUI:
cd /workspace/ComfyUI
source venv/bin/activate
python main.py --listen 0.0.0.0 --port 8188
```

Expose port 8188 via RunPod's HTTP/TCP proxy -> pod URL becomes `https://<pod-id>-8188.proxy.runpod.net`.

## Local -> RunPod workflow switch

Keep two variants of each workflow:

| Variant suffix | Model file                                        | Use                  |
| -------------- | ------------------------------------------------- | -------------------- |
| `.local.json`  | `flux2-dev-Q3_K_S.gguf` via `UnetLoaderGGUF`      | Authoring on a laptop  |
| `.runpod.json` | `flux2_dev_fp8mixed.safetensors` via `UNETLoader` | Production on RunPod |

To point the scripts at the pod:

```bash
export COMFY_URL="https://<pod-id>-8188.proxy.runpod.net"
python lora_test.py --list-loras   # lora_test.py reads COMFY_URL
```

## Files

- `bootstrap.sh` - pod bootstrap script (idempotent)
- `README.md` - this file

## Environment variables

| Var           | Default      | Purpose                                    |
| ------------- | ------------ | ------------------------------------------ |
| `COMFY_TAG`   | `v0.19.4`    | ComfyUI version - MUST match local         |
| `WORKSPACE`   | `/workspace` | Install root                               |
| `HF_TOKEN`    | (empty)      | HuggingFace token for gated models         |
| `SKIP_MODELS` | `0`          | Set to `1` to skip weights (test pod only) |

## Parity checklist

Before launching a long run on RunPod, verify:

1. `curl https://<pod>-8188.proxy.runpod.net/system_stats` -> ComfyUI 0.19.4, CUDA device
2. `curl .../object_info/UnetLoaderGGUF | jq .UnetLoaderGGUF.input.required.unet_name[0]` -> includes `flux2_dev_fp8mixed.safetensors` (or your GGUF equivalents)
3. `curl .../object_info/GeminiNanoBanana2` -> returns valid (means API key env var picked up)
4. Submit a tiny test workflow, check it queues and completes
