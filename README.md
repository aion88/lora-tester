# lora-tester

Render a strength x prompt grid for a LoRA on a ComfyUI server, then compare the results in a single-file browser gallery.

This tool **tests** LoRAs you already have. It does not train them.

## Why it exists

Picking a LoRA strength by eye, one image at a time, hides what the LoRA does. Fixing the seed per prompt and sweeping only the strength gives a clean comparison: any difference along a row is the LoRA and nothing else. Strength 0 renders the base model with no LoRA loaded, so every row has its own baseline.

## What's in the box

| File | What it does |
|------|--------------|
| `lora_test.py` | Queues one job per (prompt, strength) on a ComfyUI server, downloads the images and writes a `manifest.json` |
| `gallery.html` | Single-file viewer: grid, strips, side by side, A/B slider, PNG export. No server, no dependencies |
| `comfy_workflow_doctor.py` | Loads a workflow, reports and fixes known problems (missing node types, Flux ControlNet without a VAE). Optional, needs Playwright for the frontend check; `--api-only` does not |
| `runpod/` | A bootstrap script for a RunPod pod with matching ComfyUI tag and custom nodes |

## Install

Python 3.10 or newer. `lora_test.py` uses only the standard library.

```
git clone https://github.com/aion88/lora-tester
cd lora-tester
```

The workflow built by `lora_test.py` is Z-Image Turbo, so ComfyUI needs `z_image_turbo_bf16.safetensors`, `qwen_3_4b.safetensors` and `ae.safetensors`. Edit `build_workflow()` for another model.

## Usage

```
# list LoRAs the server can see
python lora_test.py --list-loras

# render 5 prompts x 5 strengths
python lora_test.py --lora "my-lora.safetensors" --strengths "0,0.25,0.5,0.75,1.0" --name "my lora"

# another server, another output folder
COMFY_URL=http://192.168.1.20:8188 python lora_test.py --lora my-lora.safetensors --out ~/lora-runs
```

Edit `PROMPTS`, `STEPS`, `CFG`, the sampler and `BASE_SEED` at the top of `lora_test.py`. Open `gallery.html` in a browser and pick the project folder it created under `projects/`.

Each run is a self-contained folder: `manifest.json` plus `images/p00_s000.png` (prompt 0, strength 0.00), `p00_s025.png`, and so on.

Workflow doctor:

```
python comfy_workflow_doctor.py workflow.json --api-only --comfy http://127.0.0.1:8188/
```

The RunPod script reads `HF_TOKEN` from the environment only, never from an argument or a file.

## Tests

```
uv run pytest
```

13 tests cover argument parsing, grid generation, seed handling, workflow construction and the doctor's pure helpers. They need no ComfyUI server.

## Limitations

- The built-in workflow targets one model family (Z-Image Turbo). Other models need their own `build_workflow()`.
- Jobs are queued all at once and downloaded as they finish; there is no resume if the script is interrupted.
- The sampler settings and prompt list are constants in the script, not config files.
- `comfy_workflow_doctor.py` was written against one ComfyUI version; its fix list is short and specific.

MIT license.
