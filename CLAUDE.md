# lora-tester

Rend une grille force x prompt pour un LoRA sur un serveur ComfyUI, puis compare les résultats dans une galerie HTML d'un seul fichier. Teste des LoRA existants, n'en entraîne pas.

## Statut
- Actif, version initiale. Dernier commit de fond : 2026-10-27 (`bab72ee`, "Initial release"). Dépôt public prévu (licence MIT) [inféré].

## Stack
- Python `>=3.10`, `lora_test.py` en bibliothèque standard seule (`dependencies = []`). Option `doctor` : playwright `>=1.40`. Dev : pytest `>=8`. Géré avec `uv` (`package = false`).
- Workflow intégré : Z-Image Turbo (`z_image_turbo_bf16.safetensors`, `qwen_3_4b.safetensors`, `ae.safetensors`).

## Commandes
Source : README.md.
- `python lora_test.py --list-loras` : liste les LoRA vus par le serveur
- `python lora_test.py --lora "my-lora.safetensors" --strengths "0,0.25,0.5,0.75,1.0" --name "my lora"` : rend 5 prompts x 5 forces
- `COMFY_URL=http://192.168.1.20:8188 python lora_test.py --lora my-lora.safetensors --out ~/lora-runs` : autre serveur, autre dossier
- `python comfy_workflow_doctor.py workflow.json --api-only --comfy http://127.0.0.1:8188/` : diagnostic de workflow
- `uv run pytest` : 13 tests, sans serveur ComfyUI
- Galerie : ouvrir `gallery.html` dans un navigateur et choisir le dossier sous `projects/`.

## Structure
- `lora_test.py` : met un job par (prompt, force) en file, télécharge les images, écrit `manifest.json`. `PROMPTS`, `STEPS`, `CFG`, sampler et `BASE_SEED` sont des constantes en tête de fichier.
- `gallery.html` : visionneuse autonome (grille, bandes, côte à côte, curseur A/B, export PNG).
- `comfy_workflow_doctor.py` : détecte et corrige des défauts connus (types de nodes manquants, ControlNet Flux sans VAE).
- `runpod/` : `bootstrap.sh` et `README.md` pour monter ComfyUI sur un pod RunPod.
- `tests/test_lora_test.py`. `projects/` : sorties, ignorées par git. Chaque run : `manifest.json` plus `images/p00_s000.png` (prompt 0, force 0.00).

## Règles du projet
- Seed fixée par prompt, seule la force varie : toute différence le long d'une ligne vient du LoRA.
- Un autre modèle exige sa propre `build_workflow()`.
- Le script RunPod lit `HF_TOKEN` dans l'environnement uniquement, jamais en argument ni en fichier.
- Limites connues : tous les jobs partent d'un coup, pas de reprise après interruption.

## Liens
- Dépôt : https://github.com/aion88/lora-tester
- Pas de KB `~/knowledge/lora-tester/`.
