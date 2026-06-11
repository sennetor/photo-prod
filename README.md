# Photo Prod

A from-scratch, browser-based photo editor with a modern Photoshop-style dark UI, backed by a local GPU image-generation service. Everything runs on your own machine — the editor is plain HTML/CSS/JS (no build step), and the AI features are served by a small FastAPI app wrapping open diffusion and segmentation models.

![Photo Prod](photoprod.ico)

## Features

**Editor (`editor/`)** — vanilla JS, no dependencies:
- Layers with blend modes, opacity, per-layer edge feather, thumbnails, reorder/rename
- Tools: move, rectangular/elliptical marquee, lasso, **AI object select**, brush, eraser, paint-bucket (mask-aware flood fill), eyedropper, zoom
- Mask-based selections with boolean add/subtract, **invert**, **expand** (outward-only) and **feather** (outward-only) — contour-traced marching ants
- Adjustments (brightness/contrast/saturation/hue/blur) with live preview + bake; one-click filters
- Copy / cut / paste / "layer via copy", crop-to-selection, expand-canvas (outpainting), PNG export
- Undo/redo, collapsible + resizable panels
- **Recent files** (File → Open Recent) and **automatic session restore** — an accidental refresh or crash doesn't lose your work (layers snapshot to IndexedDB)
- **Model switcher in the UI** — the Generative AI panel shows the active model (with its license) and switches backends at runtime, with a live progress bar (load / denoise steps / blend) during generation

**AI service (`genai-service/server.py`)** — FastAPI, picks the largest CUDA device automatically:
- **Generative fill** — full-image latent inpainting (no pixel paste-back seam), restoring untouched pixels at full resolution, then **Poisson seamless-clone** stitching (OpenCV `cv2.seamlessClone`, the gradient-domain technique behind Photoshop's Healing Brush) with a bled-out mask so seams vanish. Returns a batch of variations to cycle through.
- **Select Subject / Object Select** — text-prompted segmentation via SAM 3, click-to-mask via SAM 2.1
- **Reimagine** — whole-image img2img variations from a prompt + likeness slider
- Switchable inpaint backends via `PHOTOPROD_MODEL`

## Requirements

- An NVIDIA GPU (the inpainting models are large; ~24 GB VRAM for FLUX.1 Fill, less for the alternatives)
- Python 3.12, Node.js (for the static file server), a recent browser

## Setup

```bash
# 1. Python environment for the AI service
python -m venv sam-env
sam-env\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
sam-env\Scripts\pip install -r requirements.txt

# 2. Hugging Face access for gated models (FLUX, SAM) — one-time
sam-env\Scripts\hf auth login

# 3. Run the AI service (downloads model weights on first use)
sam-env\Scripts\python genai-service\server.py    # http://127.0.0.1:8765

# 4. Serve the editor (any static server works)
npx http-server editor -p 4173 -c-1               # http://localhost:4173
```

On Windows, `launch.cmd` starts both and opens the editor.

## Model backends

`PHOTOPROD_MODEL` sets the startup model; you can also switch at runtime from the editor (click the model badge in the Generative AI panel). The default is **`klein`** (Apache-2.0) so a fresh checkout is commercial-safe out of the box.

| value       | model                       | license            | notes |
|-------------|-----------------------------|--------------------|-------|
| `klein` *(default)* | black-forest-labs/FLUX.2-klein-4B | Apache 2.0, ungated | |
| `flux-fill` | black-forest-labs/FLUX.1-Fill-dev | gated, **NON-COMMERCIAL** | purpose-built fill |
| `chroma`    | lodestones/Chroma1-HD       | Apache 2.0, ungated | uncensored |
| `qwen`      | Qwen/Qwen-Image             | Apache 2.0, ungated | 20B — large |
| `juggernaut` | RunDiffusion/Juggernaut-XI-v11 | **CC-BY-NC-ND (non-commercial)** | SDXL |
| `juggernaut-cn` | Juggernaut XI + xinsir ControlNet-Union ProMax | **non-commercial** (base model) | SDXL + real inpaint conditioning — strong seams |
| `z-image`   | Tongyi-MAI/Z-Image-Turbo    | Apache 2.0, ungated | txt2img only: generates from the prompt and seam-stitches into the selection (no scene awareness) |

> ⚠️ **`flux-fill` and the two `juggernaut*` backends are non-commercial.** Opt in only if you've accepted the licenses (flux-fill is also gated on Hugging Face) and your use is personal/non-commercial.

Reimagine follows the selected backend when it has an img2img variant (chroma, qwen, juggernaut, juggernaut-cn, z-image) and falls back to Chroma otherwise. Select Subject uses `facebook/sam3` (gated, review its license) and `facebook/sam2.1-hiera-large` (Apache 2.0).

Weights download on first use into the Hugging Face cache (the UI marks not-yet-downloaded models with ⬇ — some are tens of GB). Heavy components stream from disk straight to GPU memory, so even 20B models load on machines with modest system RAM. To pre-download and verify every backend end-to-end, run the staging tool while the service is up:

```bash
sam-env\Scripts\python genai-service\stage.py    # exit 0 = every backend verified
```

`PHOTOPROD_STITCH` = `poisson` (default) or `feather`. `PHOTOPROD_HOST` defaults to `0.0.0.0` (LAN-accessible).

## Licenses

The **code** in this repository is released under the MIT License (see `LICENSE`). It is original work; AI techniques it uses (latent inpainting, Poisson seamless cloning via OpenCV) are standard published methods, not copied from any other project.

**The AI models the software downloads at runtime are governed by their own licenses, not MIT.** No model weights are included in this repo — you download them yourself and accept each model's terms at that point. In particular:

- **FLUX.1 Fill [dev]** — gated, **non-commercial** (opt-in only)
- **SAM 3 (`facebook/sam3`)** — gated; review Meta's license before use
- FLUX.2 Klein 4B, Chroma, Qwen-Image, SAM 2.1 — Apache 2.0

You are responsible for complying with each model's terms. Review them before any commercial use.

## Trademark

Not affiliated with, endorsed by, or sponsored by Adobe. "Photoshop" is a trademark of Adobe Inc., used here only descriptively to indicate a comparable category of tool. Photo Prod's name, lens logo, and UI are its own.

## Status

Personal project / work in progress. Built incrementally; expect rough edges.
