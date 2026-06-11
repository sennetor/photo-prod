"""Photo Prod generative fill service.

FastAPI inpainting/outpainting service. Receives an image + mask + prompt,
fills the masked region on the GPU (prefers the A100 when more than one CUDA
device is present), and returns the full-size result.

Backend is selected with the PHOTOPROD_MODEL env var (see MODELS below), and can
be switched at runtime from the editor UI (POST /model):
  klein      FLUX.2 Klein 4B     Apache 2.0, ungated, inpaint-conditioned (default)
  flux-fill  FLUX.1 Fill dev     purpose-built inpaint, gated + NON-COMMERCIAL, opt-in
  chroma     Chroma1-HD          Apache 2.0, ungated, uncensored, general img2img
  qwen       Qwen-Image 20B      Apache 2.0, ungated
  juggernaut Juggernaut XI v11   SDXL inpaint (Fooocus's model), CC-BY-NC-ND — personal use
  juggernaut-cn Juggernaut + ControlNet-Union ProMax inpaint (mode 7) — best seams, personal use
  z-image    Z-Image-Turbo       Apache 2.0, ungated, TEXT-TO-IMAGE ONLY (no mask
                                 conditioning) — generates then seam-stitches;
                                 needs diffusers >= 0.38. Context-aware fill
                                 awaits Z-Image-Omni/-Edit (not yet released).

Inpainting is full-image and latent-composited (Fooocus-style, no pixel paste):
the whole image is encoded, only the masked region is denoised, and the model
DECODES the entire image together so the mask boundary is harmonized by the VAE
— there is no seam to blend. The untouched areas of the original photo are then
restored at full resolution through a morphological soft mask, so only the
generated region changes and the rest stays pixel-exact.

Run:  sam-env\\Scripts\\python.exe genai-service\\server.py
"""

import base64
import gc
import inspect
import io
import os
import threading

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter
from pydantic import BaseModel

MODELS = {
    # "img2img_pipeline" is the diffusers class /reimagine uses when this backend
    # is selected; backends without one (klein, flux-fill: fill-trained
    # transformers with no img2img variant) fall back to Chroma for reimagine.
    "chroma": {
        "repo": "lodestones/Chroma1-HD",
        "pipeline": "ChromaInpaintPipeline",
        "guidance": 4.0,
        "steps": 30,
        "img2img": True,    # masked-img2img style: takes strength
        "img2img_pipeline": "ChromaImg2ImgPipeline",
    },
    "klein": {
        "repo": "black-forest-labs/FLUX.2-klein-4B",
        "pipeline": "Flux2KleinInpaintPipeline",
        "guidance": 4.0,
        "steps": 28,
        "img2img": True,
    },
    "qwen": {
        "repo": "Qwen/Qwen-Image",
        "pipeline": "QwenImageInpaintPipeline",
        "guidance": 4.0,
        "steps": 30,
        "img2img": True,
        "true_cfg": True,   # Qwen uses true_cfg_scale instead of guidance_scale
        "img2img_pipeline": "QwenImageImg2ImgPipeline",
    },
    "flux-fill": {
        "repo": "black-forest-labs/FLUX.1-Fill-dev",
        "pipeline": "FluxFillPipeline",
        "guidance": 30.0,
        "steps": 28,
        "img2img": False,   # dedicated fill model: no strength param
    },
    # Juggernaut XI v11 — RunDiffusion's SDXL fine-tune, the model behind Fooocus's
    # blending. A normal SDXL checkpoint loaded into StableDiffusionXLInpaintPipeline
    # does proper masked latent inpainting (4-ch UNet path), so it's a real, mature,
    # verified inpaint backend — exactly the img2img+strength flow below. SDXL wants
    # float16 (the VAE is force-upcast internally to dodge the fp16 NaN bug). License
    # is CC-BY-NC-ND: personal/local use only, NOT commercial — keep klein as the
    # public default.
    "juggernaut": {
        "repo": "RunDiffusion/Juggernaut-XI-v11",
        "pipeline": "StableDiffusionXLInpaintPipeline",
        "guidance": 7.0,     # SDXL classic CFG
        "steps": 30,
        "img2img": True,     # SDXL inpaint takes a strength param
        "dtype": "float16",
        "img2img_pipeline": "StableDiffusionXLImg2ImgPipeline",
    },
    # Juggernaut base + ControlNet-Union ProMax in inpaint mode (control_mode 7).
    # This adds the real inpaint *conditioning* the plain SDXL pipeline lacks — the
    # ControlNet sees the masked image and harmonizes the fill with the scene, which
    # is what kills the seam. Keeps Juggernaut's look. The community that wanted
    # Fooocus-grade inpaint moved to ControlNet-Union; this is that path. CC-BY-NC-ND
    # base → personal use only. fp16 throughout with the fp16-fix VAE.
    "juggernaut-cn": {
        "repo": "RunDiffusion/Juggernaut-XI-v11",
        "pipeline": "StableDiffusionXLControlNetUnionInpaintPipeline",
        "controlnet": "brad-twinkl/controlnet-union-sdxl-1.0-promax",
        "vae": "madebyollin/sdxl-vae-fp16-fix",
        "guidance": 7.0,
        "steps": 30,
        "img2img": True,
        "dtype": "float16",
        "cn_union_inpaint": True,
        "cn_scale": 0.9,     # ControlNet conditioning strength
        "img2img_pipeline": "StableDiffusionXLImg2ImgPipeline",
    },
    # Z-Image-Turbo (Alibaba Tongyi, Apache-2.0, ungated). diffusers ships only
    # the text-to-image ZImagePipeline today — there is NO image/mask input, so
    # it can't do context-aware masked fill on its own. We generate from the
    # prompt and seam-stitch into the selection (good for backgrounds / texture /
    # outpainting empty areas). Context-aware fill needs Z-Image-Omni / -Edit,
    # which are still "to be released"; when they land we add the image-conditioned
    # content stage + a Flux/Klein seam-refine pass. Needs diffusers >= 0.38.
    "z-image": {
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "pipeline": "ZImagePipeline",
        "guidance": 0.0,     # Turbo is guidance-distilled — must be 0
        "steps": 9,          # 8 DiT forward passes
        "txt2img": True,     # prompt-only; generate then stitch
        # NOTE: the model card suggests low_cpu_mem_usage=False, but that forces
        # naive full-RAM loading and OOMs a 32 GB machine (the repo is ~60 GB on
        # disk). The default streaming loader works with diffusers >= 0.38.
        "img2img_pipeline": "ZImageImg2ImgPipeline",
    },
}

BACKEND = os.environ.get("PHOTOPROD_MODEL", "klein")
if BACKEND not in MODELS:
    raise SystemExit(f"unknown PHOTOPROD_MODEL {BACKEND!r}; pick one of {list(MODELS)}")
MODEL = MODELS[BACKEND]
MODEL_ID = MODEL["repo"]
PORT = 8765
MAX_GEN_SIDE = 1024          # full image is scaled so its long side hits this
MASK_DILATE = 4             # px the model-side mask grows past the selection edge
MASK_FEATHER = 9            # px of gaussian feather on the composite mask edge
STITCH = os.environ.get("PHOTOPROD_STITCH", "poisson")  # poisson | feather
STITCH_BLEED = 14          # px the composite region bleeds out past the selection
                           # (moves the seam off the subject silhouette into bg)

app = FastAPI(title="Photo Prod GenAI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipe = None
# RLock shared by the fill and reimagine slots: get_img2img may need to drop the
# fill pipe while switch_model clears both — one reentrant lock avoids inversion.
_pipe_lock = threading.RLock()
_device = None

# live progress for the UI: {"phase": "idle" | "loading" | "denoising" | "stitching", ...}
_progress = {"phase": "idle"}


def set_progress(**kw):
    global _progress
    _progress = kw if kw else {"phase": "idle"}


def step_callback(total):
    """diffusers callback_on_step_end → live denoise progress for /progress."""
    def cb(pipeline, step, timestep, callback_kwargs):
        set_progress(phase="denoising", step=step + 1, total=total)
        return callback_kwargs
    return cb


def callback_kwargs_for(pipe, total):
    """Attach the progress callback only if this pipeline supports it."""
    try:
        if "callback_on_step_end" in inspect.signature(pipe.__call__).parameters:
            return {"callback_on_step_end": step_callback(total)}
    except (TypeError, ValueError):
        pass
    return {}


def vram_info():
    """Per-GPU memory: what this process holds (torch) and what's free overall."""
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        out.append({
            "index": i,
            "name": torch.cuda.get_device_properties(i).name,
            "total_gb": round(total_b / 2**30, 1),
            "free_gb": round(free_b / 2**30, 1),
            "ours_gb": round(torch.cuda.memory_allocated(i) / 2**30, 1),
        })
    return out


def repo_cached(repo: str) -> bool:
    """True if the HF hub cache already holds a snapshot of this repo."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        base = HF_HUB_CACHE
    except Exception:
        base = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    snaps = os.path.join(base, "models--" + repo.replace("/", "--"), "snapshots")
    return os.path.isdir(snaps) and bool(os.listdir(snaps))

SAM_MODEL_ID = "facebook/sam3"
_sam = None
_sam_lock = threading.Lock()

SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
_sam2 = None
_sam2_lock = threading.Lock()

# whole-image "reimagine" (img2img) — a creative variation guided by image+text,
# separate from masked inpainting. Follows the selected backend when it has an
# img2img pipeline class; falls back to Chroma (ungated, uncensored) otherwise.
IMG2IMG_MODEL_ID = "lodestones/Chroma1-HD"
_img2img = None          # (key, pipeline) — keyed so backend switches reload it
_img2img_lock = _pipe_lock      # shared reentrant lock (see _pipe_lock note)


def reimagine_plan():
    """Resolve which model /reimagine would run right now."""
    import diffusers

    name = MODEL.get("img2img_pipeline")
    if name and hasattr(diffusers, name):
        return {
            "backend": BACKEND,
            "repo": MODEL_ID,
            "pipeline": name,
            "guidance": MODEL["guidance"],
            "steps": MODEL["steps"],
            "dtype": MODEL.get("dtype"),
            "vae": MODEL.get("vae"),
        }
    return {
        "backend": "chroma",
        "repo": IMG2IMG_MODEL_ID,
        "pipeline": "ChromaImg2ImgPipeline",
        "guidance": 4.0,
        "steps": 30,
        "dtype": None,
        "vae": None,
    }


# pipeline components big enough to matter for load-time RAM
_BIG_COMPONENTS = ("text_encoder", "transformer", "unet")


def load_pipeline(cls, repo, dtype, extra):
    """Load a pipeline with its heavy components streamed straight onto the GPU.

    The classic from_pretrained(...).to(device) materializes the WHOLE pipeline
    in system RAM before copying — Chroma needs ~28 GB of staging RAM and Qwen
    far more, which kills the process on a 32 GB machine (sometimes silently).
    Pipeline-level device_map is unreliable for this (the 'balanced' strategy
    quietly leaves parameters on the meta device when boxed to one GPU), but
    per-MODEL device_map={'': device} is a first-class path in both diffusers
    and transformers: shards stream from disk to VRAM one at a time. So we read
    the pipeline's model_index.json, pre-load each big component that way, and
    hand them to the pipeline loader; the small parts (VAE, scheduler,
    tokenizers) load classically."""
    import importlib

    extra = dict(extra)
    try:
        cfg = cls.load_config(repo)
        for name, spec in cfg.items():
            if name.startswith("_") or name in extra:
                continue
            if not isinstance(spec, (list, tuple)) or len(spec) != 2 or not spec[1]:
                continue
            if not any(name.startswith(big) for big in _BIG_COMPONENTS):
                continue
            lib_name, klass_name = spec
            try:
                klass = getattr(importlib.import_module(lib_name), klass_name)
                extra[name] = klass.from_pretrained(
                    repo, subfolder=name, torch_dtype=dtype, device_map={"": _device}
                )
                gc.collect()
            except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
                # Out of GPU/CPU memory mid-stream. Do NOT fall back to the
                # classic RAM path — on this machine that kills the process.
                extra.clear()
                gc.collect()
                torch.cuda.empty_cache()
                raise HTTPException(
                    507, f"not enough memory to load {repo} ({name}): {exc}"
                ) from exc
            except Exception:
                # API mismatch etc. — let the classic loader try this component
                extra.pop(name, None)
    except HTTPException:
        raise
    except Exception:
        pass
    return cls.from_pretrained(repo, torch_dtype=dtype, **extra).to(_device)


def pick_device() -> str:
    """Prefer the A100 if present, else the largest-VRAM CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    best, best_mem = 0, 0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        if "A100" in props.name:
            return f"cuda:{i}"
        if props.total_memory > best_mem:
            best, best_mem = i, props.total_memory
    return f"cuda:{best}"


def get_pipe():
    global _pipe, _device
    with _pipe_lock:
        if _pipe is None:
            import diffusers

            cls = getattr(diffusers, MODEL["pipeline"])
            _device = pick_device()
            set_progress(phase="loading", model=MODEL_ID, cached=repo_cached(MODEL_ID))
            dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
                MODEL.get("dtype"), torch.bfloat16
            )
            extra = dict(MODEL.get("from_kwargs", {}))
            if MODEL.get("controlnet"):
                cn = diffusers.ControlNetUnionModel.from_pretrained(
                    MODEL["controlnet"], torch_dtype=dtype
                )
                extra["controlnet"] = cn.to(_device)
            if MODEL.get("vae"):
                extra["vae"] = diffusers.AutoencoderKL.from_pretrained(
                    MODEL["vae"], torch_dtype=dtype
                ).to(_device)
            _pipe = load_pipeline(cls, MODEL_ID, dtype, extra)
        return _pipe


def get_sam():
    global _sam, _device
    with _sam_lock:
        if _sam is None:
            from transformers import Sam3Model, Sam3Processor

            if _device is None:
                _device = pick_device()
            model = Sam3Model.from_pretrained(SAM_MODEL_ID).to(_device).eval()
            processor = Sam3Processor.from_pretrained(SAM_MODEL_ID)
            _sam = (model, processor)
        return _sam


def get_sam2():
    global _sam2, _device
    with _sam2_lock:
        if _sam2 is None:
            from transformers import Sam2Model, Sam2Processor

            if _device is None:
                _device = pick_device()
            model = Sam2Model.from_pretrained(SAM2_MODEL_ID).to(_device).eval()
            processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
            _sam2 = (model, processor)
        return _sam2


def get_img2img():
    global _img2img, _device, _pipe
    import diffusers

    plan = reimagine_plan()
    key = f"{plan['pipeline']}:{plan['repo']}"
    with _img2img_lock:
        if _img2img is None or _img2img[0] != key:
            if _device is None:
                _device = pick_device()
            set_progress(phase="loading", model=plan["repo"], cached=repo_cached(plan["repo"]))
            if _img2img is not None:
                _img2img = None
                gc.collect()
                torch.cuda.empty_cache()
            pipe = None
            # When reimagine runs on the same repo as the loaded fill pipeline,
            # share its components instead of loading a second copy — critical
            # for big models (Qwen 20B would otherwise be resident twice = OOM).
            same_repo = plan["repo"] == MODEL_ID
            if _pipe is not None and same_repo:
                try:
                    pipe = getattr(diffusers, plan["pipeline"]).from_pipe(_pipe)
                except Exception as exc:
                    print(f"from_pipe({plan['pipeline']}) failed: {exc!r} — "
                          f"loading fresh instead", flush=True)
                    pipe = None
            if pipe is None and same_repo and _pipe is not None:
                # Sharing failed but it's the same big model — drop the fill
                # pipe so only ONE copy is ever resident (it reloads lazily on
                # the next fill). Two copies of Qwen 20B would exceed the A100.
                _pipe = None
                gc.collect()
                torch.cuda.empty_cache()
            if pipe is None:
                dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
                    plan["dtype"], torch.bfloat16
                )
                extra = {}
                if plan["vae"]:
                    extra["vae"] = diffusers.AutoencoderKL.from_pretrained(
                        plan["vae"], torch_dtype=dtype
                    ).to(_device)
                pipe = load_pipeline(
                    getattr(diffusers, plan["pipeline"]), plan["repo"], dtype, extra
                )
            _img2img = (key, pipe)
        return _img2img[1]


def data_url_to_image(data_url: str) -> Image.Image:
    try:
        b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception as exc:
        raise HTTPException(400, f"bad image payload: {exc}") from exc


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def snap16(v: int) -> int:
    return max(16, (v // 16) * 16)


def soft_mask(mask_l, dilate, blur):
    """Morphological soft mask: dilate the hard mask outward, then feather it.
    Dilating before the blur keeps the selection fully covered while still giving
    a gradual transition into the original pixels (Fooocus morphological_open)."""
    m = mask_l
    if dilate > 0:
        m = m.filter(ImageFilter.MaxFilter(dilate * 2 + 1))
    if blur > 0:
        m = m.filter(ImageFilter.GaussianBlur(blur))
    return m


def feather_stitch(result, original, mask_l):
    """Alpha composite through a bled-out, feathered mask. The seam sits in
    background well past the subject edge and is blended over a wide soft band."""
    soft = soft_mask(mask_l, STITCH_BLEED, MASK_FEATHER * 2)
    return Image.composite(result, original, soft)


def poisson_stitch(result, original, mask_l):
    """Gradient-domain seamless clone (Poisson, à la Photoshop Healing): clones
    the generated region into the original so boundary gradients match — erasing
    any residual tonal seam, not just blurring it. Falls back to feather on the
    pathological cases OpenCV's seamlessClone can't handle (mask at the border)."""
    bled = mask_l.filter(ImageFilter.MaxFilter(STITCH_BLEED * 2 + 1))
    mask = np.array(bled)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return original

    src = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    dst = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    h, w = mask_bin.shape
    cx, cy = int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2)
    touches = xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2

    try:
        if touches:
            # seamlessClone misbehaves when the mask hits the image border, so
            # pad with reflected pixels, clone, then crop back.
            p = 96
            src_p = cv2.copyMakeBorder(src, p, p, p, p, cv2.BORDER_REFLECT)
            dst_p = cv2.copyMakeBorder(dst, p, p, p, p, cv2.BORDER_REFLECT)
            mask_p = cv2.copyMakeBorder(mask_bin, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
            out_p = cv2.seamlessClone(src_p, dst_p, mask_p, (cx + p, cy + p), cv2.NORMAL_CLONE)
            out = out_p[p:p + h, p:p + w]
        else:
            out = cv2.seamlessClone(src, dst, mask_bin, (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error:
        return feather_stitch(result, original, mask_l)

    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def stitch(result, original, mask_l, method=None):
    method = method or STITCH
    if method == "poisson":
        return poisson_stitch(result, original, mask_l)
    return feather_stitch(result, original, mask_l)


class GenerateRequest(BaseModel):
    image: str               # data URL, full canvas composite (RGB or RGBA)
    mask: str                # data URL, white = regenerate, black = keep
    prompt: str
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    stitch: str | None = None    # override: "poisson" | "feather"
    count: int = 1               # variations to generate in one batch


@app.get("/health")
def health():
    plan = reimagine_plan()
    info = {
        "backend": BACKEND,
        "model": MODEL_ID,
        "loaded": _pipe is not None,
        "sam": SAM_MODEL_ID,
        "sam_loaded": _sam is not None,
        "img2img": plan["repo"],
        "reimagine_backend": plan["backend"],
        "img2img_loaded": _img2img is not None
        and _img2img[0] == f"{plan['pipeline']}:{plan['repo']}",
        "device": _device,
    }
    info["gpus"] = vram_info()
    return info


@app.get("/progress")
def progress():
    return _progress


@app.get("/models")
def list_models():
    import diffusers

    return {
        "current": BACKEND,
        "models": [
            {
                "key": k,
                "repo": v["repo"],
                "txt2img": bool(v.get("txt2img")),
                # backends whose pipeline class is missing from this diffusers
                # build are shown but not selectable in the UI
                "available": hasattr(diffusers, v["pipeline"]),
                # uncached repos download (possibly tens of GB) on first Generate
                "cached": repo_cached(v["repo"]),
                # what /reimagine runs on when this backend is selected
                "reimagine": "native"
                if v.get("img2img_pipeline") and hasattr(diffusers, v["img2img_pipeline"])
                else "chroma",
            }
            for k, v in MODELS.items()
        ],
    }


class ModelSwitchRequest(BaseModel):
    backend: str


@app.post("/model")
def switch_model(req: ModelSwitchRequest):
    """Switch the inpaint backend at runtime. The current pipeline is dropped
    (freeing VRAM); the new one lazy-loads on the next /generate. An in-flight
    generation keeps its local reference, so it finishes on the old model."""
    global BACKEND, MODEL, MODEL_ID, _pipe, _img2img
    if req.backend not in MODELS:
        raise HTTPException(400, f"unknown backend {req.backend!r}; pick one of {list(MODELS)}")
    leak_warning = None
    with _pipe_lock:
        if req.backend != BACKEND:
            BACKEND = req.backend
            MODEL = MODELS[BACKEND]
            MODEL_ID = MODEL["repo"]
            if _pipe is not None or _img2img is not None:
                # drop the reimagine pipe too — if it shares components with the
                # fill pipe (from_pipe), keeping it would pin the old weights
                _pipe = None
                with _img2img_lock:
                    _img2img = None
                gc.collect()
                torch.cuda.empty_cache()
                # verify the unload actually freed the memory — only the small
                # persistent models (SAM segmenters) should remain after a sweep
                if _device and torch.cuda.is_available():
                    idx = int(_device.split(":")[1]) if ":" in _device else 0
                    held = torch.cuda.memory_allocated(idx) / 2**30
                    if held > 8:
                        leak_warning = (
                            f"{held:.1f} GiB still allocated after unload — "
                            "a previous model may have leaked; restart the "
                            "service if loads start failing"
                        )
                        print(f"WARNING: {leak_warning}", flush=True)
    resp = {
        "backend": BACKEND,
        "model": MODEL_ID,
        "loaded": _pipe is not None,
        "vram": vram_info(),
    }
    if leak_warning:
        resp["warning"] = leak_warning
    return resp


class SegmentRequest(BaseModel):
    image: str               # data URL, full canvas composite
    text: str                # concept to segment, e.g. "the dog"
    threshold: float = 0.5


@app.post("/segment")
def segment(req: SegmentRequest):
    image = data_url_to_image(req.image).convert("RGB")
    try:
        model, processor = get_sam()
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "gated" in msg.lower() or "403" in msg:
            raise HTTPException(
                403,
                "facebook/sam3 is gated: accept the license on huggingface.co and "
                "run sam-env\\Scripts\\hf.exe auth login",
            ) from exc
        raise

    inputs = processor(images=image, text=req.text, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=req.threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results["masks"]
    if masks is None or len(masks) == 0:
        raise HTTPException(404, f"nothing matching {req.text!r} found")

    # union all instance masks into one selection
    union = masks.any(dim=0).cpu().numpy().astype("uint8") * 255
    mask_img = Image.fromarray(union, mode="L")
    bbox = mask_img.getbbox()
    scores = [round(float(s), 3) for s in results["scores"].tolist()]

    return {
        "mask": image_to_data_url(mask_img),
        "bbox": list(bbox),
        "count": len(masks),
        "scores": scores,
    }


class SegmentPointRequest(BaseModel):
    image: str               # data URL
    points: list[list[float]]   # [[x, y], ...] in image pixels
    labels: list[int] | None = None   # 1 = include, 0 = exclude; defaults to all 1


@app.post("/segment-point")
def segment_point(req: SegmentPointRequest):
    image = data_url_to_image(req.image).convert("RGB")
    if not req.points:
        raise HTTPException(400, "at least one point required")
    model, processor = get_sam2()

    labels = req.labels or [1] * len(req.points)
    inputs = processor(
        images=image,
        input_points=[[req.points]],     # batch > object > points
        input_labels=[[labels]],
        return_tensors="pt",
    ).to(_device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]
    scores = outputs.iou_scores.cpu().reshape(-1)
    masks = masks.reshape(-1, *masks.shape[-2:])      # [num_proposals, H, W]
    best = int(scores.argmax())
    mask = (masks[best].numpy() > 0).astype("uint8") * 255

    mask_img = Image.fromarray(mask, mode="L")
    bbox = mask_img.getbbox()
    if bbox is None:
        raise HTTPException(404, "no object found at that point")
    return {
        "mask": image_to_data_url(mask_img),
        "bbox": list(bbox),
        "score": round(float(scores[best]), 3),
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    image = data_url_to_image(req.image).convert("RGB")
    mask = data_url_to_image(req.mask).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    bbox = mask.getbbox()
    if bbox is None:
        raise HTTPException(400, "mask is empty — make a selection first")

    W, H = image.size

    # Full-image inpaint: resize the WHOLE image to the model's resolution and
    # let it denoise only the masked region while decoding everything together,
    # so the mask boundary is harmonized by the VAE (no pixel paste-back seam).
    scale = min(1.0, MAX_GEN_SIDE / max(W, H))
    gen_w = snap16(round(W * scale))
    gen_h = snap16(round(H * scale))
    gen_img = image.resize((gen_w, gen_h), Image.LANCZOS)
    # mask fed to the model is dilated so it fully owns the selection edge pixels
    model_mask = mask.filter(ImageFilter.MaxFilter(MASK_DILATE * 2 + 1))
    gen_mask = model_mask.resize((gen_w, gen_h), Image.NEAREST)

    try:
        pipe = get_pipe()
        generator = (
            torch.Generator(device=_device).manual_seed(req.seed)
            if req.seed is not None
            else None
        )
        guidance = req.guidance if req.guidance is not None else MODEL["guidance"]
        count = max(1, min(4, req.count))
        steps = req.steps or MODEL["steps"]
        cb = callback_kwargs_for(pipe, steps)

        set_progress(phase="denoising", step=0, total=steps)
        if MODEL.get("txt2img"):
            # Prompt-only model (Z-Image): no image/mask conditioning. Generate full
            # frames at the doc resolution; the stitch step below drops only the
            # selected region back over the original (Poisson seam-blended).
            results = pipe(
                prompt=req.prompt,
                width=gen_w,
                height=gen_h,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                num_images_per_prompt=count,
                **cb,
            ).images
        elif MODEL.get("cn_union_inpaint"):
            # ControlNet-Union inpaint (mode 7): the control image is the photo with
            # the masked region zeroed, so the ControlNet conditions the fill on the
            # surrounding scene — harmonizing the boundary instead of leaving a seam.
            cn_np = np.array(gen_img)
            cn_np[np.array(gen_mask) > 0] = 0
            cn_img = Image.fromarray(cn_np)
            results = pipe(
                prompt=req.prompt,
                image=gen_img,
                mask_image=gen_mask,
                control_image=[cn_img],
                control_mode=[7],
                width=gen_w,
                height=gen_h,
                num_inference_steps=steps,
                guidance_scale=guidance,
                strength=1.0,
                controlnet_conditioning_scale=MODEL.get("cn_scale", 1.0),
                generator=generator,
                num_images_per_prompt=count,
                **cb,
            ).images
        else:
            kwargs = dict(
                prompt=req.prompt,
                image=gen_img,
                mask_image=gen_mask,
                width=gen_w,
                height=gen_h,
                num_inference_steps=steps,
                generator=generator,
                num_images_per_prompt=count,
                **cb,
            )
            if MODEL.get("true_cfg"):
                kwargs["true_cfg_scale"] = guidance
                kwargs["negative_prompt"] = " "
            else:
                kwargs["guidance_scale"] = guidance
            if MODEL["img2img"]:
                kwargs["strength"] = 1.0          # fully regenerate the masked region
            else:
                kwargs["max_sequence_length"] = 512
            results = pipe(**kwargs).images

        # Upscale each decoded result to full resolution, then stitch only the
        # generated region back into the original — bled out past the selection so
        # the seam lands in background, and (by default) Poisson-blended so any
        # residual tonal difference is erased rather than just feathered.
        set_progress(phase="stitching")
        images = [
            image_to_data_url(stitch(r.resize((W, H), Image.LANCZOS), image, mask, req.stitch))
            for r in results
        ]
        return {"images": images, "region": list(bbox)}
    finally:
        set_progress()


class ReimagineRequest(BaseModel):
    image: str               # data URL, full canvas
    prompt: str
    likeness: int = 55       # 0 = loose reinterpretation, 100 = stay close to original
    count: int = 3           # variations to generate in one batch
    steps: int | None = None # default: the active reimagine model's own step count
    seed: int | None = None


@app.post("/reimagine")
def reimagine(req: ReimagineRequest):
    image = data_url_to_image(req.image).convert("RGB")
    W, H = image.size
    scale = min(1.0, MAX_GEN_SIDE / max(W, H))
    gen_w = snap16(round(W * scale))
    gen_h = snap16(round(H * scale))
    gen_img = image.resize((gen_w, gen_h), Image.LANCZOS)

    # higher likeness -> lower denoise strength (stays closer to the original)
    likeness = max(0, min(100, req.likeness))
    strength = round(0.95 - (likeness / 100) * 0.78, 3)   # 0.95 (loose) .. 0.17 (close)
    count = max(1, min(6, req.count))

    try:
        plan = reimagine_plan()
        pipe = get_img2img()
        generator = (
            torch.Generator(device=_device).manual_seed(req.seed)
            if req.seed is not None
            else None
        )
        steps = req.steps or plan["steps"]
        # img2img runs ~steps×strength actual denoise iterations
        total_est = max(1, round(steps * strength))
        set_progress(phase="denoising", step=0, total=total_est)
        # one batched forward pass produces all variations (distinct initial noise)
        results = pipe(
            prompt=req.prompt,
            image=gen_img,
            strength=strength,
            guidance_scale=plan["guidance"],
            num_inference_steps=steps,
            num_images_per_prompt=count,
            generator=generator,
            **callback_kwargs_for(pipe, total_est),
        ).images

        images = [image_to_data_url(r.resize((W, H), Image.LANCZOS)) for r in results]
        return {"images": images, "strength": strength, "model": plan["repo"]}
    finally:
        set_progress()


if __name__ == "__main__":
    host = os.environ.get("PHOTOPROD_HOST", "0.0.0.0")   # LAN-accessible by default
    print(f"Photo Prod GenAI service · {MODEL_ID}")
    print(f"http://{host}:{PORT}  (POST /generate, /segment, /segment-point, GET /health)")
    uvicorn.run(app, host=host, port=PORT)
