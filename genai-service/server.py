"""Photo Prod generative fill service.

FastAPI inpainting/outpainting service. Receives an image + mask + prompt,
fills the masked region on the GPU (prefers the A100 when more than one CUDA
device is present), and returns the full-size result.

Backend is selected with the PHOTOPROD_MODEL env var (see MODELS below):
  klein      FLUX.2 Klein 4B     Apache 2.0, ungated, inpaint-conditioned (default)
  flux-fill  FLUX.1 Fill dev     highest quality, but GATED + NON-COMMERCIAL (opt-in)
  chroma     Chroma1-HD          Apache 2.0, ungated, uncensored, general img2img
  qwen       Qwen-Image 20B      Apache 2.0, ungated

The default is an Apache-2.0 model so a fresh checkout is commercial-safe. Set
PHOTOPROD_MODEL=flux-fill for the best blending, but its non-commercial license
then applies to your use of that model.

Inpainting is full-image and latent-composited (Fooocus-style, no pixel paste):
the whole image is encoded, only the masked region is denoised, and the model
DECODES the entire image together so the mask boundary is harmonized by the VAE
— there is no seam to blend. The untouched areas of the original photo are then
restored at full resolution through a morphological soft mask, so only the
generated region changes and the rest stays pixel-exact.

Run:  sam-env\\Scripts\\python.exe genai-service\\server.py
"""

import base64
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
    "chroma": {
        "repo": "lodestones/Chroma1-HD",
        "pipeline": "ChromaInpaintPipeline",
        "guidance": 4.0,
        "steps": 30,
        "img2img": True,    # masked-img2img style: takes strength
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
    },
    "flux-fill": {
        "repo": "black-forest-labs/FLUX.1-Fill-dev",
        "pipeline": "FluxFillPipeline",
        "guidance": 30.0,
        "steps": 28,
        "img2img": False,   # dedicated fill model: no strength param
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
_pipe_lock = threading.Lock()
_device = None

SAM_MODEL_ID = "facebook/sam3"
_sam = None
_sam_lock = threading.Lock()

SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
_sam2 = None
_sam2_lock = threading.Lock()

# whole-image "reimagine" (img2img) — a creative variation guided by image+text,
# separate from masked inpainting. Uses Chroma (ungated, uncensored).
IMG2IMG_MODEL_ID = "lodestones/Chroma1-HD"
_img2img = None
_img2img_lock = threading.Lock()


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
            _pipe = cls.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(_device)
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
    global _img2img, _device
    with _img2img_lock:
        if _img2img is None:
            from diffusers import ChromaImg2ImgPipeline

            if _device is None:
                _device = pick_device()
            _img2img = ChromaImg2ImgPipeline.from_pretrained(
                IMG2IMG_MODEL_ID, torch_dtype=torch.bfloat16
            ).to(_device)
        return _img2img


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
    info = {
        "backend": BACKEND,
        "model": MODEL_ID,
        "loaded": _pipe is not None,
        "sam": SAM_MODEL_ID,
        "sam_loaded": _sam is not None,
        "img2img": IMG2IMG_MODEL_ID,
        "img2img_loaded": _img2img is not None,
        "device": _device,
    }
    if torch.cuda.is_available():
        info["gpus"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_properties(i).name,
                "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1),
            }
            for i in range(torch.cuda.device_count())
        ]
    return info


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

    pipe = get_pipe()
    generator = (
        torch.Generator(device=_device).manual_seed(req.seed)
        if req.seed is not None
        else None
    )
    guidance = req.guidance if req.guidance is not None else MODEL["guidance"]
    kwargs = dict(
        prompt=req.prompt,
        image=gen_img,
        mask_image=gen_mask,
        width=gen_w,
        height=gen_h,
        num_inference_steps=req.steps or MODEL["steps"],
        generator=generator,
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
    count = max(1, min(4, req.count))
    kwargs["num_images_per_prompt"] = count
    results = pipe(**kwargs).images

    # Upscale each decoded result to full resolution, then stitch only the
    # generated region back into the original — bled out past the selection so
    # the seam lands in background, and (by default) Poisson-blended so any
    # residual tonal difference is erased rather than just feathered.
    images = [
        image_to_data_url(stitch(r.resize((W, H), Image.LANCZOS), image, mask, req.stitch))
        for r in results
    ]
    return {"images": images, "region": list(bbox)}


class ReimagineRequest(BaseModel):
    image: str               # data URL, full canvas
    prompt: str
    likeness: int = 55       # 0 = loose reinterpretation, 100 = stay close to original
    count: int = 3           # variations to generate in one batch
    steps: int = 30
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

    pipe = get_img2img()
    generator = (
        torch.Generator(device=_device).manual_seed(req.seed)
        if req.seed is not None
        else None
    )
    # one batched forward pass produces all variations (distinct initial noise)
    results = pipe(
        prompt=req.prompt,
        image=gen_img,
        strength=strength,
        guidance_scale=4.0,
        num_inference_steps=req.steps,
        num_images_per_prompt=count,
        generator=generator,
    ).images

    images = [image_to_data_url(r.resize((W, H), Image.LANCZOS)) for r in results]
    return {"images": images, "strength": strength}


if __name__ == "__main__":
    host = os.environ.get("PHOTOPROD_HOST", "0.0.0.0")   # LAN-accessible by default
    print(f"Photo Prod GenAI service · {MODEL_ID}")
    print(f"http://{host}:{PORT}  (POST /generate, /segment, /segment-point, GET /health)")
    uvicorn.run(app, host=host, port=PORT)
