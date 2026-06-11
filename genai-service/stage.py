"""Stage and verify every model backend the Photo Prod service offers.

For each backend listed by /models, this switches the live service to it,
runs a tiny /generate (and /reimagine where the backend supports it natively),
and reports OK/FAIL with timings. One model is resident at a time — switching
unloads the previous one — so this also exercises the runtime-switch path the
UI uses. First runs download any missing weights into the HF cache.

Run while the service is up:
  sam-env\\Scripts\\python.exe genai-service\\stage.py
Exit code 0 = every backend staged clean.
"""

import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8765"


def api(path, payload=None, timeout=1800):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:           # surface the server's detail
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:200]}") from exc
    except urllib.error.URLError as exc:
        print(f"\nSERVICE UNREACHABLE on {path} — it likely crashed; check its window/logs.",
              flush=True)
        sys.exit(2)


def test_images():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 512), (90, 120, 160))
    ImageDraw.Draw(img).ellipse((60, 320, 452, 500), fill=(70, 90, 70))
    mask = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(mask).rectangle((200, 200, 312, 330), fill=255)

    def durl(im, gray=False):
        b = io.BytesIO()
        im.convert("L" if gray else "RGB").save(b, "PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    return durl(img), durl(mask, gray=True)


def main():
    info = api("/models", timeout=30)
    start_backend = info["current"]
    img, mask = test_images()
    failures = []

    print(f"staging {len(info['models'])} backends (current: {start_backend})\n", flush=True)
    for m in info["models"]:
        key = m["key"]
        if not m["available"]:
            print(f"{key:14} SKIP — pipeline class missing from diffusers", flush=True)
            failures.append(key)
            continue

        api("/model", {"backend": key}, timeout=120)

        # fill — z-image keeps its distilled default step count, others run short
        try:
            t0 = time.time()
            payload = {"image": img, "mask": mask, "prompt": "a red apple", "count": 1}
            if key != "z-image":
                payload["steps"] = 12
            api("/generate", payload)
            fill = f"fill OK {time.time() - t0:5.1f}s"
        except Exception as exc:
            fill = f"fill FAIL — {exc}"
            failures.append(key)

        # reimagine — only where the backend runs it natively
        if m.get("reimagine") == "native":
            try:
                t0 = time.time()
                payload = {"image": img, "prompt": "a misty lake", "count": 1}
                if key != "z-image":
                    payload["steps"] = 12
                out = api("/reimagine", payload)
                rmg = f"reimagine OK {time.time() - t0:5.1f}s"
            except Exception as exc:
                rmg = f"reimagine FAIL — {exc}"
                failures.append(key)
        else:
            rmg = "reimagine -> chroma fallback"   # ASCII: Windows consoles are cp1252

        print(f"{key:14} {fill}   {rmg}", flush=True)

    api("/model", {"backend": start_backend}, timeout=120)
    print(f"\nrestored backend: {start_backend}", flush=True)

    bad = sorted(set(failures))
    if bad:
        print(f"FAILED: {', '.join(bad)}", flush=True)
        sys.exit(1)
    print("all backends staged clean", flush=True)


if __name__ == "__main__":
    main()
