#!/usr/bin/env python3
"""Make a vertical narrated reel (1080x1920) for Maps and Musings.

Usage: python3 tools/reel.py spec.json out.mp4

spec.json:
  lines    list of narration sentences (required)
  card     path to a news card JPG (used as the visual when no photos)
  photos   optional list of photo paths/URLs (Ken Burns slideshow)
  title    optional big title shown at the start over photos
  kicker   optional small label over the title
  credits  optional credit line shown at the end (e.g. "Photos: X / Unsplash")
Voice: Kokoro (British male blend). Model files live in ~/tts (downloaded if missing).
"""
import io, json, math, os, subprocess, sys, tempfile, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

W, H, FPS = 1080, 1920, 25
NAVY = (11, 22, 40); INK = (245, 246, 250); GOLD = (232, 176, 74)
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
REG = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SERIF = "/usr/share/fonts/truetype/google-fonts/Lora-Variable.ttf"
TTS = os.path.expanduser("~/tts") if os.path.exists(os.path.expanduser("~/tts/model.onnx")) else "/home/claude/tts"


def font(p, s, bold=False):
    try:
        f = ImageFont.truetype(p, s)
        if bold and "Lora" in p:
            try: f.set_variation_by_name("Bold")
            except Exception: pass
        return f
    except Exception:
        return ImageFont.truetype(BOLD, s)


def ensure_tts():
    os.makedirs(TTS, exist_ok=True)
    if os.path.exists(f"{TTS}/model.onnx") and os.path.exists(f"{TTS}/voices.npz"):
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", "kokoro-onnx", "soundfile", "huggingface_hub"], check=True)
    from huggingface_hub import hf_hub_download
    import shutil
    r = "onnx-community/Kokoro-82M-v1.0-ONNX"
    shutil.copy(hf_hub_download(r, "onnx/model.onnx"), f"{TTS}/model.onnx")
    vs = {v: np.fromfile(hf_hub_download(r, f"voices/{v}.bin"), dtype=np.float32).reshape(-1, 1, 256) for v in ["bm_george", "bm_fable"]}
    np.savez(f"{TTS}/voices.npz", **vs)


def narrate(lines, tmp):
    ensure_tts()
    from kokoro_onnx import Kokoro
    import soundfile as sf
    k = Kokoro(f"{TTS}/model.onnx", f"{TTS}/voices.npz")
    v = k.get_voice_style("bm_george") * 0.75 + k.get_voice_style("bm_fable") * 0.25
    out, times, t, sr = [], [], 0.0, 24000
    for ln in lines:
        a, sr = k.create(ln, voice=v, speed=0.95, lang="en-gb")
        times.append((t, t + len(a) / sr, ln)); out.append(a)
        out.append(np.zeros(int(sr * 0.45), dtype=np.float32)); t += len(a) / sr + 0.45
    raw = os.path.join(tmp, "raw.wav"); sf.write(raw, np.concatenate(out), sr)
    voice = os.path.join(tmp, "voice.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-af",
                    "highpass=f=60,equalizer=f=180:t=q:w=1:g=2.5,equalizer=f=3200:t=q:w=1.2:g=2,"
                    "acompressor=threshold=-20dB:ratio=3:attack=8:release=120,aecho=0.8:0.5:35|70:0.10|0.06,"
                    "loudnorm=I=-16:TP=-1.5:LRA=7", "-ar", "44100", voice], check=True)
    return voice, times, t


def music(dur, path):
    import soundfile as sf
    sr = 44100; n = int(dur * sr); tt = np.arange(n) / sr
    note = lambda m: 440 * 2 ** ((m - 69) / 12)
    chords = [[50, 57, 62, 65], [46, 53, 58, 62], [41, 48, 53, 57], [48, 55, 60, 64]]
    mus = np.zeros(n, np.float32); cl = 6.0
    for ci in range(int(dur / cl) + 1):
        st = int(ci * cl * sr); en = min(n, int((ci + 1) * cl * sr + 2 * sr))
        if st >= n: break
        s = tt[st:en] - tt[st]; env = np.minimum(1, s / 2) * np.clip(1 - (s - cl) / 2, 0, 1)
        for m in chords[ci % 4]:
            f = note(m); mus[st:en] += (0.5 * np.sin(2 * np.pi * f * s) + 0.2 * np.sin(2 * np.pi * f * 2.003 * s)) * env
    rev = mus.copy()
    for dl, g in ((0.11, 0.5), (0.23, 0.35), (0.37, 0.25)):
        k = int(dl * sr); rev[k:] += g * mus[:-k]
    rev /= np.abs(rev).max() + 1e-6
    rev *= np.minimum(1, tt / 2) * np.clip((dur - tt) / 2, 0, 1) * 0.14
    sf.write(path, np.stack([rev, rev], 1).astype(np.float32), sr)


def load(src):
    if str(src).startswith("http"):
        if "images.unsplash.com" in src and "?" not in src: src += "?fm=jpg&q=88&w=2000"
        data = urllib.request.urlopen(urllib.request.Request(src, headers={"User-Agent": "MapsAndMusings/1.0"}), timeout=60).read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(src).convert("RGB")


def cover(im, w, h, zoom=1.0, cx=0.5, cy=0.5):
    r = max(w / im.width, h / im.height) * zoom
    cw, ch = w / r, h / r
    x0 = min(max(cx * im.width - cw / 2, 0), im.width - cw); y0 = min(max(cy * im.height - ch / 2, 0), im.height - ch)
    return im.resize((w, h), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))


def ease(t): t = max(0.0, min(1.0, t)); return t * t * (3 - 2 * t)


def chunks(times):
    out = []
    for a, b, txt in times:
        words, cur, parts = txt.split(), [], []
        for w in words:
            cur.append(w)
            if len(" ".join(cur)) > 24: parts.append(" ".join(cur)); cur = []
        if cur: parts.append(" ".join(cur))
        tot = sum(len(p) for p in parts); t0 = a
        for p in parts:
            d = (b - a) * len(p) / tot; out.append((t0, t0 + d, p)); t0 += d
    return out


def make(spec, out):
    tmp = tempfile.mkdtemp()
    voice, times, nar = narrate(spec["lines"], tmp)
    LEAD, TAIL = 0.6, 2.6
    dur = LEAD + nar + TAIL
    music(dur, os.path.join(tmp, "music.wav"))
    subs = [(a + LEAD, b + LEAD, t) for a, b, t in chunks(times)]
    photos = [load(p) for p in spec.get("photos", [])]
    card = load(spec["card"]) if spec.get("card") else None
    if card is not None and not photos:
        bg = cover(card, W, H, 1.0).filter(ImageFilter.GaussianBlur(40))
        bg = ImageEnhance.Brightness(bg).enhance(0.45)
    fsub = font(BOLD, 54); ftitle = font(SERIF, 120, True); fk = font(BOLD, 36); fend = font(BOLD, 50); fsm = font(REG, 30)
    n = int(dur * FPS); seg = nar / max(1, len(photos)) if photos else 0
    yy, xx = np.mgrid[0:H, 0:W]
    vig = np.clip(1 - 0.5 * (((xx - W / 2) / (W * 0.8)) ** 2 + ((yy - H / 2) / (H * 0.8)) ** 2), 0.5, 1)[..., None].astype(np.float32)
    grad = np.ones((H, 1, 1), np.float32); grad[int(H * 0.6):, 0, 0] = np.linspace(1, 0.45, H - int(H * 0.6))
    shade = vig * grad
    fr_dir = os.path.join(tmp, "f"); os.makedirs(fr_dir)
    for i in range(n):
        t = i / FPS
        if photos:
            tt = max(0.0, t - LEAD); k = min(len(photos) - 1, int(tt / seg)) if seg else 0
            u = (tt - k * seg) / seg if seg else 0
            dirn = 1 if k % 2 == 0 else -1
            img = cover(photos[k], W, H, 1.08 + 0.10 * ease(u), 0.5 + 0.06 * dirn * (ease(u) - 0.5), 0.5)
            if k + 1 < len(photos) and tt > (k + 1) * seg - 0.6:
                nxt = cover(photos[k + 1], W, H, 1.08, 0.5 - 0.03 * dirn, 0.5)
                img = Image.blend(img, nxt, ease((tt - ((k + 1) * seg - 0.6)) / 0.6))
            a = np.asarray(img).astype(np.float32) * shade
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        else:
            z = 1.0 + 0.06 * (t / dur)
            img = bg.copy()
            cw, ch = int(1000 * z), int(1250 * z)
            c = card.resize((cw, ch), Image.LANCZOS)
            img.paste(c, ((W - cw) // 2, (H - ch) // 2 - 60))
        d = ImageDraw.Draw(img)
        if photos and spec.get("title") and t < LEAD + times[0][1] + 0.3:
            al = ease((t - 0.2) / 0.8) * (1 - ease((t - (LEAD + times[0][1] - 0.3)) / 0.5))
            if al > 0:
                c8 = tuple(int(v * al) for v in GOLD); ci = tuple(int(v * al) for v in INK)
                if spec.get("kicker"): d.text((W / 2, 560), spec["kicker"].upper(), font=fk, fill=c8, anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))
                lines, cur = [], ""
                for w in spec["title"].split():
                    tst = (cur + " " + w).strip()
                    if d.textlength(tst, font=ftitle) < W - 120: cur = tst
                    else: lines.append(cur); cur = w
                lines.append(cur)
                for j, ln in enumerate(lines):
                    d.text((W / 2, 680 + j * 140), ln, font=ftitle, fill=ci, anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))
        for a0, b0, txt in subs:
            if a0 <= t < b0:
                ysub = 1560 if photos else 1790
                tw = d.textlength(txt, font=fsub)
                d.rounded_rectangle([W / 2 - tw / 2 - 26, ysub - 46, W / 2 + tw / 2 + 26, ysub + 46], radius=18, fill=(0, 0, 0))
                d.text((W / 2, ysub), txt, font=fsub, fill=INK, anchor="mm")
        if t > LEAD + nar + 0.1:
            al = ease((t - LEAD - nar - 0.1) / 0.6)
            ov = Image.new("RGB", (W, H), NAVY); img = Image.blend(img, ov, 0.94 * al); d = ImageDraw.Draw(img)
            d.text((W / 2, 900), "MAPS AND MUSINGS", font=fend, fill=tuple(int(v * al) for v in INK), anchor="mm")
            d.text((W / 2, 975), "Follow @aveiwfromhere", font=fk, fill=tuple(int(v * al) for v in GOLD), anchor="mm")
            if spec.get("credits"): d.text((W / 2, 1700), spec["credits"], font=fsm, fill=tuple(int(170 * al) for _ in range(3)), anchor="mm")
        if t < 0.4: img = Image.blend(Image.new("RGB", (W, H)), img, t / 0.4)
        img.save(os.path.join(fr_dir, f"{i:05d}.jpg"), quality=90)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(fr_dir, "%05d.jpg"),
                    "-i", voice, "-i", os.path.join(tmp, "music.wav"),
                    "-filter_complex", f"[1:a]adelay={int(LEAD*1000)}|{int(LEAD*1000)},aresample=44100[v];[v][2:a]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-profile:v", "high", "-pix_fmt", "yuv420p",
                    "-crf", "24", "-maxrate", "6M", "-bufsize", "12M", "-c:a", "aac", "-b:a", "160k", "-t", f"{dur:.2f}",
                    "-movflags", "+faststart", out], check=True)
    subprocess.run(["rm", "-rf", tmp])
    print("ok", out, round(dur, 1), "s")


if __name__ == "__main__":
    make(json.load(open(sys.argv[1])), sys.argv[2])
