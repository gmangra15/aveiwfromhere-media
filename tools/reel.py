#!/usr/bin/env python3
"""Make a professional vertical narrated reel (1080x1920) for Maps and Musings.

Usage: python3 tools/reel.py spec.json out.mp4

spec.json:
  lines    list of narration sentences (required; numbers written as words)
  scenes   list of scenes, one per narration line (the last scene repeats if fewer).
           {"type":"map", "countries":["FRA"], "pin":[lat,lon], "label":"Paris", "span":18}
               animated map that flies in from a wide view; countries = ISO3 codes to highlight
           {"type":"photo", "src":"https://images.unsplash.com/photo-...", "tag":"File photo"}
               real licensed photo with a slow camera move (tag optional, shown small)
           {"type":"stat", "big":"2,397", "small":"ARRESTS ON SUNDAY", "src":"<optional photo>"}
               animated number callout (counts up if numeric)
           {"type":"people", "people":[{"src":..., "name":"Masoud Pezeshkian", "role":"President of Iran"}], "caption":"Met in Tehran", "credit":"Photo: ..."}
               real, licensed portraits (1-2) of public figures in the story, with name/role bars
           {"type":"card", "src":"cards/<file>.jpg"}   the branded news card
  title, kicker   headline shown over the first scene
  credits         end-screen credit line (photo credits)
Legacy specs with only "card" and/or "photos" still work.
"""
import io, json, math, os, re, subprocess, sys, tempfile, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

W, H, FPS = 1080, 1920, 25
NAVY = (11, 22, 40); INK = (245, 246, 250); GOLD = (232, 176, 74); MUTED = (176, 186, 204)
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
REG = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SERIF = "/usr/share/fonts/truetype/google-fonts/Lora-Variable.ttf"
HERE = os.path.dirname(os.path.abspath(__file__))
TTS = "/home/claude/tts"
GEO = os.path.join(HERE, "ne_50m_countries.geojson")
GEO_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"
FC = {}


def font(p, s, bold=False):
    k = (p, s, bold)
    if k not in FC:
        try:
            f = ImageFont.truetype(p, s)
            if bold and "Lora" in p:
                try: f.set_variation_by_name("Bold")
                except Exception: pass
        except Exception:
            f = ImageFont.truetype(BOLD, s)
        FC[k] = f
    return FC[k]


def ease(t): t = max(0.0, min(1.0, t)); return t * t * (3 - 2 * t)
def eo(t): t = max(0.0, min(1.0, t)); return 1 - (1 - t) ** 3


# ---------------- audio ----------------
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


def music(dur, path, mood="news"):
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
    if mood == "news":  # soft pulse for news
        pulse = (np.sin(2 * np.pi * 1.5 * tt) > 0.92).astype(np.float32)
        kick = np.convolve(pulse, np.exp(-np.arange(int(0.12 * sr)) / (0.03 * sr)) * np.sin(2 * np.pi * 55 * np.arange(int(0.12 * sr)) / sr), mode="same")
        mus += 0.6 * kick.astype(np.float32)
    rev = mus.copy()
    for dl, g in ((0.11, 0.5), (0.23, 0.35), (0.37, 0.25)):
        k = int(dl * sr); rev[k:] += g * mus[:-k]
    rev /= np.abs(rev).max() + 1e-6
    rev *= np.minimum(1, tt / 2) * np.clip((dur - tt) / 2, 0, 1) * 0.13
    sf.write(path, np.stack([rev, rev], 1).astype(np.float32), sr)


# ---------------- visuals ----------------
def load(src):
    src = str(src)
    if src.startswith("http"):
        if "images.unsplash.com" in src and "?" not in src: src += "?fm=jpg&q=88&w=2000"
        data = urllib.request.urlopen(urllib.request.Request(src, headers={"User-Agent": "MapsAndMusings/1.0"}), timeout=60).read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    if not os.path.isabs(src): src = os.path.join(os.path.dirname(HERE), src)
    return Image.open(src).convert("RGB")


def cover(im, w, h, zoom=1.0, cx=0.5, cy=0.5):
    r = max(w / im.width, h / im.height) * zoom
    cw, ch = w / r, h / r
    x0 = min(max(cx * im.width - cw / 2, 0), im.width - cw); y0 = min(max(cy * im.height - ch / 2, 0), im.height - ch)
    return im.resize((w, h), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))


_shade = None
def shade(img, strength=1.0):
    global _shade
    if _shade is None:
        yy, xx = np.mgrid[0:H, 0:W]
        vig = np.clip(1 - 0.55 * (((xx - W / 2) / (W * 0.8)) ** 2 + ((yy - H / 2) / (H * 0.8)) ** 2), 0.45, 1)
        g = np.ones((H, W), np.float32)
        g[int(H * 0.62):] *= np.linspace(1, 0.4, H - int(H * 0.62))[:, None]
        g[:int(H * 0.25)] *= np.linspace(0.55, 1, int(H * 0.25))[:, None]
        _shade = (vig * g)[..., None].astype(np.float32)
    a = np.asarray(img).astype(np.float32)
    s = 1 - (1 - _shade) * strength
    return Image.fromarray(np.clip(a * s, 0, 255).astype(np.uint8))


_geo = None
def geo():
    global _geo
    if _geo is None:
        if not os.path.exists(GEO):
            urllib.request.urlretrieve(GEO_URL, GEO)
        d = json.load(open(GEO))
        _geo = []
        for f in d["features"]:
            code = f["properties"].get("ADM0_A3") or f["properties"].get("ISO_A3")
            g = f["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            rings = [np.array(p[0], dtype=np.float64) for p in polys]
            bb = (min(r[:, 0].min() for r in rings), min(r[:, 1].min() for r in rings),
                  max(r[:, 0].max() for r in rings), max(r[:, 1].max() for r in rings))
            _geo.append((code, rings, bb))
    return _geo


def merc(lat): return np.log(np.tan(np.pi / 4 + np.radians(np.clip(lat, -85, 85)) / 2))


def map_frame(sc, u, t_abs):
    clat, clon = sc.get("pin", [30, 70])
    span_end = sc.get("span", 22.0); span_start = sc.get("span_start", 120.0)
    p = eo(u * 2.2)  # fly-in during first ~45% of the scene
    span = span_start * (span_end / span_start) ** p * (1 - 0.04 * u)
    lon0 = clon + (1 - p) * sc.get("from_lon_offset", 0)
    lat0 = clat
    scale = W / np.radians(span)
    img = Image.new("RGB", (W, H), (7, 14, 28)); d = ImageDraw.Draw(img)
    # graticule
    step = 10 if span > 40 else 5
    for lo in range(-180, 181, step):
        x = (np.radians(lo - lon0)) * scale + W / 2
        if 0 <= x <= W: d.line([(x, 0), (x, H)], fill=(14, 26, 46), width=1)
    for la in range(-80, 81, step):
        y = H / 2 - (merc(la) - merc(lat0)) * scale
        if 0 <= y <= H: d.line([(0, y), (W, y)], fill=(14, 26, 46), width=1)
    hl = set(sc.get("countries", []))
    half_lon = span * 0.6; half_lat = span * 1.2
    for code, rings, bb in geo():
        if bb[2] < lon0 - half_lon - 5 or bb[0] > lon0 + half_lon + 5 or bb[3] < lat0 - half_lat - 5 or bb[1] > lat0 + half_lat + 5:
            continue
        on = code in hl
        for r in rings:
            xs = np.radians(r[:, 0] - lon0) * scale + W / 2
            ys = H / 2 - (merc(r[:, 1]) - merc(lat0)) * scale
            pts = list(zip(xs.tolist(), ys.tolist()))
            if len(pts) < 3: continue
            if on:
                a = ease(u * 3)
                fill = tuple(int(28 + (GOLD[i] * 0.55 - 28) * a) for i in range(3))
                d.polygon(pts, fill=fill, outline=GOLD)
            else:
                d.polygon(pts, fill=(26, 40, 64), outline=(52, 72, 104))
    # pin
    px, py = W / 2, H / 2
    pa = ease((u - 0.35) * 4)
    if pa > 0:
        for k in range(3):
            ph = ((t_abs * 0.8) + k / 3) % 1
            r = 14 + ph * 70
            d.ellipse([px - r, py - r, px + r, py + r], outline=tuple(int(c * (1 - ph) * pa) for c in GOLD), width=4)
        d.ellipse([px - 14, py - 14, px + 14, py + 14], fill=GOLD, outline=(255, 255, 255), width=3)
        if sc.get("label"):
            f = font(BOLD, 44); lab = sc["label"].upper(); tw = d.textlength(lab, font=f)
            bx = px + 40; by = py - 34
            if bx + tw + 40 > W - 30: bx = px - 40 - tw - 40
            d.rounded_rectangle([bx, by, bx + tw + 40, by + 68], radius=12, fill=(0, 0, 0))
            d.text((bx + 20, by + 34), lab, font=f, fill=INK, anchor="lm")
    return img


def photo_frame(sc, u, k, cache):
    im = cache[sc["src"]]
    dirn = 1 if k % 2 == 0 else -1
    fr = cover(im, W, H, 1.06 + 0.12 * ease(u), 0.5 + 0.07 * dirn * (ease(u) - 0.5), sc.get("cy", 0.5))
    fr = shade(fr)
    if sc.get("tag"):
        d = ImageDraw.Draw(fr); f = font(BOLD, 26); tw = d.textlength(sc["tag"].upper(), font=f)
        d.rounded_rectangle([W - tw - 90, 150, W - 50, 196], radius=8, fill=(0, 0, 0))
        d.text((W - 70, 173), sc["tag"].upper(), font=f, fill=MUTED, anchor="rm")
    return fr


def people_frame(sc, u, cache):
    """1-2 real, licensed portraits with TV-style name/role lower-thirds.
    sc = {"type":"people", "people":[{"src":..., "name":..., "role":..., "cy":0.35}], "caption":"MET IN TEHRAN", "credit":"Photo: ..."}"""
    ppl = sc["people"][:2]
    first = cache[ppl[0]["src"]]
    bg = ImageEnhance.Brightness(cover(first, W, H, 1.15).filter(ImageFilter.GaussianBlur(45))).enhance(0.3)
    img = bg; d = ImageDraw.Draw(img)
    a = eo(u * 4); z = 1.0 + 0.06 * ease(u)
    if len(ppl) == 1:
        boxes = [((W - 820) // 2, 300, 820, 980)]
    else:
        boxes = [(50, 380, 470, 640), (W - 520, 380, 470, 640)]
    for (bx, by, bw, bh), p in zip(boxes, ppl):
        im = cache[p["src"]]
        crop = cover(im, int(bw * z), int(bh * z), 1.0, p.get("cx", 0.5), p.get("cy", 0.35))
        crop = crop.crop(((crop.width - bw) // 2, (crop.height - bh) // 2, (crop.width - bw) // 2 + bw, (crop.height - bh) // 2 + bh))
        mask = Image.new("L", (bw, bh), 0); ImageDraw.Draw(mask).rounded_rectangle([0, 0, bw, bh], radius=28, fill=int(255 * a))
        yoff = int(40 * (1 - a))
        img.paste(crop, (bx, by + yoff), mask)
        d.rounded_rectangle([bx - 3, by + yoff - 3, bx + bw + 3, by + bh + yoff + 3], radius=30, outline=tuple(int(c * a) for c in GOLD), width=4)
        # lower third
        ly = by + bh + 30 + yoff
        fn = font(BOLD, 52 if len(ppl) == 1 else 38); fr_ = font(BOLD, 32 if len(ppl) == 1 else 26)
        name = p["name"]; role = p.get("role", "").upper()
        tw = max(d.textlength(name, font=fn), d.textlength(role, font=fr_)) + 50
        cx = bx + bw / 2
        d.rounded_rectangle([cx - tw / 2, ly, cx + tw / 2, ly + (130 if len(ppl) == 1 else 104)], radius=14, fill=(0, 0, 0))
        d.rectangle([cx - tw / 2, ly, cx - tw / 2 + 10, ly + (130 if len(ppl) == 1 else 104)], fill=GOLD)
        d.text((cx + 5, ly + (44 if len(ppl) == 1 else 36)), name, font=fn, fill=tuple(int(c * a) for c in INK), anchor="mm")
        d.text((cx + 5, ly + (98 if len(ppl) == 1 else 78)), role, font=fr_, fill=tuple(int(c * a) for c in GOLD), anchor="mm")
    if sc.get("caption"):
        f = font(BOLD, 40); cap = sc["caption"].upper(); tw = d.textlength(cap, font=f)
        d.rounded_rectangle([W / 2 - tw / 2 - 24, 230, W / 2 + tw / 2 + 24, 290], radius=10, fill=tuple(int(c * a) for c in GOLD))
        d.text((W / 2, 260), cap, font=f, fill=NAVY, anchor="mm")
    if sc.get("credit"):
        d.text((W - 50, 1470), sc["credit"], font=font(REG, 24), fill=(150, 150, 160), anchor="rm")
    return img


def stat_frame(sc, u, cache):
    if sc.get("src"):
        bg = cover(cache[sc["src"]], W, H, 1.1 + 0.05 * u).filter(ImageFilter.GaussianBlur(18))
        bg = ImageEnhance.Brightness(bg).enhance(0.35)
    else:
        bg = Image.new("RGB", (W, H), NAVY)
        d0 = ImageDraw.Draw(bg)
        for x in range(0, W, 60): d0.line([(x, 0), (x, H)], fill=(16, 29, 50))
        for y in range(0, H, 60): d0.line([(0, y), (W, y)], fill=(16, 29, 50))
    d = ImageDraw.Draw(bg)
    big = sc["big"]; m = re.fullmatch(r"([^\d]*)([\d,\.]+)(.*)", big)
    if m:
        num = float(m.group(2).replace(",", "")); dec = len(m.group(2).split(".")[1]) if "." in m.group(2) else 0
        val = num * eo(u * 1.8)
        s = f"{val:,.{dec}f}" if "," in m.group(2) else f"{val:.{dec}f}"
        big = m.group(1) + s + m.group(3)
    a = eo(u * 4)
    fb = font(BOLD, 190 if len(big) <= 7 else 140)
    d.text((W / 2, 860 + 40 * (1 - a)), big, font=fb, fill=tuple(int(c * a) for c in INK), anchor="mm")
    d.rectangle([W / 2 - 60, 990, W / 2 + 60, 998], fill=tuple(int(c * a) for c in GOLD))
    fs = font(BOLD, 40); words = sc.get("small", "").upper().split(); lines, cur = [], ""
    for w in words:
        tst = (cur + " " + w).strip()
        if d.textlength(tst, font=fs) < W - 160: cur = tst
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    for j, ln in enumerate(lines):
        d.text((W / 2, 1060 + j * 56), ln, font=fs, fill=tuple(int(c * a) for c in GOLD), anchor="mm")
    return bg


def card_frame(sc, u, cache):
    card = cache[sc["src"]]
    key = "__bg__" + sc["src"]
    if key not in cache:
        cache[key] = ImageEnhance.Brightness(cover(card, W, H).filter(ImageFilter.GaussianBlur(40))).enhance(0.45)
    img = cache[key].copy()
    z = 1.0 + 0.05 * u
    cw, ch = int(1000 * z), int(1250 * z)
    img.paste(card.resize((cw, ch), Image.LANCZOS), ((W - cw) // 2, (H - ch) // 2 - 80))
    return img


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


def legacy_scenes(spec, n):
    if spec.get("photos"):
        return [{"type": "photo", "src": spec["photos"][i % len(spec["photos"])]} for i in range(n)]
    return [{"type": "card", "src": spec["card"]}] * n


def make(spec, out):
    tmp = tempfile.mkdtemp()
    voice, times, nar = narrate(spec["lines"], tmp)
    LEAD, TAIL = 0.6, 2.8
    dur = LEAD + nar + TAIL
    if dur < 60 and not spec.get("allow_short"):
        sys.exit(f"TOO SHORT: reel would be {dur:.1f}s; write more narration lines (minimum 60s)")
    scenes = spec.get("scenes") or legacy_scenes(spec, len(times))
    scenes = [scenes[min(i, len(scenes) - 1)] for i in range(len(times))]
    music(dur, os.path.join(tmp, "music.wav"), "news" if any(s["type"] in ("map", "stat") for s in scenes) else "calm")
    cache = {}
    for s in scenes:
        if s.get("src") and s["src"] not in cache: cache[s["src"]] = load(s["src"])
        for pp in s.get("people", []):
            if pp["src"] not in cache: cache[pp["src"]] = load(pp["src"])
    starts = [LEAD + times[0][0] - LEAD] + [LEAD + times[i][0] - 0.2 for i in range(1, len(times))]
    ends = starts[1:] + [dur]
    subs = [(a + LEAD, b + LEAD, t) for a, b, t in chunks(times)]
    fsub = font(BOLD, 54); ftitle = font(SERIF, 112, True); fk = font(BOLD, 34); fend = font(BOLD, 52)
    fsm = font(REG, 28); fbrand = font(BOLD, 28)
    XF = 0.5

    def scene_img(i, t):
        sc = scenes[i]; u = (t - starts[i]) / max(0.1, ends[i] - starts[i] + XF)
        if sc["type"] == "map": return map_frame(sc, u, t)
        if sc["type"] == "photo": return photo_frame(sc, u, i, cache)
        if sc["type"] == "stat": return stat_frame(sc, u, cache)
        if sc["type"] == "people": return people_frame(sc, u, cache)
        return card_frame(sc, u, cache)

    fr_dir = os.path.join(tmp, "f"); os.makedirs(fr_dir)
    n = int(dur * FPS)
    for fi in range(n):
        t = fi / FPS
        i = max(j for j in range(len(scenes)) if starts[j] <= t)
        img = scene_img(i, t)
        if i + 1 < len(scenes) and t > starts[i + 1] - XF:
            img = Image.blend(img, scene_img(i + 1, t), ease((t - (starts[i + 1] - XF)) / XF))
        d = ImageDraw.Draw(img)
        # brand bar
        d.rectangle([60, 80, 68, 116], fill=GOLD)
        d.text((86, 98), "MAPS AND MUSINGS", font=fbrand, fill=INK, anchor="lm", stroke_width=2, stroke_fill=(0, 0, 0))
        # title over first scene
        if spec.get("title") and t < starts[1] if len(starts) > 1 else False:
            al = (1.0 if t < 0.9 else 1.0) * (1 - ease((t - (starts[1] - 0.6)) / 0.5))
            if al > 0:
                y0 = 300
                if spec.get("kicker"):
                    kk = spec["kicker"].upper(); tw = d.textlength(kk, font=fk)
                    d.rounded_rectangle([W / 2 - tw / 2 - 20, y0 - 30, W / 2 + tw / 2 + 20, y0 + 30], radius=8, fill=tuple(int(c * al) for c in GOLD))
                    d.text((W / 2, y0), kk, font=fk, fill=tuple(int(c * al) for c in NAVY), anchor="mm")
                lines, cur = [], ""
                for w in spec["title"].split():
                    tst = (cur + " " + w).strip()
                    if d.textlength(tst, font=ftitle) < W - 120: cur = tst
                    else: lines.append(cur); cur = w
                lines.append(cur)
                for j, ln in enumerate(lines[:3]):
                    d.text((W / 2, y0 + 120 + j * 128), ln, font=ftitle, fill=tuple(int(c * al) for c in INK), anchor="mm",
                           stroke_width=3, stroke_fill=(0, 0, 0))
        # subtitles
        for a0, b0, txt in subs:
            if a0 <= t < b0:
                ysub = 1580
                tw = d.textlength(txt, font=fsub)
                d.rounded_rectangle([W / 2 - tw / 2 - 26, ysub - 46, W / 2 + tw / 2 + 26, ysub + 46], radius=18, fill=(0, 0, 0))
                d.text((W / 2, ysub), txt, font=fsub, fill=INK, anchor="mm")
        # progress bar
        d.rectangle([0, H - 10, int(W * min(1, t / (LEAD + nar))), H], fill=GOLD)
        # end screen
        if t > LEAD + nar + 0.1:
            al = ease((t - LEAD - nar - 0.1) / 0.6)
            img = Image.blend(img, Image.new("RGB", (W, H), NAVY), 0.94 * al); d = ImageDraw.Draw(img)
            d.text((W / 2, 900), "MAPS AND MUSINGS", font=fend, fill=tuple(int(v * al) for v in INK), anchor="mm")
            d.text((W / 2, 975), "Follow @aveiwfromhere", font=fk, fill=tuple(int(v * al) for v in GOLD), anchor="mm")
            if spec.get("credits"):
                d.text((W / 2, 1700), spec["credits"], font=fsm, fill=tuple(int(170 * al) for _ in range(3)), anchor="mm")
        img.save(os.path.join(fr_dir, f"{fi:05d}.jpg"), quality=90)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(fr_dir, "%05d.jpg"),
                    "-i", voice, "-i", os.path.join(tmp, "music.wav"),
                    "-filter_complex", f"[1:a]adelay={int(LEAD*1000)}|{int(LEAD*1000)},aresample=44100[v];[v][2:a]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-profile:v", "high", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-maxrate", "7M", "-bufsize", "14M", "-c:a", "aac", "-b:a", "160k", "-t", f"{dur:.2f}",
                    "-movflags", "+faststart", out], check=True)
    subprocess.run(["rm", "-rf", tmp])
    print("ok", out, round(dur, 1), "s")


if __name__ == "__main__":
    make(json.load(open(sys.argv[1])), sys.argv[2])
