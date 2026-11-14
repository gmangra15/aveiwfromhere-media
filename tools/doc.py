#!/usr/bin/env python3
"""Render a 16:9 narrated documentary for Maps and Musings (YouTube).

Usage: python3 tools/doc.py spec.json out.mp4 [--only SHOT_INDEX,...] [--still SHOT:FRAC ...]

Spec (JSON):
  title, subtitle, voice, speed, credits (list of strings), sources (list of {title,url})
  shots: list of shots. Every shot may have "lines" (narration sentences), "chapter" (marks a YouTube chapter),
         "lower": ["Title", "Subtitle"] (lower third), "pre"/"post" (seconds of silence), "dur" (if no lines).
    {"type":"intro"}                                   logo sting
    {"type":"sat", "cam":[[lat,lon,zoom],...], "rivers":[...], "labels":[...], "borders":true, "highlight":[ISO3...]}
        animated satellite map (Sentinel-2 cloudless 2016, EOX) with camera keyframes and vector overlays
        rivers: {"name":"Indus","t0":0,"t1":1,"color":[94,200,255],"width":5,"flow":true}
        labels: {"lat","lon","text","style":"pin|region|city|water","at":<line index in shot>, "until":<line idx>}
    {"type":"title", "text":"THE INDUS", "sub":"...", "bg":[lat,lon,zoom]}        main / chapter title card
    {"type":"flight", "spec":"reels/flights/x.json"}   3D flyover (tools/flight.py) at 16:9
    {"type":"stat", "big":"92%", "small":"...", "bg":[lat,lon,zoom]}
    {"type":"bars", "items":[{"label":"1833","value":13900,"text":"13,900 km²"}], "caption":"...", "bg":[...]}
    {"type":"timeline", "events":[{"year":"c. 2500 BCE","text":"..."}], "bg":[...]}  (event i appears with line i)
    {"type":"quote", "text":"...", "who":"...", "bg":[...]}
    {"type":"outro"}
Also writes out.srt (subtitles) and out.txt (YouTube description with chapters and sources).
"""
import io, json, math, os, subprocess, sys, tempfile, shutil
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cairo

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
W, H, FPS = 1920, 1080, 25
NAVY = (11, 22, 40); INK = (245, 246, 250); GOLD = (232, 176, 74); MUTED = (176, 186, 204); WATER = (94, 200, 255)
INTER = "/usr/share/fonts/opentype/inter/Inter-{}.otf"
SERIF = "/usr/share/fonts/truetype/google-fonts/Lora-Variable.ttf"
TILECACHE = "/home/claude/flighttest/tilecache"
FC = {}


def F(kind, size):
    k = (kind, size)
    if k not in FC:
        if kind == "serif":
            f = ImageFont.truetype(SERIF, size)
            try: f.set_variation_by_name("Bold")
            except Exception: pass
        else:
            f = ImageFont.truetype(INTER.format(kind), size)
        FC[k] = f
    return FC[k]


def ease(t): t = max(0.0, min(1.0, t)); return t * t * (3 - 2 * t)
def eo(t): t = max(0.0, min(1.0, t)); return 1 - (1 - t) ** 3
def eio(t): t = max(0.0, min(1.0, t)); return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


# ---------------------------------------------------------------- tiles / satellite
import requests
_S = None; _TILES = {}
def tile_img(z, x, y, src="eox"):
    global _S
    n = 2 ** z; x %= n
    if y < 0 or y >= n: return None
    k = (src, z, x, y)
    if k in _TILES: return _TILES[k]
    p = os.path.join(TILECACHE, f"img_{z}_{x}_{y}" if src == "eox" else f"bm_{z}_{x}_{y}")
    if not os.path.exists(p):
        if _S is None: _S = requests.Session(); _S.headers["User-Agent"] = "MapsAndMusings/1.0"
        u = (f"https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/{z}/{y}/{x}.jpg" if src == "eox" else
             f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_NextGeneration/default/2004-08-01/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg")
        for a in range(5):
            try:
                r = _S.get(u, timeout=60); r.raise_for_status()
                tmpf = f"{p}.{os.getpid()}.part"; open(tmpf, "wb").write(r.content); os.replace(tmpf, p); break  # atomic
            except requests.RequestException:
                if a == 4: raise
                import time; time.sleep(2 * (a + 1))
    try:
        im = np.asarray(Image.open(p).convert("RGB"))
    except Exception:
        os.remove(p); return tile_img(z, x, y, src)
    if len(_TILES) > 900: _TILES.clear()
    _TILES[k] = im
    return im


def wx(lon, z): return (lon + 180.0) / 360.0 * 256 * 2 ** z
def wy(lat, z):
    lat = np.clip(lat, -85, 85)
    return (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2 * 256 * 2 ** z


class Cam:
    def __init__(self, lat, lon, zoom):
        self.lat, self.lon, self.zoom = lat, lon, zoom
        self.cx, self.cy = wx(lon, zoom), wy(lat, zoom)
    def px(self, lat, lon):
        return (wx(np.asarray(lon, float), self.zoom) - self.cx + W / 2, wy(np.asarray(lat, float), self.zoom) - self.cy + H / 2)


BM_FULL, EOX_FULL = 6.0, 6.9
def sat_view(lat, lon, zoom, maxz=13):
    """NASA Blue Marble (public domain) for continent-scale views, Sentinel-2 cloudless (EOX) when zoomed in, blended between."""
    if zoom <= BM_FULL: return _view(lat, lon, zoom, "bm", 8)
    if zoom >= EOX_FULL: return _view(lat, lon, zoom, "eox", maxz)
    r = ease((zoom - BM_FULL) / (EOX_FULL - BM_FULL))
    return Image.blend(_view(lat, lon, zoom, "bm", 8), _view(lat, lon, zoom, "eox", maxz), r)


def _view(lat, lon, zoom, src, maxz):
    L = int(min(maxz, max(1, math.ceil(zoom - 0.15) + (1 if src == "bm" else 0))))
    s = 2 ** (zoom - L)
    cx, cy = wx(lon, L), wy(lat, L)
    hw, hh = W / 2 / s, H / 2 / s
    x0, y0, x1, y1 = cx - hw, cy - hh, cx + hw, cy + hh
    tx0, ty0, tx1, ty1 = int(x0 // 256), int(y0 // 256), int(x1 // 256), int(y1 // 256)
    can = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), np.uint8)
    can[:] = (12, 30, 52)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            t = tile_img(L, tx, ty, src)
            if t is not None:
                can[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = t
    im = Image.fromarray(can)
    bx0, by0 = x0 - tx0 * 256, y0 - ty0 * 256
    return im.resize((W, H), Image.BICUBIC, box=(bx0, by0, bx0 + 2 * hw, by0 + 2 * hh), reducing_gap=2.0)


_VIG = None; _GRAIN = None
def grade(im, dark=0.0, sat=0.92, contrast=1.08):
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = ((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2
        _VIG = np.clip(1 - 0.42 * r, 0.35, 1)[..., None]
    a = np.asarray(im).astype(np.float32)
    m = a.mean(axis=2, keepdims=True)
    a = m + (a - m) * sat
    a = (a - 128) * contrast + 128
    a = a * np.array([0.97, 1.0, 1.04], np.float32)  # slightly cool, cinematic
    a = a * _VIG * (1 - dark)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def grain(im, fi):
    global _GRAIN
    if _GRAIN is None:
        rng = np.random.default_rng(7)
        _GRAIN = [rng.normal(0, 3.2, (H, W, 1)).astype(np.float32) for _ in range(4)]
    a = np.asarray(im).astype(np.float32) + _GRAIN[fi % 4]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- vector data
_RIV = None; _CTRY = None
def rivers():
    global _RIV
    if _RIV is None:
        _RIV = {k: np.array(v, float) for k, v in json.load(open(os.path.join(HERE, "data", "indus_rivers.json"))).items()}
    return _RIV


def countries():
    global _CTRY
    if _CTRY is None:
        d = json.load(open(os.path.join(HERE, "ne_50m_countries.geojson")))
        _CTRY = []
        for f in d["features"]:
            code = f["properties"].get("ADM0_A3") or f["properties"].get("ISO_A3")
            g = f["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            rings = [np.array(r, float) for p in polys for r in p[:1]]
            _CTRY.append((code, rings))
    return _CTRY


def to_surface(im):
    a = np.asarray(im.convert("RGB"))
    bgra = np.empty((H, W, 4), np.uint8); bgra[..., 0] = a[..., 2]; bgra[..., 1] = a[..., 1]; bgra[..., 2] = a[..., 0]; bgra[..., 3] = 255
    surf = cairo.ImageSurface.create_for_data(memoryview(bgra), cairo.FORMAT_ARGB32, W, H)
    return surf, bgra


def from_surface(surf, bgra):
    surf.flush()
    return Image.fromarray(np.ascontiguousarray(bgra[..., [2, 1, 0]]))


def polyline(ctx, xs, ys):
    ctx.move_to(float(xs[0]), float(ys[0]))
    for x, y in zip(xs[1:], ys[1:]): ctx.line_to(float(x), float(y))


def partial(xs, ys, frac):
    if frac >= 1: return xs, ys
    d = np.r_[0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
    if d[-1] == 0 or frac <= 0: return xs[:1], ys[:1]
    tgt = d[-1] * frac; i = int(np.searchsorted(d, tgt))
    i = max(1, min(i, len(xs) - 1)); r = (tgt - d[i - 1]) / max(1e-9, d[i] - d[i - 1])
    return np.r_[xs[:i], xs[i - 1] + (xs[i] - xs[i - 1]) * r], np.r_[ys[:i], ys[i - 1] + (ys[i] - ys[i - 1]) * r]


def draw_overlays(im, cam, sh, u, tl, t):
    """u: shot progress 0..1, tl: time (s) since shot start; t: absolute time."""
    surf, buf = to_surface(im); ctx = cairo.Context(surf)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND); ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    if sh.get("borders") or sh.get("highlight"):
        hl = set(sh.get("highlight", []))
        for code, rings in countries():
            for r in rings:
                if r[:, 0].max() < cam.lon - 60 or r[:, 0].min() > cam.lon + 60: continue
                xs, ys = cam.px(r[:, 1], r[:, 0])
                if xs.max() < -50 or xs.min() > W + 50 or ys.max() < -50 or ys.min() > H + 50: continue
                polyline(ctx, xs, ys); ctx.close_path()
                if code in hl:
                    al = ease(tl / 1.2)
                    ctx.set_source_rgba(232 / 255, 176 / 255, 74 / 255, 0.16 * al); ctx.fill_preserve()
                    ctx.set_source_rgba(232 / 255, 176 / 255, 74 / 255, 0.9 * al); ctx.set_line_width(2.6); ctx.stroke()
                elif sh.get("borders"):
                    ctx.set_source_rgba(1, 1, 1, 0.38); ctx.set_line_width(1.3); ctx.set_dash([6, 5]); ctx.stroke(); ctx.set_dash([])
                else:
                    ctx.new_path()
    R = rivers()
    for rv in sh.get("rivers", []):
        pts = R[rv["name"]]
        if "at" in rv:
            a0 = sh["_ltime"](rv["at"]) / max(0.1, sh["_dur"]); a1 = a0 + rv.get("draw", 3.0) / max(0.1, sh["_dur"])
        else:
            a0, a1 = rv.get("t0", 0), rv.get("t1", 0.001)
        frac = eio((u - a0) / max(1e-3, a1 - a0)) if a1 > a0 else 1
        if frac <= 0: continue
        xs, ys = cam.px(pts[:, 1], pts[:, 0]); xs, ys = partial(xs, ys, frac)
        if len(xs) < 2: continue
        col = [c / 255 for c in rv.get("color", WATER)]; wd = rv.get("width", 4.5)
        fade = rv.get("alpha", 1.0) * (1 - ease((cam.zoom - 8.4) / 1.2))  # coarse river lines fade out in close-ups
        if fade <= 0.02: continue
        for w_, a_ in ((wd * 5, 0.10), (wd * 2.6, 0.22), (wd, 0.95)):
            polyline(ctx, xs, ys); ctx.set_source_rgba(*col, a_ * fade); ctx.set_line_width(w_); ctx.stroke()
        if rv.get("flow"):
            polyline(ctx, xs, ys); ctx.set_source_rgba(1, 1, 1, 0.55 * fade); ctx.set_line_width(max(1.5, wd * 0.45))
            ctx.set_dash([10, 26], -tl * 60); ctx.stroke(); ctx.set_dash([])
        if frac < 1:  # glowing head
            hx, hy = xs[-1], ys[-1]
            g = cairo.RadialGradient(hx, hy, 0, hx, hy, wd * 6)
            g.add_color_stop_rgba(0, 1, 1, 1, 0.95 * fade); g.add_color_stop_rgba(0.3, *col, 0.7 * fade); g.add_color_stop_rgba(1, *col, 0)
            ctx.set_source(g); ctx.arc(hx, hy, wd * 6, 0, 2 * math.pi); ctx.fill()
    # pins (vector part)
    labs = []
    for lb in sh.get("labels", []):
        ta = sh["_ltime"](lb.get("at", 0)); tb = sh["_ltime"](lb["until"]) if "until" in lb else 1e9
        al = ease((tl - ta) / 0.6) * (1 - ease((tl - tb) / 0.6))
        if al <= 0.01: continue
        x, y = cam.px(lb["lat"], lb["lon"]); x, y = float(x), float(y)
        st = lb.get("style", "pin")
        if st in ("pin", "city"):
            pulse = (tl - ta) % 2.0 / 2.0
            ctx.set_source_rgba(232 / 255, 176 / 255, 74 / 255, 0.6 * (1 - pulse) * al); ctx.set_line_width(2.5)
            ctx.arc(x, y, 9 + 26 * pulse, 0, 2 * math.pi); ctx.stroke()
            ctx.set_source_rgba(0, 0, 0, 0.5 * al); ctx.arc(x, y, 10, 0, 2 * math.pi); ctx.fill()
            ctx.set_source_rgba(232 / 255, 176 / 255, 74 / 255, al); ctx.arc(x, y, 7, 0, 2 * math.pi); ctx.fill()
            ctx.set_source_rgba(1, 1, 1, al); ctx.set_line_width(2); ctx.arc(x, y, 7, 0, 2 * math.pi); ctx.stroke()
        labs.append((x, y, lb, al, st, tl - ta))
    im = from_surface(surf, buf)
    d = ImageDraw.Draw(im, "RGBA")
    for x, y, lb, al, st, age in labs:
        txt = lb["text"]; a8 = int(255 * al)
        dx, dy = lb.get("dx", 0), lb.get("dy", 0)
        if st == "region":
            f = F("SemiBold", lb.get("size", 34)); s = " ".join(txt.upper())
            d.text((x + dx, y + dy - 6 * (1 - eo(age / 0.8))), s, font=f, fill=(255, 255, 255, int(a8 * 0.88)), anchor="mm",
                   stroke_width=3, stroke_fill=(0, 0, 0, int(a8 * 0.35)))
        elif st == "water":
            f = F("MediumItalic", lb.get("size", 30))
            d.text((x + dx, y + dy), txt, font=f, fill=(170, 225, 255, a8), anchor="mm", stroke_width=3, stroke_fill=(0, 20, 40, int(a8 * 0.5)))
        else:
            f = F("SemiBold", lb.get("size", 30)); tw = d.textlength(txt, font=f)
            side = lb.get("side", "r")
            bx = x + 22 + dx if side == "r" else x - 22 - tw - 32 + dx
            by = y - 24 + dy
            slide = (1 - eo(age / 0.5)) * (-14 if side == "r" else 14)
            d.rounded_rectangle([bx + slide, by, bx + slide + tw + 32, by + 48], radius=8, fill=(8, 16, 30, int(a8 * 0.82)))
            d.rectangle([bx + slide, by, bx + slide + 4, by + 48], fill=GOLD + (a8,))
            d.text((bx + slide + 18, by + 24), txt, font=f, fill=(245, 246, 250, a8), anchor="lm")
            if lb.get("sub"):
                fs = F("Regular", 22)
                d.text((bx + slide + 18, by + 64), lb["sub"], font=fs, fill=(220, 226, 236, a8), anchor="lm", stroke_width=2, stroke_fill=(0, 0, 0, int(a8 * 0.6)))
    return im


def cam_at(keys, u):
    if len(keys) == 1: return keys[0]
    n = len(keys) - 1; p = eio(u) * n if len(keys) == 2 else u * n
    i = min(int(p), n - 1); r = p - i
    if len(keys) > 2: r = eio(r)
    a, b = keys[i], keys[i + 1]
    # interpolate in projected space at a fixed zoom for straight paths; zoom linearly
    z = a[2] + (b[2] - a[2]) * r
    za = 10
    xa, ya, xb, yb = wx(a[1], za), wy(a[0], za), wx(b[1], za), wy(b[0], za)
    x, y = xa + (xb - xa) * r, ya + (yb - ya) * r
    lon = x / (256 * 2 ** za) * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / (256 * 2 ** za)))))
    return lat, lon, z


# ---------------------------------------------------------------- shot renderers
_BG = {}
def bg_still(key, slow=0.0):
    """darkened, blurred satellite background for cards (cached), with slow push-in."""
    k = tuple(key)
    if k not in _BG:
        im = grade(sat_view(*key), dark=0.15)
        _BG[k] = im
    im = _BG[k]
    z = 1 + 0.06 * slow
    if z > 1.001:
        cw, ch = W / z, H / z
        im = im.resize((W, H), Image.BICUBIC, box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))
    return im


def card_bg(sh, u, dark=0.55, blur=6):
    key = sh.get("bg", [30.0, 70.0, 5.0])
    im = bg_still(key, u)
    if blur: im = im.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(im).astype(np.float32) * (1 - dark) + np.array(NAVY, np.float32) * dark * 0.6
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def wrap(d, text, f, maxw):
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= maxw: cur = t
        else: out.append(cur); cur = w
    if cur: out.append(cur)
    return out


def r_sat(sh, u, tl, t, fi):
    lat, lon, z = cam_at(sh["cam"], u)
    im = grade(sat_view(lat, lon, z), dark=sh.get("dark", 0.05))
    return draw_overlays(im, Cam(lat, lon, z), sh, u, tl, t)


def r_title(sh, u, tl, t, fi):
    im = card_bg(sh, u, dark=0.5, blur=3 if sh.get("main") else 8)
    if sh.get("rivers"):
        lat, lon, z = sh["bg"]
        im = draw_overlays(im, Cam(lat, lon, z * 1.0 + 0.06 * u * 0), dict(sh, labels=[]), min(1, u * 2.2), tl, t)
    d = ImageDraw.Draw(im, "RGBA")
    a = ease(tl / 0.9)
    if sh.get("chapter"):
        f = F("SemiBold", 30); s = " ".join(sh["chapter"].upper())
        d.text((W / 2, 430 - 10 * (1 - eo(tl / 1.0))), s, font=f, fill=GOLD + (int(255 * a),), anchor="mm")
    big = sh.get("main")
    f = F("serif", 150 if big else 92)
    lines = wrap(d, sh["text"], f, W - 300)
    y0 = H / 2 - (len(lines) - 1) * (f.size * 0.6) + (10 if big else 30)
    for j, ln in enumerate(lines):
        aj = ease((tl - 0.25 - 0.15 * j) / 0.9)
        d.text((W / 2, y0 + j * f.size * 1.15 + 18 * (1 - eo((tl - 0.25) / 1.2))), ln, font=f, fill=(245, 246, 250, int(255 * aj)), anchor="mm",
               stroke_width=2, stroke_fill=(0, 0, 0, int(80 * aj)))
    lw = 260 * eo((tl - 0.6) / 0.9)
    yl = y0 + (len(lines) - 1) * f.size * 1.15 + f.size * 0.75
    d.rectangle([W / 2 - lw / 2, yl, W / 2 + lw / 2, yl + 4], fill=GOLD + (255,))
    if sh.get("sub"):
        fs = F("Light", 40); asub = ease((tl - 1.0) / 0.9)
        d.text((W / 2, yl + 60), sh["sub"], font=fs, fill=(230, 234, 242, int(255 * asub)), anchor="mm")
    return im


def r_stat(sh, u, tl, t, fi):
    im = card_bg(sh, u)
    d = ImageDraw.Draw(im, "RGBA")
    big = sh["big"]; a = ease(tl / 0.7)
    import re
    m = re.match(r"^([^\d]*)([\d,.]+)(.*)$", big)
    if m and sh.get("count", True):
        num = float(m.group(2).replace(",", "")); dec = len(m.group(2).split(".")[1]) if "." in m.group(2) else 0
        v = num * eo(tl / 1.6)
        s = f"{v:,.{dec}f}" if "," in m.group(2) or num >= 1000 else f"{v:.{dec}f}"
        big = m.group(1) + s + m.group(3)
    f = F("serif", sh.get("size", 200))
    d.text((W / 2, H / 2 - 60), big, font=f, fill=(245, 246, 250, int(255 * a)), anchor="mm")
    lw = 200 * eo((tl - 0.3) / 0.8)
    d.rectangle([W / 2 - lw / 2, H / 2 + 70, W / 2 + lw / 2, H / 2 + 74], fill=GOLD + (255,))
    fs = F("Medium", 44); a2 = ease((tl - 0.5) / 0.8)
    for j, ln in enumerate(wrap(d, sh["small"], fs, W - 500)):
        d.text((W / 2, H / 2 + 140 + j * 58), ln, font=fs, fill=(225, 230, 240, int(255 * a2)), anchor="mm")
    return im


def r_bars(sh, u, tl, t, fi):
    im = card_bg(sh, u)
    d = ImageDraw.Draw(im, "RGBA")
    items = sh["items"]; mx = max(it["value"] for it in items)
    if sh.get("caption"):
        d.text((W / 2, 170), sh["caption"], font=F("SemiBold", 46), fill=(245, 246, 250, int(255 * ease(tl / 0.6))), anchor="mm")
    n = len(items); bw = min(300, 1200 / n - 60); gap = (W - 2 * 360 - n * bw) / max(1, n - 1) if n > 1 else 0
    base = H - 220; top = 300
    for i, it in enumerate(items):
        x = 360 + i * (bw + gap) if n > 1 else W / 2 - bw / 2
        g = eo((tl - 0.4 - 0.5 * i) / 1.4)
        hgt = (base - top) * it["value"] / mx * g
        col = tuple(it.get("color", GOLD if i == 0 else WATER))
        d.rounded_rectangle([x, base - hgt, x + bw, base], radius=10, fill=col + (235,))
        d.text((x + bw / 2, base + 50), it["label"], font=F("SemiBold", 40), fill=(235, 238, 245, 255), anchor="mm")
        if g > 0.05:
            d.text((x + bw / 2, base - hgt - 42), it.get("text", f"{it['value']:,}"), font=F("Bold", 44), fill=(245, 246, 250, int(255 * min(1, g * 2))), anchor="mm")
    d.line([(300, base), (W - 300, base)], fill=(255, 255, 255, 120), width=2)
    return im


def r_timeline(sh, u, tl, t, fi):
    im = card_bg(sh, u)
    d = ImageDraw.Draw(im, "RGBA")
    ev = sh["events"]; n = len(ev)
    y = H / 2 + 20; x0, x1 = 220, W - 220
    lw = (x1 - x0) * eo(tl / 1.2)
    d.rectangle([x0, y - 2, x0 + lw, y + 2], fill=(255, 255, 255, 170))
    for i, e in enumerate(ev):
        ta = sh["_ltime"](e.get("at", i))
        a = ease((tl - ta) / 0.6)
        if a <= 0: continue
        x = x0 + (x1 - x0) * (i + 0.5) / n
        up = i % 2 == 0
        d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=GOLD + (int(255 * a),), outline=(255, 255, 255, int(255 * a)), width=3)
        yy = y - 70 if up else y + 70
        d.line([(x, y + (-14 if up else 14)), (x, yy + (30 if up else -30))], fill=(255, 255, 255, int(150 * a)), width=2)
        d.text((x, yy - (20 if up else -20) - 8 * (1 - eo((tl - ta) / 0.6))), e["year"], font=F("Bold", 42), fill=GOLD + (int(255 * a),), anchor="mm")
        fs = F("Regular", 28)
        lines = wrap(d, e["text"], fs, 330)
        for j, ln in enumerate(lines):
            yt = (yy - 80 - (len(lines) - 1 - j) * 36) if up else (yy + 64 + j * 36)
            d.text((x, yt), ln, font=fs, fill=(230, 234, 242, int(255 * a)), anchor="mm")
    return im


def r_quote(sh, u, tl, t, fi):
    im = card_bg(sh, u)
    d = ImageDraw.Draw(im, "RGBA"); a = ease(tl / 0.8)
    f = F("serif", 72)
    lines = wrap(d, "“" + sh["text"] + "”", f, W - 420)
    y0 = H / 2 - (len(lines) - 1) * 46 - 40
    for j, ln in enumerate(lines):
        d.text((W / 2, y0 + j * 92), ln, font=f, fill=(245, 246, 250, int(255 * ease((tl - 0.1 * j) / 0.8))), anchor="mm")
    if sh.get("who"):
        d.text((W / 2, y0 + len(lines) * 92 + 40), "— " + sh["who"], font=F("Medium", 38), fill=GOLD + (int(255 * ease((tl - 0.8) / 0.8)),), anchor="mm")
    return im


def r_intro(sh, u, tl, t, fi):
    im = Image.new("RGB", (W, H), (5, 10, 20)); d = ImageDraw.Draw(im, "RGBA")
    g = eo((tl - 0.3) / 1.0); a = ease((tl - 0.7) / 0.8); out = 1 - ease((tl - (sh["_dur"] - 0.8)) / 0.8)
    bh = 90 * g
    d.rectangle([W / 2 - 330, H / 2 - bh / 2, W / 2 - 320, H / 2 + bh / 2], fill=GOLD + (int(255 * out),))
    d.text((W / 2 - 290, H / 2), "MAPS AND MUSINGS", font=F("Bold", 64), fill=(245, 246, 250, int(255 * a * out)), anchor="lm")
    d.text((W / 2 - 288, H / 2 + 70), "Pakistan and the world, explained", font=F("Light", 30), fill=(200, 208, 222, int(255 * ease((tl - 1.3) / 0.8) * out)), anchor="lm")
    return im


def r_outro(sh, u, tl, t, fi):
    im = card_bg(sh, u, dark=0.7, blur=10); d = ImageDraw.Draw(im, "RGBA"); a = ease(tl / 0.8)
    d.text((W / 2, 250), "Thank you for watching", font=F("serif", 80), fill=(245, 246, 250, int(255 * a)), anchor="mm")
    # subscribe button
    bw, bh = 420, 96; bx, by = W / 2 - bw / 2, 360
    pop = eo((tl - 0.6) / 0.5)
    d.rounded_rectangle([bx, by, bx + bw * pop, by + bh], radius=14, fill=(204, 0, 0, int(255 * a)))
    if pop > 0.9: d.text((W / 2, by + bh / 2), "SUBSCRIBE", font=F("Bold", 44), fill=(255, 255, 255, 255), anchor="mm")
    d.text((W / 2, 520), "Maps and Musings  ·  @aveiwfromhere", font=F("Medium", 36), fill=GOLD + (int(255 * a),), anchor="mm")
    # leave space for YouTube end-screen elements (two boxes)
    for k, x in enumerate((W / 2 - 560, W / 2 + 80)):
        d.rounded_rectangle([x, 620, x + 480, 890], radius=16, outline=(255, 255, 255, int(70 * a)), width=2)
    cr = sh.get("_credits", [])
    for j, c in enumerate(cr[:3]):
        d.text((W / 2, 950 + j * 34), c, font=F("Regular", 24), fill=(190, 198, 212, int(255 * ease((tl - 1.2) / 1))), anchor="mm")
    return im


FLIGHT = {}
def r_flight(sh, u, tl, t, fi):
    fdir = sh["_fdir"]; k = min(sh["_fn"] - 1, max(0, int(tl * FPS)))
    im = Image.open(os.path.join(fdir, f"{k:05d}.jpg")).convert("RGB")
    return grade(im, sat=0.98, contrast=1.04)


RENDER = dict(sat=r_sat, title=r_title, stat=r_stat, bars=r_bars, timeline=r_timeline, quote=r_quote, intro=r_intro, outro=r_outro, flight=r_flight)


# ---------------------------------------------------------------- audio
def narrate(spec, tmp):
    import hashlib, pickle
    key = hashlib.sha1(json.dumps([spec.get("voice"), spec.get("speed")] + [[sh.get("lines"), sh.get("pre"), sh.get("post"), sh.get("dur")] for sh in spec["shots"]]).encode()).hexdigest()[:16]
    cdir = "/home/claude/flighttest/narr_cache"; os.makedirs(cdir, exist_ok=True)
    cp = os.path.join(cdir, key)
    if os.path.exists(cp + ".pkl"):
        timing, t = pickle.load(open(cp + ".pkl", "rb")); shutil.copy(cp + ".wav", os.path.join(tmp, "voice.wav"))
        return os.path.join(tmp, "voice.wav"), timing, t
    voice, timing, t = _narrate(spec, tmp)
    shutil.copy(voice, cp + ".wav"); pickle.dump((timing, t), open(cp + ".pkl", "wb"))
    return voice, timing, t


def _narrate(spec, tmp):
    sys.path.insert(0, HERE)
    from reel import ensure_tts
    ensure_tts()
    from kokoro_onnx import Kokoro
    import soundfile as sf
    TTS = "/home/claude/tts"
    k = Kokoro(f"{TTS}/model.onnx", f"{TTS}/voices.npz")
    vname = spec.get("voice", "bm_george")
    v = (k.get_voice_style("bm_george") * 0.75 + k.get_voice_style("bm_fable") * 0.25) if vname == "blend" else vname
    sr = 24000; out = []; t = 0.0
    timing = []  # per shot: (start, end, [(a,b,text)])
    for sh in spec["shots"]:
        s0 = t
        pre = sh.get("pre", 0.0); out.append(np.zeros(int(sr * pre), np.float32)); t += pre
        lt = []
        for i, ln in enumerate(sh.get("lines", [])):
            a, sr = k.create(ln, voice=v, speed=spec.get("speed", 0.92), lang="en-gb")
            nz = np.where(np.abs(a) > 0.004)[0]
            if len(nz): a = a[max(0, nz[0] - int(0.03 * sr)): nz[-1] + int(0.08 * sr)]
            lt.append((t, t + len(a) / sr, ln)); out.append(a.astype(np.float32)); t += len(a) / sr
            gap = 0.5 if i < len(sh["lines"]) - 1 else 0.7
            out.append(np.zeros(int(sr * gap), np.float32)); t += gap
        if not sh.get("lines"):
            dd = sh.get("dur", 4.0); out.append(np.zeros(int(sr * dd), np.float32)); t += dd
        post = sh.get("post", 0.0); out.append(np.zeros(int(sr * post), np.float32)); t += post
        timing.append((s0, t, lt))
    raw = os.path.join(tmp, "raw.wav"); sf.write(raw, np.concatenate(out), sr)
    voice = os.path.join(tmp, "voice.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-af",
                    "highpass=f=70,lowshelf=f=140:g=2,equalizer=f=320:t=q:w=1.2:g=-1.5,equalizer=f=3000:t=q:w=1.4:g=2.2,"
                    "deesser=i=0.4,acompressor=threshold=-22dB:ratio=2.5:attack=10:release=150:makeup=2,"
                    "aecho=0.85:0.6:22|41:0.06|0.035,loudnorm=I=-16:TP=-1.5:LRA=7", "-ar", "48000", "-ac", "2", voice], check=True)
    return voice, timing, t


def score(dur, path, sting_at=0.0):
    """Cinematic ambient bed: drone, slow pads, sparse felt-piano notes, soft wind. Royalty-free (synthesised here)."""
    import soundfile as sf
    sr = 48000; n = int(dur * sr); tt = np.arange(n) / sr
    rng = np.random.default_rng(3)
    note = lambda m: 440 * 2 ** ((m - 69) / 12)
    prog = [[50, 57, 62, 65, 69], [46, 53, 58, 62, 65], [41, 48, 53, 57, 60], [48, 55, 60, 64, 67]]  # Dm Bb F C
    L = np.zeros(n, np.float32); Rr = np.zeros(n, np.float32)
    cl = 8.0
    for ci in range(int(dur / cl) + 2):
        st = int(ci * cl * sr); en = min(n, int((ci + 1) * cl * sr + 3 * sr))
        if st >= n: break
        s = tt[st:en] - tt[st]; env = np.minimum(1, s / 3.0) * np.clip(1 - (s - cl) / 3.0, 0, 1)
        for j, m in enumerate(prog[ci % 4][:4]):
            f = note(m - 12 if j == 0 else m)
            ph = rng.uniform(0, 6.28)
            w = (np.sin(2 * np.pi * f * s + ph) + 0.35 * np.sin(2 * np.pi * f * 2.002 * s) + 0.12 * np.sin(2 * np.pi * f * 3.01 * s))
            trem = 0.85 + 0.15 * np.sin(2 * np.pi * (0.11 + 0.03 * j) * s + ph)
            pan = 0.3 + 0.4 * (j / 3)
            L[st:en] += (w * env * trem * (1 - pan) * 0.16).astype(np.float32); Rr[st:en] += (w * env * trem * pan * 0.16).astype(np.float32)
        # sparse piano notes
        for k in range(3):
            if rng.random() < 0.75:
                m = rng.choice(prog[ci % 4][1:]) + 12
                t0 = st + int((0.8 + k * 2.4 + rng.uniform(0, 0.6)) * sr)
                if t0 >= n: continue
                ln = min(n - t0, int(4.0 * sr)); s2 = np.arange(ln) / sr; f = note(m)
                tone = (np.sin(2 * np.pi * f * s2) + 0.4 * np.sin(2 * np.pi * 2 * f * s2) * np.exp(-s2 * 3) + 0.15 * np.sin(2 * np.pi * 3 * f * s2) * np.exp(-s2 * 5))
                env2 = np.exp(-s2 * 1.1) * np.minimum(1, s2 / 0.004)
                pn = rng.uniform(0.3, 0.7)
                L[t0:t0 + ln] += (tone * env2 * 0.10 * (1 - pn)).astype(np.float32); Rr[t0:t0 + ln] += (tone * env2 * 0.10 * pn).astype(np.float32)
    # drone
    dr = 0.2 * np.sin(2 * np.pi * note(38) * tt) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.05 * tt))
    L += dr.astype(np.float32); Rr += dr.astype(np.float32)
    # wind: filtered noise
    from scipy.signal import lfilter
    wn = rng.normal(0, 1, n).astype(np.float32)
    wn = lfilter([0.02], [1, -0.98], wn).astype(np.float32) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * tt)).astype(np.float32)
    L += wn * 0.5; Rr += np.roll(wn, 1200) * 0.5
    # reverb-ish
    for dl, g in ((0.13, 0.45), (0.29, 0.3), (0.47, 0.2), (0.71, 0.12)):
        k_ = int(dl * sr); L[k_:] += g * Rr[:-k_] * 0.6; Rr[k_:] += g * L[:-k_] * 0.6
    mx = max(np.abs(L).max(), np.abs(Rr).max()) + 1e-6
    fade = (np.minimum(1, tt / 3) * np.clip((dur - tt) / 4, 0, 1)).astype(np.float32)
    L = L / mx * fade * 0.5; Rr = Rr / mx * fade * 0.5
    # whoosh for the intro sting
    if sting_at is not None:
        ln = int(1.6 * sr); s3 = np.arange(ln) / sr
        wsh = lfilter([0.08], [1, -0.92], rng.normal(0, 1, ln)).astype(np.float32) * np.sin(np.pi * np.clip(s3 / 1.6, 0, 1)) ** 2 * 0.6
        a0 = int(sting_at * sr); L[a0:a0 + ln] += wsh[:n - a0]; Rr[a0:a0 + ln] += wsh[:n - a0]
    sf.write(path, np.stack([L, Rr], 1), sr)


# ---------------------------------------------------------------- main
def srt_time(x):
    h = int(x // 3600); m = int(x % 3600 // 60); s = x % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"


def write_srt(timing, path):
    k = 1; out = []
    for s0, s1, lt in timing:
        for a, b, txt in lt:
            words = txt.split(); parts, cur = [], []
            for w in words:
                cur.append(w)
                if len(" ".join(cur)) > 40: parts.append(" ".join(cur)); cur = []
            if cur:
                if parts and len(" ".join(cur)) < 12: parts[-1] += " " + " ".join(cur)
                else: parts.append(" ".join(cur))
            tot = sum(len(p) for p in parts); t0 = a
            for p in parts:
                dd = (b - a) * len(p) / tot
                out.append(f"{k}\n{srt_time(t0)} --> {srt_time(t0 + dd)}\n{p}\n"); k += 1; t0 += dd
    open(path, "w").write("\n".join(out))


PLAN = None
def frame_job(args):
    fi, path = args
    t = fi / FPS
    shots = PLAN["shots"]
    i = max(j for j, s in enumerate(shots) if s["_t0"] <= t + 1e-9)
    img = render_shot(i, t, fi)
    nx = i + 1
    if nx < len(shots):
        xf = shots[nx].get("xfade", 0.7)
        if t > shots[nx]["_t0"] - xf:
            img = Image.blend(img, render_shot(nx, t, fi), eio((t - (shots[nx]["_t0"] - xf)) / xf))
    img = chrome(img, t, i)
    img = grain(img, fi)
    img.save(path, quality=93)


def render_shot(i, t, fi):
    sh = PLAN["shots"][i]
    tl = t - sh["_t0"]; u = max(0.0, min(1.0, tl / max(0.1, sh["_dur"])))
    return RENDER[sh["type"]](sh, u, max(0.0, tl), t, fi)


def chrome(img, t, i):
    sh = PLAN["shots"][i]; d = ImageDraw.Draw(img, "RGBA")
    if sh["type"] not in ("intro", "outro"):
        tot = PLAN["total"]
        if sh["type"] != "title":
            d.rectangle([60, 56, 66, 86], fill=GOLD + (210,))
            d.text((80, 71), "MAPS AND MUSINGS", font=F("Bold", 22), fill=(255, 255, 255, 190), anchor="lm", stroke_width=2, stroke_fill=(0, 0, 0, 70))
            ch = sh.get("_chap")
            if ch: d.text((W - 60, 71), ch.upper(), font=F("Medium", 20), fill=(255, 255, 255, 150), anchor="rm", stroke_width=2, stroke_fill=(0, 0, 0, 70))
    # lower third
    if sh.get("lower"):
        tl = t - sh["_t0"]; dur = sh.get("lower_dur", 6.0)
        a = eo((tl - 0.6) / 0.6) * (1 - ease((tl - 0.6 - dur) / 0.5))
        if a > 0.01:
            ttl, sub = sh["lower"][0], (sh["lower"][1] if len(sh["lower"]) > 1 else "")
            f1, f2 = F("Bold", 44), F("Regular", 28)
            tw = max(d.textlength(ttl, font=f1), d.textlength(sub, font=f2)) + 70
            x = 90 - (1 - a) * 60; y = H - 230
            d.rectangle([x, y, x + tw * a, y + 118], fill=(8, 16, 30, int(215 * a)))
            d.rectangle([x, y, x + 7, y + 118], fill=GOLD + (int(255 * a),))
            d.text((x + 34, y + 42), ttl, font=f1, fill=(245, 246, 250, int(255 * a)), anchor="lm")
            if sub: d.text((x + 34, y + 88), sub, font=f2, fill=(205, 212, 226, int(255 * a)), anchor="lm")
    return img


def make(spec, out, only=None, stills=None):
    global PLAN
    tmp = tempfile.mkdtemp(dir="/home/claude/flighttest")
    voice, timing, total = narrate(spec, tmp)
    shots = spec["shots"]; chap = None; chapters = []
    for sh, (s0, s1, lt) in zip(shots, timing):
        sh["_t0"] = s0; sh["_dur"] = s1 - s0; sh["_lines"] = [(a - s0, b - s0) for a, b, _ in lt]
        if sh.get("chapter"): chap = sh["chapter"] + (" · " + sh["text"] if sh["type"] == "title" else ""); chapters.append((s0, sh.get("chapter_name", sh.get("text", sh["chapter"]))))
        sh["_chap"] = chap
        sh["_credits"] = spec.get("credits", [])
    PLAN = {"shots": shots, "total": total}
    def mk_lt(sh):
        L = sh["_lines"]
        return lambda i: (L[min(int(i), len(L) - 1)][0] + (i - int(i)) * (L[min(int(i), len(L) - 1)][1] - L[min(int(i), len(L) - 1)][0])) if L else 0.0
    for sh in shots: sh["_ltime"] = mk_lt(sh)
    # 3D flights
    import flight as FL
    jobs = []
    for i, sh in enumerate(shots):
        if sh["type"] != "flight" or (only and i not in only): continue
        fs = json.load(open(os.path.join(ROOT, sh["spec"])))
        fs["seconds"] = sh["_dur"] + 1.0
        fs.update({"out": [W, H], "render": sh.get("render", [800, 450]), "horizon": fs.get("horizon16", 0.30), "label_size": 34,
                   "label_font": INTER.format("SemiBold")})
        fs_tc = TILECACHE
        if "pitch16" in fs: fs["pitch"] = fs["pitch16"]
        labs = []
        for lb in fs.get("labels", []):
            if "line" in lb:
                a0 = sh["_ltime"](lb["line"]); a1 = sh["_ltime"](lb.get("to_line", lb["line"]) + 0.999) + 0.8
                labs.append(dict(lb, t0=max(0, a0), t1=a1))
            else: labs.append(lb)
        fs["labels"] = labs; fs.pop("only", None)
        import hashlib
        fs.pop("tilecache", None)
        fkey = hashlib.sha1(json.dumps(fs, sort_keys=True).encode()).hexdigest()[:16]
        fdir = os.path.join("/home/claude/flighttest/fcache", fkey); os.makedirs(fdir, exist_ok=True); sh["_fdir"] = fdir
        n = int(fs["seconds"] * 25)
        if not stills and len(os.listdir(fdir)) >= n:
            sh["_fn"] = n; continue
        if stills:
            fs["only"] = sorted({min(n - 1, max(0, int(fr * sh["_dur"] * FPS) + dd)) for (si, fr) in stills if si == i for dd in (-1, 0, 1)})
            if not fs["only"]: continue
            FL.render(fs, fdir)
        else:
            todo = [k for k in range(n) if not os.path.exists(os.path.join(fdir, f"{k:05d}.jpg"))]
            if todo: jobs.append((dict(fs, only=todo, tilecache=fs_tc), fdir))  # one process: 3D terrain needs ~3 GB RAM
        sh["_fn"] = n
    for job in jobs:
        FL.render(*job)
    if stills:
        res = []
        for si, fr in stills:
            sh = shots[si]; t = sh["_t0"] + fr * sh["_dur"]; fi = int(t * FPS)
            if sh["type"] == "flight":
                sh["_fn"] = int((sh["_dur"] + 1) * 25)
            p = f"{out}_still_{si}_{int(fr*100)}.jpg"; frame_job((fi, p)); res.append(p)
        print("\n".join(res)); return
    # frames
    fr_dir = os.environ.get("DOC_FRAMES") or os.path.join(tmp, "f"); os.makedirs(fr_dir, exist_ok=True)  # DOC_FRAMES resumes a render
    dur = total
    n = int(dur * FPS)
    rng = range(n)
    if only:
        sel = [fi for fi in rng if any(shots[k]["_t0"] <= fi / FPS < shots[k]["_t0"] + shots[k]["_dur"] for k in only)]
    else:
        sel = list(rng)
    with Pool(2) as p:
        todo = [(fi, os.path.join(fr_dir, f"{j:06d}.jpg")) for j, fi in enumerate(sel)]
        todo = [a for a in todo if not os.path.exists(a[1])]
        for k, _ in enumerate(p.imap(frame_job, todo, chunksize=8)):
            if k % 500 == 0: print("frame", k, "/", len(sel), flush=True)
    t_off = sel[0] / FPS; t_len = len(sel) / FPS
    score(dur, os.path.join(tmp, "music.wav"), sting_at=0.2 if shots[0]["type"] == "intro" else None)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(fr_dir, "%06d.jpg"),
                    "-ss", f"{t_off:.3f}", "-t", f"{t_len:.3f}", "-i", voice, "-ss", f"{t_off:.3f}", "-t", f"{t_len:.3f}", "-i", os.path.join(tmp, "music.wav"),
                    "-filter_complex",
                    "[2:a]volume=0.55[m];[m][1:a]sidechaincompress=threshold=0.03:ratio=6:attack=40:release=600[md];"
                    "[1:a][md]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.93[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "medium", "-profile:v", "high", "-pix_fmt", "yuv420p",
                    "-crf", "19", "-g", "50", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out], check=True)
    base = os.path.splitext(out)[0]
    write_srt(timing, base + ".srt")
    desc = [spec.get("description", ""), "", "Chapters:"]
    for s0, name in chapters:
        desc.append(f"{int(s0 // 60)}:{int(s0 % 60):02d} {name}")
    if chapters and chapters[0][0] > 0: desc.insert(3, "0:00 Intro")
    desc += ["", "Sources:"] + [f"- {s['title']}: {s['url']}" for s in spec.get("sources", [])]
    desc += ["", "Credits:"] + spec.get("credits", [])
    open(base + ".txt", "w").write("\n".join(desc))
    shutil.rmtree(tmp, ignore_errors=True)
    print("ok", out, round(dur, 1), "s")


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1])); out = sys.argv[2]
    only = None; stills = None
    if "--only" in sys.argv: only = [int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")]
    if "--still" in sys.argv:
        stills = []
        for a in sys.argv[sys.argv.index("--still") + 1:]:
            if a.startswith("--"): break
            si, fr = a.split(":"); stills.append((int(si), float(fr)))
    make(spec, out, only, stills)
