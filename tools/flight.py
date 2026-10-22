#!/usr/bin/env python3
"""Render a 3D flythrough over real terrain (AWS Terrain Tiles + Sentinel-2 cloudless imagery).

Usage: python3 tools/flight.py spec.json out_dir
spec: {"bbox": [lat_min, lat_max, lon_min, lon_max], "dem_zoom": 13, "img_zoom": 14,
       "seconds": 60, "waypoints": [[lat, lon, altitude_m_above_ground, look_ahead_deg_offset], ...],
       "labels": [{"lat":..,"lon":..,"text":"ATTABAD LAKE","t0":5,"t1":14}], "exaggerate": 1.0}
Writes out_dir/00000.jpg ... at 1080x1920, 25 fps.
Credits: Terrain: AWS Terrain Tiles (SRTM et al.) · Imagery: Sentinel-2 cloudless 2016 by EOX (CC BY 4.0)
"""
import io, json, math, os, sys, concurrent.futures as cf
import numpy as np, requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates, gaussian_filter

OW, OH, FPS = 1080, 1920, 25
RW, RH = 540, 960
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


def tile(lat, lon, z):
    n = 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def fetch(z, kind, bbox, cache):
    lat0, lat1, lon0, lon1 = bbox
    fx0, fy0 = tile(lat1, lon0, z); fx1, fy1 = tile(lat0, lon1, z)
    x0, y0, x1, y1 = int(fx0), int(fy0), int(fx1), int(fy1)
    S = requests.Session(); S.headers["User-Agent"] = "MapsAndMusings/1.0"
    def f(xy):
        x, y = xy
        p = os.path.join(cache, f"{kind}_{z}_{x}_{y}")
        if not os.path.exists(p):
            u = (f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png" if kind == "dem"
                 else f"https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/{z}/{y}/{x}.jpg")
            r = S.get(u, timeout=60); r.raise_for_status(); open(p, "wb").write(r.content)
        return xy, Image.open(p).convert("RGB")
    out = Image.new("RGB", (256 * (x1 - x0 + 1), 256 * (y1 - y0 + 1)))
    with cf.ThreadPoolExecutor(8) as ex:
        for (x, y), im in ex.map(f, [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]):
            out.paste(im, ((x - x0) * 256, (y - y0) * 256))
    return out, (x0, y0)


def build(spec, cache):
    os.makedirs(cache, exist_ok=True)
    bbox = spec["bbox"]; zi = spec.get("img_zoom", 14); zd = spec.get("dem_zoom", 13)
    img, (ix0, iy0) = fetch(zi, "img", bbox, cache)
    dem, (dx0, dy0) = fetch(zd, "dem", bbox, cache)
    a = np.asarray(dem).astype(np.float32)
    hdem = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
    tex = np.asarray(img).astype(np.float32)
    Hh, Ww = tex.shape[:2]
    # resample DEM onto image grid
    sc = 2 ** (zi - zd)
    gy, gx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    dy = (iy0 * 256 + gy) / sc - dy0 * 256; dx = (ix0 * 256 + gx) / sc - dx0 * 256
    h = map_coordinates(hdem, [dy, dx], order=1, mode="nearest")
    h = gaussian_filter(h, 1.2)
    lat_c = (bbox[0] + bbox[1]) / 2
    m_per_px = 40075016.686 * math.cos(math.radians(lat_c)) / (256 * 2 ** zi)
    hp = h / m_per_px * spec.get("exaggerate", 1.0)  # heights in pixel units
    # hillshade baked into texture
    gyh, gxh = np.gradient(hp)
    nx, ny, nz = -gxh, -gyh, np.ones_like(hp)
    nrm = np.sqrt(nx * nx + ny * ny + nz * nz); nx /= nrm; ny /= nrm; nz /= nrm
    sun = np.array([-0.5, -0.6, 0.62]); sun /= np.linalg.norm(sun)
    shade = np.clip(nx * sun[0] + ny * sun[1] + nz * sun[2], 0, 1)
    shade = 0.72 + 0.5 * shade
    tex = np.clip(tex * shade[..., None] * 1.05, 0, 255)
    # gentle grade: more contrast / saturation
    mean = tex.mean(axis=2, keepdims=True)
    tex = np.clip(mean + (tex - mean) * 1.18, 0, 255)
    def to_px(lat, lon):
        fx, fy = tile(lat, lon, zi)
        return (fx - ix0) * 256, (fy - iy0) * 256
    return hp, tex, m_per_px, to_px


def catmull(pts, n):
    pts = np.array(pts, dtype=np.float64)
    P = np.vstack([pts[0], pts, pts[-1]])
    out = []
    segs = len(pts) - 1
    for i in range(n):
        t = i / (n - 1) * segs; k = min(int(t), segs - 1); u = t - k
        p0, p1, p2, p3 = P[k], P[k + 1], P[k + 2], P[k + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    return np.array(out)


def render(spec, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    hp, tex, mpp, to_px = build(spec, os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "tilecache"))
    Hh, Ww = hp.shape
    n = int(spec.get("seconds", 60) * FPS)
    wps = []
    for lat, lon, alt, *rest in spec["waypoints"]:
        x, y = to_px(lat, lon); wps.append([x, y, alt / mpp])
    path = catmull(wps, n)
    # smooth heading along path
    look = np.roll(path, -int(FPS * 1.5), axis=0); look[-int(FPS * 1.5):] = path[-1] + (path[-1] - path[-2]) * np.arange(1, int(FPS * 1.5) + 1)[:, None]
    fov = math.radians(spec.get("fov", 62)); focal = (RW / 2) / math.tan(fov / 2)
    D = 900; near, far = 4.0, spec.get("far_px", 3200.0)
    dist = near * (far / near) ** (np.arange(D) / (D - 1))
    ang = np.linspace(-fov / 2, fov / 2, RW)
    sky_top = np.array([62, 104, 160], np.float32); sky_hor = np.array([196, 214, 232], np.float32)
    rows = np.arange(RH)
    font = ImageFont.truetype(BOLD, 40)
    only = spec.get("only")
    for fi in range(n):
        if only and fi not in only: continue
        cx, cy, calt = path[fi]
        hx, hy = look[fi][0] - cx, look[fi][1] - cy
        head = math.atan2(hy, hx)
        ground = map_coordinates(hp, [[cy], [cx]], order=1, mode="nearest")[0]
        camz = ground + calt
        # keep camera above terrain directly ahead
        dirs = head + ang
        X = cx + dist[:, None] * np.cos(dirs)[None, :]
        Y = cy + dist[:, None] * np.sin(dirs)[None, :]
        inside = (X >= 0) & (X < Ww - 1) & (Y >= 0) & (Y < Hh - 1)
        Z = map_coordinates(hp, [Y.ravel(), X.ravel()], order=1, mode="nearest").reshape(D, RW)
        Z = np.where(inside, Z, -1e6)
        pitch = spec.get("pitch", -0.18)
        horizon = RH * 0.40 - pitch * focal
        perp = dist[:, None] * np.cos(ang)[None, :]
        sy = horizon - (Z - camz) / perp * focal
        cm = np.minimum.accumulate(sy, axis=0)
        img = np.empty((RH, RW, 3), np.float32)
        # sky
        t = np.clip(rows / max(1, horizon), 0, 1)[:, None]
        skycol = sky_top * (1 - t) + sky_hor * t
        for c in range(RW):
            col = cm[:, c]
            # first depth where cm <= row  (cm is non-increasing)
            d_idx = np.searchsorted(-col, -rows, side="left")
            hit = d_idx < D
            di = np.minimum(d_idx, D - 1)
            xs = X[di, c]; ys = Y[di, c]
            colr = np.stack([map_coordinates(tex[..., k], [ys, xs], order=1, mode="nearest") for k in range(3)], 1)
            fogk = np.clip((dist[di] / far) ** 1.4 * 0.85, 0, 0.85)[:, None]
            colr = colr * (1 - fogk) + sky_hor * fogk
            img[:, c] = np.where(hit[:, None], colr, skycol)
        fr = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((OW, OH), Image.LANCZOS)
        fr = fr.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
        # labels (projected)
        d = ImageDraw.Draw(fr)
        tsec = fi / FPS
        for lb in spec.get("labels", []):
            if not (lb["t0"] <= tsec <= lb["t1"]): continue
            lx, ly = to_px(lb["lat"], lb["lon"])
            vx, vy = lx - cx, ly - cy
            a_rel = math.atan2(vy, vx) - head; a_rel = (a_rel + math.pi) % (2 * math.pi) - math.pi
            if abs(a_rel) > fov / 2 * 0.95: continue
            rng = math.hypot(vx, vy)
            if rng < 5: continue
            gz = map_coordinates(hp, [[ly], [lx]], order=1, mode="nearest")[0]
            px = RW / 2 + math.tan(a_rel) * focal
            py = horizon - (gz - camz) / (rng * math.cos(a_rel)) * focal
            X2, Y2 = px * OW / RW, py * OH / RH
            al = min(1, (tsec - lb["t0"]) / 0.6, (lb["t1"] - tsec) / 0.6)
            if not (40 < X2 < OW - 40 and 200 < Y2 < OH - 300): continue
            txt = lb["text"].upper(); tw = d.textlength(txt, font=font)
            top = Y2 - 150
            d.line([(X2, Y2), (X2, top + 60)], fill=(232, 176, 74), width=4)
            d.ellipse([X2 - 10, Y2 - 10, X2 + 10, Y2 + 10], fill=(232, 176, 74), outline=(255, 255, 255), width=3)
            bx = min(max(X2 - tw / 2 - 22, 20), OW - tw - 64)
            d.rounded_rectangle([bx, top, bx + tw + 44, top + 62], radius=12, fill=(0, 0, 0))
            d.text((bx + 22, top + 31), txt, font=font, fill=(int(245 * al), int(246 * al), int(250 * al)), anchor="lm")
        fr.save(os.path.join(out_dir, f"{fi:05d}.jpg"), quality=90)
        if fi % 100 == 0: print("frame", fi, "/", n, flush=True)
    print("done", n)


if __name__ == "__main__":
    render(json.load(open(sys.argv[1])), sys.argv[2])
