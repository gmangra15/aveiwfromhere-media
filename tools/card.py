#!/usr/bin/env python3
"""Render a 1080x1350 news card for Maps and Musings (@aveiwfromhere).

Usage: python3 tools/card.py spec.json out.jpg

spec.json fields:
  kicker    short topic label, e.g. "PAKISTAN" or "WORLD"            (required)
  headline  6-12 word headline                                       (required)
  facts     list of 1-3 short fact lines                             (optional)
  source    e.g. "Source: Dawn, Reuters"                             (optional)
  photo     path or https://images.unsplash.com/... URL (optional background photo)
  credit    photo credit, e.g. "Photo: Name / Unsplash"             (needed if photo)
"""
import io, json, sys, textwrap, urllib.request
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

W, H = 1080, 1350
NAVY = (11, 22, 40); INK = (245, 246, 250); MUTED = (176, 186, 204); GOLD = (232, 176, 74)
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
REG = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SERIF_CANDIDATES = ["/usr/share/fonts/truetype/google-fonts/Lora-Variable.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"]


def font(path, size, bold=False):
    f = ImageFont.truetype(path, size)
    if bold and "Lora" in path:
        try: f.set_variation_by_name("Bold")
        except Exception: pass
    return f


def serif(size):
    for p in SERIF_CANDIDATES:
        try: return font(p, size, bold=True)
        except Exception: continue
    return font(BOLD, size)


def load_photo(src):
    if src.startswith("http"):
        if "images.unsplash.com" in src and "?" not in src:
            src += "?fm=jpg&q=90&w=1600"
        req = urllib.request.Request(src, headers={"User-Agent": "MapsAndMusings/1.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(src).convert("RGB")


def cover(im, w, h):
    r = max(w / im.width, h / im.height)
    im = im.resize((int(im.width * r) + 1, int(im.height * r) + 1), Image.LANCZOS)
    x, y = (im.width - w) // 2, (im.height - h) // 3
    return im.crop((x, y, x + w, y + h))


def wrap(d, text, f, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= maxw: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines


def render(spec, out):
    img = Image.new("RGB", (W, H), NAVY)
    photo_h = 0
    if spec.get("photo"):
        try:
            p = cover(load_photo(spec["photo"]), W, 760)
            p = ImageEnhance.Color(p).enhance(0.9)
            img.paste(p, (0, 0)); photo_h = 760
            # fade photo into navy
            fade = Image.new("L", (1, 760))
            for y in range(760):
                fade.putpixel((0, y), int(255 * max(0, min(1, (y - 380) / 380)) ** 1.4))
            img.paste(Image.new("RGB", (W, 760), NAVY), (0, 0), fade.resize((W, 760)))
            top = Image.new("L", (1, 760))
            for y in range(760):
                top.putpixel((0, y), int(150 * max(0, 1 - y / 220)))
            img.paste(Image.new("RGB", (W, 760), (0, 0, 0)), (0, 0), top.resize((W, 760)))
        except Exception as e:
            print("photo failed, using plain background:", e, file=sys.stderr)
            photo_h = 0
    d = ImageDraw.Draw(img)
    if not photo_h:
        # subtle grid pattern for plain cards
        for x in range(0, W, 54): d.line([(x, 0), (x, H)], fill=(16, 29, 50))
        for y in range(0, H, 54): d.line([(0, y), (W, y)], fill=(16, 29, 50))
    M = 80
    # brand bar
    d.rectangle([M, 70, M + 8, 110], fill=GOLD)
    d.text((M + 26, 90), "MAPS AND MUSINGS", font=font(BOLD, 30), fill=INK, anchor="lm")
    hs = 88
    while True:
        hf = serif(hs); lines = wrap(d, spec["headline"], hf, W - 2 * M)
        if len(lines) <= 3 or hs <= 56: break
        hs -= 6
    ff = font(REG, 38)
    block = 90 + len(lines) * int(hs * 1.18) + 26
    for fact in spec.get("facts", [])[:3]:
        block += 50 * len(wrap(d, fact, ff, W - 2 * M - 40)) + 18
    y = (photo_h - 90) if photo_h else max(200, (H - 130 - 150 - block) // 2 + 150)
    kicker = spec["kicker"].upper()
    kf = font(BOLD, 30); kw = d.textlength(kicker, font=kf)
    d.rounded_rectangle([M, y, M + kw + 40, y + 52], radius=8, fill=GOLD)
    d.text((M + 20, y + 26), kicker, font=kf, fill=NAVY, anchor="lm")
    y += 90
    for ln in lines:
        d.text((M, y), ln, font=hf, fill=INK); y += int(hs * 1.18)
    y += 26
    for fact in spec.get("facts", [])[:3]:
        fl = wrap(d, fact, ff, W - 2 * M - 40)
        d.ellipse([M, y + 16, M + 14, y + 30], fill=GOLD)
        for ln in fl:
            d.text((M + 40, y), ln, font=ff, fill=MUTED); y += 50
        y += 18
    # footer
    d.line([(M, H - 130), (W - M, H - 130)], fill=(40, 56, 84), width=2)
    if spec.get("source"):
        d.text((M, H - 92), spec["source"], font=font(REG, 26), fill=MUTED, anchor="lm")
    d.text((W - M, H - 92), "@aveiwfromhere", font=font(BOLD, 28), fill=GOLD, anchor="rm")
    if photo_h and spec.get("credit"):
        d.text((M, H - 52), spec["credit"], font=font(REG, 22), fill=(130, 140, 160), anchor="lm")
    img.save(out, quality=92)


if __name__ == "__main__":
    render(json.load(open(sys.argv[1])), sys.argv[2])
    print("ok", sys.argv[2])
