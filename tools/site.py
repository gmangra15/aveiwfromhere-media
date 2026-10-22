#!/usr/bin/env python3
"""Build the Maps and Musings blog into docs/ (served by GitHub Pages from main /docs).

Articles live in articles/<slug>.md with a JSON header between '---' lines:

---
{"title": "...", "date": "2026-10-22", "kicker": "World", "summary": "...",
 "image": "cards/2026-10-22-00-france-curfews.jpg", "image_credit": "...",
 "sources": [{"title": "Dawn", "url": "https://..."}]}
---
Markdown body (## headings, paragraphs, lists)...

Usage: python3 tools/site.py
"""
import html, json, os, shutil, glob, datetime
import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs")
BASE = "https://gmangra15.github.io/aveiwfromhere-media/"
SITE = "Maps and Musings"
TAGLINE = "Pakistan and the world, explained."

CSS = """
:root{--bg:#f7f5f0;--paper:#ffffff;--ink:#14161c;--muted:#5d6472;--line:#e3dfd6;--accent:#b9822a;--navy:#0b1628}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0b1220;--paper:#111a2c;--ink:#eef0f4;--muted:#a3adbf;--line:#22304a;--accent:#e8b04a}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.65 "Source Sans 3",system-ui,sans-serif}
a{color:var(--accent)}a:hover{opacity:.85}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px}
header.site{background:var(--navy);color:#fff}
header.site .wrap{display:flex;align-items:center;justify-content:space-between;gap:16px;padding-top:18px;padding-bottom:18px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:12px;color:#fff;text-decoration:none}
.brand .bar{width:6px;height:30px;background:#e8b04a;border-radius:2px}
.brand b{font:700 20px/1 "Source Sans 3",sans-serif;letter-spacing:.06em}
.tag{color:#c9d1df;font-size:15px}
.hero{padding:40px 0 10px}
.hero h1{font:700 clamp(30px,5vw,46px)/1.15 Lora,Georgia,serif;margin:0 0 8px}
.hero p{color:var(--muted);margin:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:24px;padding:28px 0 60px}
.card{background:var(--paper);border:1px solid var(--line);border-radius:14px;overflow:hidden;display:flex;flex-direction:column;text-decoration:none;color:inherit}
.card img{width:100%;aspect-ratio:4/5;object-fit:cover;display:block;background:#0b1628}
.card .b{padding:16px 18px 20px}
.kick{display:inline-block;font:700 12px/1 "Source Sans 3",sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:8px}
.card h2{font:700 21px/1.3 Lora,Georgia,serif;margin:0 0 8px}
.card p{margin:0;color:var(--muted);font-size:15px}
.meta{color:var(--muted);font-size:14px;margin-top:10px}
article{max-width:720px;margin:0 auto;padding:36px 20px 70px}
article h1{font:700 clamp(30px,5.5vw,44px)/1.15 Lora,Georgia,serif;margin:6px 0 12px}
article .dek{font-size:20px;color:var(--muted);margin:0 0 18px}
article figure{margin:22px 0}
article figure img{width:100%;border-radius:12px;display:block}
article figcaption{color:var(--muted);font-size:13px;margin-top:6px}
article h2{font:700 25px/1.3 Lora,Georgia,serif;margin:34px 0 8px}
article blockquote{margin:20px 0;padding:4px 18px;border-left:4px solid var(--accent);color:var(--muted)}
.sources{border-top:1px solid var(--line);margin-top:36px;padding-top:16px;font-size:15px}
.sources h3{font-size:14px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.follow{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-top:28px}
footer{border-top:1px solid var(--line);color:var(--muted);font-size:14px;padding:24px 0 40px}
"""

HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><meta name="description" content="{desc}">
<meta property="og:title" content="{ogt}"><meta property="og:description" content="{desc}"><meta property="og:type" content="{ogtype}">
<meta property="og:image" content="{ogi}"><meta property="og:url" content="{url}"><meta name="twitter:card" content="summary_large_image">
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Lora:wght@700&family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{rel}style.css"></head><body>
<header class="site"><div class="wrap"><a class="brand" href="{rel}index.html"><span class="bar"></span><b>MAPS AND MUSINGS</b></a><span class="tag">{tagline}</span></div></header>
"""
FOOT = """<footer><div class="wrap">© {year} Maps and Musings · Follow us on Facebook and Instagram: <b>@aveiwfromhere</b> · Facts checked against at least two reputable sources.</div></footer></body></html>"""


def parse(path):
    raw = open(path, encoding="utf-8").read()
    _, head, body = raw.split("---", 2)
    meta = json.loads(head)
    meta["slug"] = os.path.splitext(os.path.basename(path))[0]
    meta["html"] = markdown.markdown(body.strip(), extensions=["extra"])
    return meta


def img_for(meta):
    src = meta.get("image")
    if not src: return None
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    name = os.path.basename(src)
    shutil.copy(os.path.join(ROOT, src), os.path.join(OUT, "img", name))
    return "img/" + name


def nice_date(d):
    return datetime.date.fromisoformat(d).strftime("%d %B %Y").lstrip("0")


def build():
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "style.css"), "w").write(CSS)
    open(os.path.join(OUT, ".nojekyll"), "w").write("")
    arts = sorted((parse(p) for p in glob.glob(os.path.join(ROOT, "articles", "*.md"))), key=lambda m: (m["date"], m["slug"]), reverse=True)
    year = datetime.date.today().year
    for a in arts:
        a["img"] = img_for(a)
        url = BASE + a["slug"] + ".html"
        h = HEAD.format(title=html.escape(a["title"] + " | " + SITE), desc=html.escape(a["summary"]), ogt=html.escape(a["title"]),
                        ogtype="article", ogi=BASE + (a["img"] or ""), url=url, rel="", tagline=TAGLINE)
        fig = ""
        if a["img"]:
            fig = f'<figure><img src="{a["img"]}" alt="{html.escape(a["title"])}"><figcaption>{html.escape(a.get("image_credit", "Maps and Musings"))}</figcaption></figure>'
        src = "".join(f'<li><a href="{html.escape(s["url"])}" rel="noopener">{html.escape(s["title"])}</a></li>' for s in a.get("sources", []))
        page = h + f"""<article><span class="kick">{html.escape(a.get('kicker', 'News'))}</span><h1>{html.escape(a['title'])}</h1>
<p class="dek">{html.escape(a['summary'])}</p><div class="meta">By Maps and Musings · {nice_date(a['date'])}</div>{fig}
{a['html']}
<div class="follow">Watch the one-minute video version on our Facebook and Instagram: <b>@aveiwfromhere</b></div>
<div class="sources"><h3>Sources</h3><ul>{src}</ul></div></article>""" + FOOT.format(year=year)
        open(os.path.join(OUT, a["slug"] + ".html"), "w", encoding="utf-8").write(page)
    cards = "".join(f"""<a class="card" href="{a['slug']}.html">{f'<img src="{a["img"]}" alt="" loading="lazy">' if a['img'] else ''}
<div class="b"><span class="kick">{html.escape(a.get('kicker', 'News'))}</span><h2>{html.escape(a['title'])}</h2><p>{html.escape(a['summary'])}</p>
<div class="meta">{nice_date(a['date'])}</div></div></a>""" for a in arts)
    idx = HEAD.format(title=SITE + " | " + TAGLINE, desc="News and stories from Pakistan and the world, fact-checked and explained.", ogt=SITE,
                      ogtype="website", ogi=BASE + (arts[0]["img"] if arts and arts[0]["img"] else ""), url=BASE, rel="", tagline=TAGLINE)
    idx += f"""<main class="wrap"><section class="hero"><h1>Pakistan and the world, explained.</h1><p>Fact-checked stories on politics, the region, world news, nature and heritage.</p></section>
<section class="grid">{cards}</section></main>""" + FOOT.format(year=year)
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(idx)
    print("built", len(arts), "articles ->", OUT)


if __name__ == "__main__":
    build()
