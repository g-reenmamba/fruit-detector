"""
ui.py
-----
Look and feel for the Streamlit app: a pastel pixel-art world with the app
inside a retro OS window. Kept separate from app.py so the app logic stays
short and readable.

Fonts: Pixelify Sans (playful, rounded pixel display) for headings and
verdicts, VT323 (pixel terminal) for everything else.
"""

import base64
import html
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- colour tokens ---------------------------------------------------------
INK = "#2E2F5B"        # deep indigo: text, outlines, hard shadows
PAPER = "#F3F1FB"      # window body
FRAME = "#3D4C7E"      # window border
TITLEBAR = "#7189BF"   # window title bar
CHROME = "#CBD3B4"     # title-bar buttons
FRESH, FRESH_DARK = "#7BC98F", "#3E8A5A"
ROTTEN, ROTTEN_DARK = "#C07AA8", "#6E3A63"
UNSURE, UNSURE_DARK = "#F2D27A", "#9A7724"


# ---- pixel sprites: ASCII art -> crisp inline SVG -------------------------
def sprite(rows, palette, px=4, title=""):
    h, w = len(rows), max(len(r) for r in rows)
    rects = []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in palette:
                rects.append(f'<rect x="{x}" y="{y}" width="1" height="1" fill="{palette[ch]}"/>')
    label = f'role="img" aria-label="{title}"' if title else 'aria-hidden="true"'
    return (f'<svg {label} class="fd-sprite" width="{w * px}" height="{h * px}" viewBox="0 0 {w} {h}" '
            f'shape-rendering="crispEdges" xmlns="http://www.w3.org/2000/svg">{"".join(rects)}</svg>')


APPLE = sprite([
    "......s.....",
    ".....s.LL...",
    "..RRRsRRLL..",
    ".RhhRRRRRR..",
    "RhhRRRRRRRR.",
    "RhRRRRRRRRRd",
    "RRRRRRRRRRRd",
    "RRRRRRRRRRdd",
    ".RRRRRRRRRd.",
    ".RRRRRRRRdd.",
    "..RRdRRddd..",
    "...dd..dd...",
], {"R": "#E2546B", "h": "#FFB3BF", "d": "#A8324F", "s": "#6B4A3A", "L": "#6CC08B"}, px=5, title="pixel apple")

ICON_FRESH = sprite([
    "........GG",
    ".......GGG",
    "......GGG.",
    "GG...GGG..",
    "GGG.GGG...",
    ".GGGGG....",
    "..GGG.....",
], {"G": FRESH_DARK}, px=4)

ICON_ROTTEN = sprite([
    ".WWWWW.",
    "WWWWWWW",
    "WKKWKKW",
    "WKKWKKW",
    "WWWKWWW",
    ".WWWWW.",
    ".W.W.W.",
], {"W": "#FFFFFF", "K": ROTTEN_DARK}, px=4)

ICON_UNSURE = sprite([
    ".QQQQ.",
    "QQ..QQ",
    "....QQ",
    "...QQ.",
    "..QQ..",
    "......",
    "..QQ..",
], {"Q": UNSURE_DARK}, px=4)


def _background_uri():
    path = os.path.join(ROOT, "assets", "background.png")
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def css():
    bg = _background_uri()
    return f"""
<style>
/* Fonts ship with the app (static/fonts, SIL Open Font License) and are served by
   Streamlit's static file server, so the look never depends on a third-party CDN. */
@font-face {{ font-family: 'Pixelify Sans'; src: url('app/static/fonts/PixelifySans.ttf') format('truetype');
  font-weight: 400 700; font-display: swap; }}
@font-face {{ font-family: 'VT323'; src: url('app/static/fonts/VT323.ttf') format('truetype');
  font-weight: 400; font-display: swap; }}

:root {{
  --ink: {INK}; --paper: {PAPER}; --frame: {FRAME}; --titlebar: {TITLEBAR}; --chrome: {CHROME};
  --fresh: {FRESH}; --fresh-dark: {FRESH_DARK}; --rotten: {ROTTEN}; --rotten-dark: {ROTTEN_DARK};
  --unsure: {UNSURE}; --unsure-dark: {UNSURE_DARK};
  --display: 'Pixelify Sans', 'VT323', monospace;
  --body: 'VT323', 'Courier New', monospace;
}}

/* ---------- world ---------- */
.stApp {{
  background: #B9C2EE url("{bg}") center bottom / cover no-repeat fixed;
  image-rendering: pixelated;
  color: var(--ink);
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1120px; padding: 2.2rem 24px 3rem; }}
[data-testid="stHorizontalBlock"] {{ align-items: flex-start; }}
.st-key-win_info {{ margin-top: 14px; }}

.stApp p, .stApp label, .stApp li, .stApp input, .stApp button, .stApp small,
.stApp [data-testid="stMarkdownContainer"], [role="listbox"] [role="option"] {{
  font-family: var(--body);
  font-size: 21px;
  color: var(--ink);
}}
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}

/* ---------- title ---------- */
.fd-hero {{ text-align: center; margin-bottom: 18px; }}
.fd-title {{
  display: inline-flex; align-items: center; gap: 14px;
  font-family: var(--display); font-weight: 700;
  font-size: clamp(38px, 10vw, 60px); line-height: 1.05; color: #FFFFFF;
  letter-spacing: 1px;
  text-shadow:
    -3px 0 0 var(--ink), 3px 0 0 var(--ink), 0 -3px 0 var(--ink), 0 3px 0 var(--ink),
    6px 6px 0 rgba(46, 47, 91, 0.45);
}}
.fd-title .fd-sprite {{ filter: drop-shadow(4px 4px 0 rgba(46, 47, 91, 0.45)); flex: none; }}
.fd-tagline {{
  font-family: var(--body); font-size: 25px; color: var(--ink);
  background: rgba(243, 241, 251, 0.82); display: inline-block;
  padding: 4px 12px; margin-top: 10px; line-height: 1.1; box-shadow: 3px 3px 0 rgba(46, 47, 91, 0.25);
}}

/* ---------- windows (keyed containers win_main / win_verdict / win_about) ---------- */
[class*="st-key-win_"] {{
  background: var(--paper);
  border: 4px solid var(--frame);
  box-shadow: 8px 8px 0 rgba(46, 47, 91, 0.35);
  padding: 0 20px 22px;
  gap: 14px;
}}
.fd-titlebar {{
  margin: 0 -20px 4px; padding: 6px 8px 6px 12px;
  background: var(--titlebar); border-bottom: 4px solid var(--frame);
  display: flex; align-items: center; gap: 10px;
  font-family: var(--display); font-size: 20px; color: #FFFFFF; letter-spacing: 1px;
}}
.fd-titlebar .fd-name {{ flex: 1; }}
.fd-winbtn {{
  width: 26px; height: 24px; display: inline-flex; align-items: center; justify-content: center;
  background: var(--chrome); border: 3px solid var(--frame); color: var(--frame);
  font-family: var(--display); font-weight: 700; font-size: 16px; line-height: 1;
}}
.fd-label {{ font-family: var(--display); font-size: 20px; font-weight: 600; margin: 4px 0 -2px; }}

/* uploader */
[data-testid="stFileUploaderDropzone"] {{
  background: #FFFFFF; border: 3px dashed var(--frame); border-radius: 0;
}}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {{
  border-radius: 0; border: 3px solid var(--ink); box-shadow: 3px 3px 0 var(--ink);
  background: var(--chrome); color: var(--ink);
}}
[data-testid="stFileUploaderDropzoneInstructions"] * {{ font-family: var(--body); font-size: 19px; }}
[data-testid="stFileChip"], [data-testid="stFileChip"] *:not([data-testid="stIconMaterial"]) {{ font-family: var(--body) !important; font-size: 19px; color: var(--ink); }}

/* sample picker (Streamlit's combobox) */
[data-testid="stSelectbox"] [role="group"] {{
  border-radius: 0 !important; border: 3px solid var(--ink) !important; background: #FFFFFF !important;
  box-shadow: 3px 3px 0 var(--ink);
}}
[data-testid="stSelectbox"] input::placeholder {{ color: #7B7DA8; opacity: 1; }}
[role="listbox"] {{ border-radius: 0 !important; border: 3px solid var(--ink); }}
[role="listbox"] [role="option"][data-focused="true"], [role="listbox"] [role="option"]:hover {{ background: #E3E7F8; }}

/* photo, and the empty slot shown before one is picked */
[class*="st-key-win_"] [data-testid="stImage"] img {{
  border: 4px solid var(--ink); background: #FFFFFF;
  max-height: 340px; object-fit: contain;
}}

/* main button */
.stButton > button {{
  font-family: var(--display); font-weight: 600; font-size: 22px;
  background: #AFC0F0; color: var(--ink);
  border: 3px solid var(--ink); border-radius: 0;
  box-shadow: 5px 5px 0 var(--ink);
  padding: 8px 22px; transition: none;
}}
.stButton > button:hover {{ background: #C3D0F5; color: var(--ink); border-color: var(--ink); }}
.stButton > button:active {{ transform: translate(5px, 5px); box-shadow: none; }}
.stButton > button:focus-visible, [data-testid="stFileUploaderDropzone"] button:focus-visible {{
  outline: 3px dashed var(--frame); outline-offset: 4px;
}}
.stButton > button p {{ font-family: var(--display); font-size: 22px; }}

/* ---------- verdict ---------- */
.fd-verdict {{
  display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; align-items: center;
  border: 4px solid var(--ink); padding: 14px 16px; box-shadow: 6px 6px 0 rgba(46, 47, 91, 0.35);
}}
.fd-verdict .fd-sprite {{ grid-row: span 2; }}
.fd-verdict h3 {{
  font-family: var(--display); font-weight: 700; font-size: 32px; line-height: 1;
  margin: 0; padding: 0; color: var(--ink);
}}
.fd-verdict p {{ margin: 0; font-size: 21px; line-height: 1.15; }}
.fd-verdict.fresh {{ background: #D7F0DD; }}
.fd-verdict.rotten {{ background: #F1DCEA; }}
.fd-verdict.unsure {{ background: #FBF0CC; }}

.fd-cells {{ display: flex; gap: 3px; flex: 1; padding: 3px; background: var(--ink); }}
.fd-cell {{ flex: 1; height: 16px; background: #5A5D8C; }}
.fd-cells.fresh .fd-cell.on {{ background: var(--fresh); box-shadow: inset 0 3px 0 #B4E6C0; }}
.fd-cells.rotten .fd-cell.on {{ background: var(--rotten); box-shadow: inset 0 3px 0 #E6B9D6; }}
.fd-cells.unsure .fd-cell.on {{ background: var(--unsure); box-shadow: inset 0 3px 0 #FFF0B8; }}

@media (prefers-reduced-motion: no-preference) {{
  .fd-verdict {{ animation: fd-pop 0.28s steps(4, end) both; }}
  @keyframes fd-pop {{ from {{ transform: scale(0.86); opacity: 0; }} to {{ transform: none; opacity: 1; }} }}
}}

/* score bars (one per class) */
.fd-scores {{ display: grid; grid-template-columns: auto 1fr auto; gap: 8px 12px; align-items: center; margin-top: 16px; }}
.fd-scores .fd-name {{ font-family: var(--display); font-size: 18px; }}
.fd-scores .fd-val {{ font-size: 21px; min-width: 52px; text-align: right; }}
.fd-guess {{ margin: 14px 0 0; font-size: 21px; }}
.fd-guess b {{ font-family: var(--display); font-weight: 700; font-size: 20px; }}

/* empty states */
.fd-slot, .fd-waiting {{
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px;
  border: 4px dashed #A9AFD6; background: #FFFFFF; text-align: center; color: #6B6E9C;
}}
.fd-slot {{ min-height: 230px; }}
.fd-waiting {{ min-height: 200px; padding: 18px; }}
.fd-slot p, .fd-waiting p {{ margin: 0; color: #6B6E9C; max-width: 30ch; line-height: 1.15; }}
.fd-ghost {{ opacity: 0.35; }}

/* fruit_info window: rules note, then facts about the identified produce */
.fd-rules {{
  background: #FBF0CC; border: 3px solid var(--ink); box-shadow: 4px 4px 0 rgba(46, 47, 91, 0.35);
  padding: 10px 14px; margin: 2px 0 16px;
}}
.fd-rules p {{ margin: 0; line-height: 1.15; }}
.fd-rules .fd-rules-title {{ font-family: var(--display); font-weight: 700; font-size: 20px; margin-bottom: 4px; }}
.fd-rules b {{ font-family: var(--display); font-weight: 700; font-size: 20px; }}
/* about window */
.fd-info-name {{ font-family: var(--display) !important; font-weight: 700; font-size: 28px; line-height: 1; margin: 0 0 6px; padding: 0; color: var(--ink); }}
.fd-fact {{ margin: 0 0 12px; line-height: 1.15; }}
.fd-facts {{ margin: 0; display: grid; gap: 4px; }}
.fd-facts dt {{ font-family: var(--display); font-weight: 600; font-size: 18px; display: flex; align-items: center; gap: 8px; }}
.fd-facts dt::before {{ content: ""; width: 10px; height: 10px; flex: none; background: var(--titlebar); box-shadow: 2px 2px 0 var(--frame); }}
.fd-facts dd {{ margin: 0 0 8px 18px; line-height: 1.15; }}
.fd-info-empty {{ display: flex; flex-direction: column; align-items: center; gap: 10px; text-align: center;
  border: 4px dashed #A9AFD6; background: #FFFFFF; padding: 22px 16px; color: #6B6E9C; }}
.fd-info-empty p {{ margin: 0; color: #6B6E9C; max-width: 32ch; line-height: 1.15; }}
.fd-note {{ margin: 14px 0 0; font-size: 18px; color: #6B6E9C; line-height: 1.15; }}
.fd-note a {{ color: var(--frame); font-weight: 700; text-decoration: underline; text-underline-offset: 3px; }}

@media (max-width: 640px) {{
  [data-testid="stHorizontalBlock"] {{ row-gap: 24px; }}  /* columns stack on phones */
}}
@media (max-width: 480px) {{
  [class*="st-key-win_"] {{ padding: 0 14px 18px; }}
  .fd-titlebar {{ margin: 0 -14px 4px; }}
  .fd-title {{ gap: 8px; font-size: clamp(28px, 8.4vw, 40px); }}
  .fd-title .fd-sprite {{ width: 40px; height: 40px; }}
  .fd-tagline {{ font-size: 21px; }}
  .fd-verdict h3 {{ font-size: 27px; }}
}}
</style>
"""


def hero():
    return (f'<div class="fd-hero"><div class="fd-title">{APPLE}<span>Fruit Detector</span></div><br>'
            '<div class="fd-tagline">Fresh or rotten? Show it a fruit and find out.</div></div>')


def titlebar(name="fruit_detector.exe"):
    return (f'<div class="fd-titlebar"><span class="fd-name">{name}</span>'
            '<span class="fd-winbtn" aria-hidden="true">_</span>'
            '<span class="fd-winbtn" aria-hidden="true">□</span>'
            '<span class="fd-winbtn" aria-hidden="true">×</span></div>')


def _pct(p):
    # never claim 100%, and round DOWN so 69.8% shows as 69% (below the 70% "unsure" line)
    return "over 99%" if p >= 0.995 else f"{int(p * 100)}%"


def _cells(p, kind):
    filled = round(p * 10)
    cells = "".join(f'<div class="fd-cell{" on" if i < filled else ""}"></div>' for i in range(10))
    return f'<div class="fd-cells {kind}" role="img" aria-label="{kind} {int(p * 100)}%">{cells}</div>'


def verdict_card(label, pred_class, confidence, all_probs, guess=None):
    pct = _pct(confidence)
    if label == "not_produce":
        return (f'<div class="fd-verdict unsure" role="status">{ICON_UNSURE}<h3>That\'s not a fruit!</h3>'
                '<p>This doesn\'t look like a fruit or vegetable. Try a photo of one piece of produce.</p></div>')
    if label == "fresh":
        icon, title, line = ICON_FRESH, "Looks fresh!", f"Good to eat, {pct} sure."
    elif label == "rotten":
        icon, title, line = ICON_ROTTEN, "Looks rotten.", f"Better not eat it, {pct} sure."
    else:
        icon, title = ICON_UNSURE, "Hmm, not sure."
        line = (f"It leans {pred_class}, but it's only {pct} sure. "
                "A closer, well-lit photo of one fruit helps.")
    rows = "".join(
        f'<span class="fd-name">{cls.capitalize()}</span>{_cells(p, cls)}'
        f'<span class="fd-val">{"<1" if p < 0.01 else int(p * 100)}%</span>'
        for cls, p in all_probs.items()
    )
    guess_html = f'<p class="fd-guess">Looks like: <b>{guess}</b></p>' if guess else ""
    return (f'<div class="fd-verdict {label}" role="status">{icon}<h3>{title}</h3><p>{line}</p></div>'
            f'<div class="fd-scores">{rows}</div>{guess_html}')


def photo_slot():
    return (f'<div class="fd-slot"><span class="fd-ghost">{APPLE}</span>'
            '<p>Your fruit shows up here.</p></div>')


def waiting():
    return (f'<div class="fd-waiting"><span class="fd-ghost">{ICON_UNSURE}</span>'
            '<p>Upload a photo or pick a sample, then press Check freshness.</p></div>')


RULES = ('<div class="fd-rules" role="note"><p class="fd-rules-title">Rules</p>'
         '<p>Upload <b>one</b> fruit or vegetable per photo. A photo with several can give wrong results.</p></div>')

# USDA rule for mouldy produce: firm items can be trimmed, soft ones should go.
MOULD_FIRM = ("Firm produce: if it's just a small mouldy spot, cut away at least 1 inch (2.5 cm) around and "
              "below it and use the rest. If it's soft, slimy or mouldy all over, throw it out.")
MOULD_SOFT = ("Soft produce: throw it away. Mould can spread below the surface where you can't see it. "
              "Compost it if you can.")


def _rows(rows):
    return '<dl class="fd-facts">' + "".join(f"<dt>{k}</dt><dd>{html.escape(v)}</dd>" for k, v in rows) + "</dl>"


def fruit_info(result, info, repo_url):
    """
    The fruit_info.txt window: the upload rule, then facts and advice about
    whatever produce was identified, tailored to the verdict.
    result: None before a check, else the app's result dict (label, guess, ...).
    info:   the "produce" table from src/produce_info.json.
    """
    footer = ('<p class="fd-note">Storage times are rough guides. When in doubt, throw it out. '
              f'<a href="{repo_url}" target="_blank" rel="noopener">How this app works</a></p>')
    if result is None:
        body = (f'<div class="fd-info-empty"><span class="fd-ghost">{APPLE}</span>'
                '<p>Check a fruit and facts about it show up here.</p></div>')
    elif result["label"] == "not_produce":
        body = '<div class="fd-info-empty"><p>No fruit or vegetable spotted, so there\'s nothing to show yet.</p></div>'
    elif not result.get("guess") or result["guess"] not in info:
        body = ('<h4 class="fd-info-name">Couldn\'t tell which one this is</h4>'
                '<p class="fd-fact">So here is the general rule for mouldy produce:</p>'
                + _rows([("Firm, like carrots or peppers", MOULD_FIRM.split(": ", 1)[1]),
                         ("Soft, like berries or tomatoes", MOULD_SOFT.split(": ", 1)[1])]))
    else:
        name, d = result["guess"], info[result["guess"]]
        rows = [("Good source of", d["good_for"])]
        if result["label"] == "rotten":
            rows += [("What to do now", MOULD_FIRM if d["firm"] else MOULD_SOFT),
                     ("Signs it's going bad", d["spoil"])]
        elif result["label"] == "unsure":
            rows += [("Check it yourself", d["spoil"]), ("How to store it", d["store"]), ("Keeps for", d["keeps"])]
        else:
            rows += [("How to store it", d["store"]), ("Keeps for", d["keeps"])]
        body = (f'<h4 class="fd-info-name">{html.escape(name.capitalize())}</h4>'
                f'<p class="fd-fact">{html.escape(d["fact"])}</p>' + _rows(rows))
    return RULES + f'<div class="fd-info" aria-live="polite">{body}</div>' + footer
