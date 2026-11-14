
import base64
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CropCare AI",
    page_icon="🌱",
    layout="wide"
)


# ==================================================
# 1A. VISUAL IDENTITY: "BOTANICAL AI LAB" (UI ONLY)
# ==================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,700;1,9..144,500&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&family=Noto+Sans+Kannada:wght@400;500;600&display=swap');

:root {
    --midnight: #07231a; --forest: #0d3b2c; --emerald: #12a36b; --lime: #c6f432;
    --sage: #a9bfa8; --sage-soft: #e4ece1; --ivory: #fbf8f0; --ink: #10261d;
    --muted: #5b6e62; --line: rgba(13,59,44,.14);
    --amber: #e8a317; --rust: #d4582a;
    --leaf-mark: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><path d='M50 95C20 70 15 35 50 5c35 30 30 65 0 90z' fill='%2312a36b' fill-opacity='.10'/><path d='M50 95V15' stroke='%2312a36b' stroke-opacity='.25' stroke-width='1.5'/></svg>");
}
html { scroll-behavior: smooth; }
html, body, .stApp, [class*="css"] { font-family: 'Manrope', 'Noto Sans Kannada', sans-serif; color: var(--ink); }
.stApp {
    background:
        radial-gradient(900px 500px at 100% 0%, rgba(18,163,107,.10), transparent 60%),
        radial-gradient(700px 500px at 0% 40%, rgba(198,244,50,.08), transparent 60%),
        var(--ivory);
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { max-width: 1240px; padding: 1rem 1.6rem 3rem; }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes drift { 0%,100% { transform: translateY(0) rotate(0); } 50% { transform: translateY(-8px) rotate(.8deg); } }
.rise { animation: rise .7s ease both; }

/* ---------- Navigation ---------- */
.nav {
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    background: rgba(255,255,255,.7); backdrop-filter: blur(14px);
    border: 1px solid var(--line); border-radius: 999px; padding: .55rem .6rem .55rem 1.2rem;
}
.brand { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.25rem; color: var(--midnight); display: flex; align-items: center; gap: .55rem; }
.brand .logo { width: 30px; height: 30px; border-radius: 10px 10px 10px 2px; background: linear-gradient(135deg, var(--emerald), var(--lime)); display: grid; place-items: center; font-size: .95rem; }
.brand b { color: var(--emerald); font-weight: 700; margin-left: .15rem; }
.links { display: flex; gap: .3rem; flex-wrap: wrap; }
.links a { color: var(--forest) !important; text-decoration: none !important; font-weight: 600; font-size: .88rem; padding: .5rem 1rem; border-radius: 999px; transition: all .2s ease; }
.links a:hover { background: var(--midnight); color: var(--lime) !important; }

/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden; margin: 1.1rem 0 2rem;
    display: grid; grid-template-columns: 1.05fr 1fr; gap: 1rem; align-items: center;
    background: radial-gradient(700px 420px at 85% 40%, #0f5a40 0%, transparent 65%),
                linear-gradient(145deg, #041a13 0%, var(--midnight) 50%, var(--forest) 100%);
    border-radius: 44px 150px 44px 44px; padding: 3rem 2.4rem 3rem 3.2rem;
    box-shadow: 0 30px 70px rgba(7,35,26,.28);
}
.badge { display: inline-flex; align-items: center; gap: .5rem; padding: .4rem .95rem; border-radius: 999px; font-size: .74rem; font-weight: 700; letter-spacing: .1em; color: var(--lime); background: rgba(198,244,50,.08); border: 1px solid rgba(198,244,50,.35); }
.badge i { width: 7px; height: 7px; border-radius: 50%; background: var(--lime); box-shadow: 0 0 10px var(--lime); }
.hero h1 { font-family: 'Fraunces', serif; font-weight: 700; font-size: 3.6rem; line-height: 1.02; letter-spacing: -.02em; color: #f4f7ef; margin: 1.2rem 0 1rem; padding: 0; }
.hero h1 em { font-style: italic; font-weight: 500; color: var(--lime); }
.hero p { color: rgba(236,244,232,.78); font-size: 1.03rem; line-height: 1.7; max-width: 520px; margin: 0 0 1.6rem; }
.cta { display: inline-flex; align-items: center; gap: .7rem; background: var(--lime); color: var(--midnight) !important; font-weight: 800; text-decoration: none !important; padding: .9rem 1.5rem; border-radius: 999px; box-shadow: 0 10px 30px rgba(198,244,50,.28); transition: all .25s ease; }
.cta:hover { transform: translateY(-3px); box-shadow: 0 16px 38px rgba(198,244,50,.4); }
.cta span { display: grid; place-items: center; width: 26px; height: 26px; border-radius: 50%; background: var(--midnight); color: var(--lime); font-size: .85rem; }
.stats { display: flex; gap: 0; margin-top: 2.2rem; border-top: 1px solid rgba(255,255,255,.12); padding-top: 1.2rem; }
.stats div { padding-right: 1.8rem; margin-right: 1.8rem; border-right: 1px solid rgba(255,255,255,.12); }
.stats div:last-child { border: none; margin: 0; padding: 0; }
.stats b { display: block; font-family: 'Fraunces', serif; font-size: 2.2rem; color: #fff; line-height: 1; }
.stats small { color: var(--sage); font-size: .72rem; letter-spacing: .1em; font-weight: 700; }
.hero-art { animation: drift 9s ease-in-out infinite; }
.hero-art svg { width: 100%; height: auto; display: block; }
.art-note { text-align: right; color: rgba(169,191,168,.6); font-size: .66rem; letter-spacing: .08em; margin-top: -.4rem; }

/* ---------- Section headings ---------- */
.sec { margin: 3rem 0 1.1rem; display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem; flex-wrap: wrap; animation: rise .6s ease both; }
.sec .no { font-family: 'JetBrains Mono', monospace; color: var(--emerald); font-size: .78rem; letter-spacing: .12em; }
.sec h2 { font-family: 'Fraunces', serif; font-size: 2.3rem; line-height: 1.05; letter-spacing: -.015em; margin: .2rem 0 0; padding: 0; color: var(--midnight); }
.sec h2 em { color: var(--emerald); font-weight: 500; }
.sec p { max-width: 380px; margin: 0; color: var(--muted); font-size: .92rem; line-height: 1.55; }

/* ---------- Upload ---------- */
[data-testid="stFileUploaderDropzone"] {
    flex-direction: column; justify-content: center; align-items: center; gap: .9rem; min-height: 300px; position: relative;
    background: linear-gradient(160deg, rgba(255,255,255,.85), var(--sage-soft));
    border: 2px dashed #6fa98a; border-radius: 44px 44px 44px 10px; transition: all .25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--emerald); transform: translateY(-3px); box-shadow: 0 24px 50px rgba(13,59,44,.12); }
[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
[data-testid="stFileUploaderDropzone"]::before {
    content: "Drop your leaf image here"; display: block; text-align: center; font-family: 'Fraunces', serif; font-size: 1.9rem; font-weight: 700; color: var(--midnight);
    padding-top: 84px; background: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><circle cx='32' cy='32' r='30' fill='%23c6f432'/><path d='M32 44V21M22.5 30.5L32 21l9.5 9.5' stroke='%2307231a' stroke-width='4' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>") no-repeat top center / 66px;
}
[data-testid="stFileUploaderDropzone"]::after { content: "Supported formats: JPG · JPEG · PNG"; font-family: 'JetBrains Mono', monospace; font-size: .74rem; letter-spacing: .06em; color: var(--muted); }
[data-testid="stFileUploaderDropzone"] button { background: var(--midnight); color: var(--lime); border: none; border-radius: 999px; padding: .65rem 1.6rem; font-weight: 700; transition: all .2s ease; }
[data-testid="stFileUploaderDropzone"] button:hover { background: var(--emerald); color: #fff; transform: scale(1.04); }
.tips { height: 100%; background: var(--midnight); border-radius: 10px 44px 44px 44px; padding: 1.8rem 1.7rem; color: #e9f1e6; position: relative; overflow: hidden; }
.tips::after { content: ""; position: absolute; right: -30px; bottom: -30px; width: 190px; height: 190px; background: var(--leaf-mark) no-repeat center / contain; filter: saturate(3) brightness(1.6); opacity: .6; }
.tips h4 { font-family: 'Fraunces', serif; font-size: 1.35rem; margin: 0 0 1rem; color: var(--lime); }
.tips ol { list-style: none; counter-reset: t; margin: 0; padding: 0; position: relative; z-index: 1; }
.tips li { counter-increment: t; display: flex; gap: .8rem; margin-bottom: .85rem; font-size: .9rem; line-height: 1.45; color: rgba(233,241,230,.85); }
.tips li::before { content: "0" counter(t); font-family: 'JetBrains Mono', monospace; color: var(--lime); font-size: .78rem; padding-top: .15rem; }

/* ---------- Native widgets ---------- */
.stButton > button { border-radius: 999px; font-weight: 700; padding: .7rem 1.4rem; border: 1px solid var(--line); background: #fff; transition: all .22s ease; }
.stButton > button:hover { border-color: var(--emerald); color: var(--emerald); transform: translateY(-2px); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, var(--emerald), #0b7a52); color: #fff; border: none; padding: 1rem 1.4rem; font-size: 1.05rem; box-shadow: 0 14px 30px rgba(18,163,107,.35); }
.stButton > button[kind="primary"]:hover { color: var(--lime); box-shadow: 0 20px 40px rgba(18,163,107,.45); }
[data-baseweb="select"] > div { border-radius: 14px; background: rgba(255,255,255,.85); border-color: var(--line); }
[data-testid="stImage"] img { border-radius: 28px 28px 28px 6px; box-shadow: 0 22px 50px rgba(7,35,26,.22); border: 5px solid #fff; }
[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid var(--line) !important; border-radius: 26px !important; background: rgba(255,255,255,.55); backdrop-filter: blur(8px); }
[data-testid="stAlert"] { border-radius: 18px; border: 1px solid var(--line); }
[data-testid="stSpinner"] { color: var(--emerald); font-weight: 600; }
[data-testid="stSpinner"] i { border-top-color: var(--emerald) !important; }
[data-testid="stExpander"] { border-radius: 18px; border: 1px solid var(--line); background: rgba(255,255,255,.6); }

/* ---------- Screening strip ---------- */
.screen { display: grid; grid-template-columns: auto 1fr; gap: 1.2rem; align-items: center; background: #fff; border: 1px solid var(--line); border-radius: 22px 22px 22px 4px; padding: 1rem 1.4rem; }
.screen small, .mini small { font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .1em; color: var(--muted); text-transform: uppercase; }
.screen b { font-family: 'Fraunces', serif; font-size: 1.9rem; color: var(--midnight); display: block; }
.track { height: 8px; background: var(--sage-soft); border-radius: 99px; overflow: hidden; }
.track i { display: block; height: 100%; background: linear-gradient(90deg, var(--emerald), var(--lime)); border-radius: 99px; }

/* ---------- Lab result panel ---------- */
.lab {
    position: relative; overflow: hidden; color: #eef5ea; border-radius: 10px 60px 10px 60px; padding: 2rem 2.2rem;
    background: radial-gradient(500px 300px at 100% 0%, rgba(18,163,107,.35), transparent 70%), linear-gradient(150deg, #041a13, var(--forest));
    box-shadow: 0 28px 60px rgba(7,35,26,.3); border: 1px solid rgba(198,244,50,.18);
}
.lab::before { content: ""; position: absolute; inset: 0; background-image: linear-gradient(rgba(198,244,50,.05) 1px, transparent 1px), linear-gradient(90deg, rgba(198,244,50,.05) 1px, transparent 1px); background-size: 34px 34px; pointer-events: none; }
.lab-top { display: flex; justify-content: space-between; align-items: center; position: relative; }
.status { font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .1em; color: var(--lime); display: inline-flex; align-items: center; gap: .5rem; }
.status i { width: 8px; height: 8px; border-radius: 50%; background: var(--lime); box-shadow: 0 0 12px var(--lime); }
.lab-main { display: grid; grid-template-columns: 1fr auto; gap: 1.5rem; align-items: center; margin: 1.3rem 0; position: relative; }
.lab small { font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .1em; color: var(--sage); text-transform: uppercase; }
.lab .crop { font-size: 1.05rem; font-weight: 700; color: #fff; margin: .2rem 0 1rem; }
.lab h3 { font-family: 'Fraunces', serif; font-size: 2.5rem; line-height: 1.05; margin: .3rem 0 1rem; color: #fff; padding: 0; }
.pill { display: inline-block; padding: .38rem 1rem; border-radius: 999px; font-weight: 800; font-size: .78rem; }
.pill.ok { background: var(--lime); color: var(--midnight); }
.pill.warn { background: var(--amber); color: #2b1c00; }
.pill.bad { background: var(--rust); color: #fff; }
.ring { --p: 0; width: 172px; height: 172px; border-radius: 50%; display: grid; place-items: center; background: conic-gradient(var(--lime) calc(var(--p) * 1%), rgba(255,255,255,.1) 0); filter: drop-shadow(0 0 18px rgba(198,244,50,.28)); }
.ring div { width: 140px; height: 140px; border-radius: 50%; background: var(--midnight); display: flex; flex-direction: column; align-items: center; justify-content: center; }
.ring b { font-family: 'Fraunces', serif; font-size: 2rem; color: #fff; line-height: 1; }
.facts { display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem; position: relative; }
.facts div { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1); border-radius: 16px; padding: .8rem 1rem; backdrop-filter: blur(6px); }
.facts b { display: block; color: #fff; font-size: 1.1rem; margin-top: .2rem; }

/* ---------- Disease information (editorial) ---------- */
.edit { display: grid; grid-template-columns: 1fr 1.15fr; gap: 1.1rem; }
.edit .stack { display: grid; gap: 1.1rem; }
.panel { position: relative; overflow: hidden; border-radius: 30px; padding: 1.7rem 1.8rem; background: #fff; border: 1px solid var(--line); transition: all .25s ease; }
.panel:hover { transform: translateY(-4px); box-shadow: 0 24px 50px rgba(13,59,44,.12); }
.panel::after { content: ""; position: absolute; right: -14px; bottom: -14px; width: 120px; height: 120px; background: var(--leaf-mark) no-repeat center / contain; }
.panel .ic { width: 42px; height: 42px; border-radius: 14px 14px 14px 3px; display: grid; place-items: center; font-size: 1.2rem; background: var(--sage-soft); margin-bottom: .9rem; }
.panel h4 { font-family: 'Fraunces', serif; font-size: 1.35rem; margin: 0 0 .5rem; padding: 0; color: var(--midnight); }
.panel p { margin: 0; font-size: 1.03rem; line-height: 1.7; color: var(--ink); position: relative; z-index: 1; }
.panel.overview { background: linear-gradient(160deg, var(--sage-soft), #d5e4d1); border-radius: 60px 14px 14px 14px; }
.panel.overview .kv { display: flex; justify-content: space-between; border-bottom: 1px dashed rgba(13,59,44,.25); padding: .55rem 0; font-size: .95rem; position: relative; z-index: 1; }
.panel.overview .kv b { color: var(--midnight); }
.panel.symptoms { border-left: 6px solid var(--amber); border-radius: 14px 30px 30px 14px; }
.panel.guide { background: var(--midnight); border: none; border-radius: 30px 14px 60px 30px; }
.panel.guide h4 { color: var(--lime); } .panel.guide p { color: #e9f1e6; } .panel.guide .ic { background: rgba(198,244,50,.15); }

/* ---------- Top 3 ---------- */
.top3 { display: grid; grid-template-columns: 1.4fr 1fr 1fr; gap: 1rem; }
.t3 { background: #fff; border: 1px solid var(--line); border-radius: 26px; padding: 1.3rem 1.4rem; transition: all .25s ease; }
.t3:hover { transform: translateY(-4px); box-shadow: 0 20px 40px rgba(13,59,44,.12); }
.t3.first { background: linear-gradient(150deg, var(--midnight), var(--forest)); color: #fff; border-radius: 10px 44px 10px 44px; }
.t3 .rk { font-family: 'JetBrains Mono', monospace; font-size: .75rem; color: var(--emerald); letter-spacing: .1em; }
.t3.first .rk { color: var(--lime); }
.t3 h5 { font-family: 'Fraunces', serif; font-size: 1.25rem; margin: .4rem 0 .2rem; padding: 0; }
.t3 .pc { font-family: 'Fraunces', serif; font-size: 2rem; font-weight: 700; margin-bottom: .5rem; }
.t3.first .track { background: rgba(255,255,255,.15); }

/* ---------- History timeline ---------- */
.tl { position: relative; padding-left: 2.2rem; }
.tl::before { content: ""; position: absolute; left: 11px; top: 8px; bottom: 8px; width: 2px; background: linear-gradient(var(--emerald), transparent); }
.tl-item { position: relative; display: flex; align-items: center; gap: 1.2rem; background: #fff; border: 1px solid var(--line); border-radius: 22px; padding: .9rem 1.2rem; margin-bottom: .9rem; transition: all .22s ease; animation: rise .5s ease both; }
.tl-item:hover { transform: translateX(6px); box-shadow: 0 14px 34px rgba(13,59,44,.1); }
.tl-item::before { content: ""; position: absolute; left: -2.2rem; top: 50%; width: 14px; height: 14px; margin: -7px 0 0 5px; border-radius: 50%; background: var(--lime); border: 3px solid var(--forest); }
.tl-item img, .tl-item .ph { width: 62px; height: 62px; border-radius: 16px 16px 16px 4px; object-fit: cover; flex: none; background: var(--sage-soft); display: grid; place-items: center; }
.tl-body { flex: 1; min-width: 0; }
.tl-body b { font-size: 1.02rem; } .tl-body span { display: block; color: var(--muted); font-size: .8rem; margin-top: .15rem; }
.tl-conf { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.35rem; color: var(--forest); }
.tl-tag { font-size: .68rem; font-weight: 800; padding: .25rem .65rem; border-radius: 99px; background: var(--sage-soft); color: var(--forest); letter-spacing: .05em; }
.empty { text-align: center; padding: 2.2rem; border: 2px dashed var(--line); border-radius: 30px; color: var(--muted); background: rgba(255,255,255,.5); }

.foot { margin-top: 3.5rem; padding: 1.6rem 2rem; border-radius: 30px; background: var(--midnight); color: var(--sage); font-size: .85rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: .5rem; }
.foot b { color: var(--lime); font-family: 'Fraunces', serif; font-size: 1.05rem; }

@media (max-width: 900px) {
    .hero { grid-template-columns: 1fr; border-radius: 36px 70px 36px 36px; padding: 2rem 1.5rem; }
    .hero h1 { font-size: 2.6rem; } .nav { border-radius: 26px; flex-direction: column; align-items: flex-start; }
    .lab-main, .edit, .top3, .facts { grid-template-columns: 1fr; } .lab h3 { font-size: 2rem; }
    .sec h2 { font-size: 1.8rem; }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def html(markup):
    """Render HTML. Lines are joined with spaces so Markdown never sees code blocks."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def section(number, title_plain, title_accent, blurb, anchor=""):
    anchor_html = f'<div id="{anchor}"></div>' if anchor else ""
    html(f"""
    {anchor_html}
    <div class="sec">
        <div><span class="no">{number}</span><h2>{title_plain} <em>{title_accent}</em></h2></div>
        <p>{blurb}</p>
    </div>
    """)


# ==================================================
# 1B. NAVIGATION  (language selector is rendered in section 5)
# ==================================================

nav_col, lang_col = st.columns([3.3, 1.2], vertical_alignment="center")

with nav_col:
    html("""
    <div class="nav">
        <div class="brand"><span class="logo">🌱</span>CropCare<b>AI</b></div>
        <div class="links">
            <a href="#home" target="_self">Home</a>
            <a href="#detect" target="_self">Disease Detection</a>
            <a href="#history" target="_self">Prediction History</a>
        </div>
    </div>
    """)


# ==================================================
# 1C. HERO
# ==================================================

HERO_SVG = """
<svg viewBox="0 0 520 520" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Illustration of a leaf with scientific annotations">
  <defs>
    <linearGradient id="lg1" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2fd18b"/><stop offset="1" stop-color="#0a5a3e"/></linearGradient>
    <linearGradient id="lg2" x1="1" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#7be08a"/><stop offset="1" stop-color="#0d3b2c"/></linearGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <circle cx="260" cy="270" r="215" fill="none" stroke="rgba(198,244,50,.14)" stroke-dasharray="2 8"/>
  <circle cx="260" cy="270" r="165" fill="none" stroke="rgba(198,244,50,.10)"/>
  <g transform="rotate(-38 260 280)" opacity=".75">
    <path d="M260 470C170 390 150 250 260 110c110 140 90 280 0 360z" fill="url(#lg2)"/>
    <path d="M260 470V130" stroke="rgba(255,255,255,.25)" stroke-width="2"/>
  </g>
  <path d="M255 480C140 390 110 220 255 50c145 170 115 340 0 430z" fill="url(#lg1)"/>
  <g stroke="rgba(236,255,230,.5)" stroke-width="1.6" fill="none" stroke-linecap="round">
    <path d="M255 480V70" stroke-width="3"/>
    <path d="M255 410Q205 390 165 340M255 410Q305 390 345 340"/>
    <path d="M255 340Q200 318 158 262M255 340Q310 318 352 262"/>
    <path d="M255 270Q208 250 178 200M255 270Q302 250 332 200"/>
    <path d="M255 200Q222 184 205 148M255 200Q288 184 305 148"/>
  </g>
  <g fill="none" stroke="#c6f432" stroke-width="3" stroke-linecap="round" filter="url(#glow)">
    <path d="M95 110V80h30M425 80h30v30M95 440v30h30M455 440v30h-30"/>
  </g>
  <g stroke="#c6f432" stroke-width="1" stroke-opacity=".75" fill="none">
    <path d="M200 250L110 205H62"/><path d="M335 200L400 150h58"/><path d="M300 370L385 400h74"/>
  </g>
  <g fill="#c6f432" filter="url(#glow)"><circle cx="200" cy="250" r="5"/><circle cx="335" cy="200" r="5"/><circle cx="300" cy="370" r="5"/></g>
  <g fill="rgba(198,244,50,.25)"><circle cx="200" cy="250" r="14"/><circle cx="335" cy="200" r="14"/><circle cx="300" cy="370" r="14"/></g>
  <g font-family="JetBrains Mono, monospace" font-size="11" fill="#e8f5d8" letter-spacing=".8">
    <text x="48" y="198">INPUT 224×224 px</text>
    <text x="392" y="142">8 CLASSES</text>
    <text x="344" y="418">VALIDATOR 160 px</text>
  </g>
</svg>
"""

html(f"""
<div id="home"></div>
<div class="hero rise">
    <div>
        <span class="badge"><i></i> AI-POWERED AGRICULTURE</span>
        <h1>Give Every Leaf a <em>Healthier</em> Future.</h1>
        <p>CropCare AI studies a photograph of a tomato, potato or bell pepper leaf and
        estimates its condition, with confidence scores, symptoms and practical guidance
        in English and ಕನ್ನಡ. Built to support, never replace, local agricultural expertise.</p>
        <a class="cta" href="#detect" target="_self">Analyse a Leaf <span>→</span></a>
        <div class="stats">
            <div><b>3</b><small>CROPS</small></div>
            <div><b>8</b><small>CONDITIONS</small></div>
            <div><b>2</b><small>LANGUAGES</small></div>
        </div>
    </div>
    <div>
        <div class="hero-art">{HERO_SVG}</div>
        <div class="art-note">ILLUSTRATION · NOT A LIVE SCAN</div>
    </div>
</div>
""")


# ==================================================
# 2. PATHS AND SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

DISEASE_MODEL_PATH = MODEL_DIR / "best_cropcare_model.keras"
VALIDATOR_MODEL_PATH = MODEL_DIR / "crop_validator.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)
VALIDATOR_SIZE = (160, 160)

# Preliminary threshold. Must be calibrated with real-world images.
VALIDATOR_THRESHOLD = 0.50
DISEASE_THRESHOLD = 0.60


# ==================================================
# 3. LOAD MODELS
# ==================================================

@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model(DISEASE_MODEL_PATH)


@st.cache_resource
def load_validator_model():
    return tf.keras.models.load_model(VALIDATOR_MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as file:
        return json.load(file)


try:
    disease_model = load_disease_model()
    validator_model = load_validator_model()
    class_names = load_class_names()

except Exception as error:
    st.error(f"Unable to load AI models: {error}")
    st.stop()


# ==================================================
# 4. DISEASE INFORMATION
# ==================================================

disease_info = {

    "Tomato_healthy": {
        "crop": "Tomato",
        "status": "Healthy",
        "symptoms": "No visible disease symptoms.",
        "guidance": "Maintain regular watering, proper nutrition and routine plant monitoring.",
        "kn_symptoms": "ರೋಗದ ಯಾವುದೇ ಸ್ಪಷ್ಟ ಲಕ್ಷಣಗಳು ಕಾಣಿಸುತ್ತಿಲ್ಲ.",
        "kn_guidance": "ನಿಯಮಿತವಾಗಿ ನೀರು ನೀಡಿ, ಪೋಷಕಾಂಶಗಳನ್ನು ಒದಗಿಸಿ ಮತ್ತು ಗಿಡವನ್ನು ಗಮನಿಸುತ್ತಿರಿ."
    },

    "Tomato_Early_blight": {
        "crop": "Tomato",
        "status": "Early Blight",
        "symptoms": "Brown spots with concentric rings, usually on older leaves.",
        "guidance": "Remove affected leaves, avoid overhead watering and maintain good airflow.",
        "kn_symptoms": "ಹಳೆಯ ಎಲೆಗಳಲ್ಲಿ ವೃತ್ತಾಕಾರದ ಕಂದು ಚುಕ್ಕೆಗಳು ಕಾಣಿಸಬಹುದು.",
        "kn_guidance": "ಬಾಧಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ. ಎಲೆಗಳ ಮೇಲೆ ನೇರವಾಗಿ ನೀರು ಹಾಕುವುದನ್ನು ತಪ್ಪಿಸಿ."
    },

    "Tomato_Late_blight": {
        "crop": "Tomato",
        "status": "Late Blight",
        "symptoms": "Dark, water-soaked patches that may spread rapidly.",
        "guidance": "Remove affected plant material and seek advice from a local agricultural expert.",
        "kn_symptoms": "ಎಲೆಗಳ ಮೇಲೆ ಗಾಢ ಬಣ್ಣದ ನೀರಿನಿಂದ ನೆನೆದಂತಹ ಕಲೆಗಳು ಕಾಣಿಸಬಹುದು.",
        "kn_guidance": "ಬಾಧಿತ ಸಸ್ಯ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
    },

    "Potato___healthy": {
        "crop": "Potato",
        "status": "Healthy",
        "symptoms": "No visible disease symptoms.",
        "guidance": "Maintain proper irrigation and regularly inspect the plants.",
        "kn_symptoms": "ರೋಗದ ಯಾವುದೇ ಸ್ಪಷ್ಟ ಲಕ್ಷಣಗಳು ಕಾಣಿಸುತ್ತಿಲ್ಲ.",
        "kn_guidance": "ಸರಿಯಾದ ನೀರಾವರಿ ನೀಡಿ ಮತ್ತು ಗಿಡಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ."
    },

    "Potato___Early_blight": {
        "crop": "Potato",
        "status": "Early Blight",
        "symptoms": "Brown circular spots, often with concentric rings.",
        "guidance": "Remove affected leaves and avoid prolonged leaf wetness.",
        "kn_symptoms": "ಎಲೆಗಳ ಮೇಲೆ ವೃತ್ತಾಕಾರದ ಕಂದು ಚುಕ್ಕೆಗಳು ಕಾಣಿಸಬಹುದು.",
        "kn_guidance": "ಬಾಧಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಎಲೆಗಳು ಹೆಚ್ಚು ಸಮಯ ತೇವವಾಗಿರುವುದನ್ನು ತಪ್ಪಿಸಿ."
    },

    "Potato___Late_blight": {
        "crop": "Potato",
        "status": "Late Blight",
        "symptoms": "Dark irregular lesions that can spread quickly.",
        "guidance": "Remove affected plant material and consult an agricultural expert.",
        "kn_symptoms": "ಎಲೆಗಳ ಮೇಲೆ ಅನಿಯಮಿತ ಆಕಾರದ ಗಾಢ ಕಲೆಗಳು ಕಾಣಿಸಬಹುದು.",
        "kn_guidance": "ಬಾಧಿತ ಸಸ್ಯ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
    },

    "Pepper__bell___healthy": {
        "crop": "Bell Pepper",
        "status": "Healthy",
        "symptoms": "No visible disease symptoms.",
        "guidance": "Maintain regular watering and inspect plants routinely.",
        "kn_symptoms": "ರೋಗದ ಯಾವುದೇ ಸ್ಪಷ್ಟ ಲಕ್ಷಣಗಳು ಕಾಣಿಸುತ್ತಿಲ್ಲ.",
        "kn_guidance": "ನಿಯಮಿತವಾಗಿ ನೀರು ನೀಡಿ ಮತ್ತು ಗಿಡಗಳನ್ನು ಪರಿಶೀಲಿಸುತ್ತಿರಿ."
    },

    "Pepper__bell___Bacterial_spot": {
        "crop": "Bell Pepper",
        "status": "Bacterial Spot",
        "symptoms": "Small dark or water-soaked spots may appear on leaves.",
        "guidance": "Avoid overhead irrigation, remove affected material and seek local agricultural advice.",
        "kn_symptoms": "ಎಲೆಗಳ ಮೇಲೆ ಸಣ್ಣ ಗಾಢ ಅಥವಾ ನೀರಿನಿಂದ ನೆನೆದಂತಹ ಕಲೆಗಳು ಕಾಣಿಸಬಹುದು.",
        "kn_guidance": "ಎಲೆಗಳ ಮೇಲೆ ನೇರವಾಗಿ ನೀರು ಹಾಕುವುದನ್ನು ತಪ್ಪಿಸಿ ಮತ್ತು ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
    }
}


# ==================================================
# 5. LANGUAGE  (rendered inside the navigation row)
# ==================================================

with lang_col:
    language = st.selectbox(
        "Choose Language / ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        ["English", "ಕನ್ನಡ"],
        label_visibility="collapsed"
    )

is_kannada = language == "ಕನ್ನಡ"


# ==================================================
# 6. IMAGE UPLOAD
# ==================================================

section(
    "01 / UPLOAD", "Drop in a", "leaf.",
    "A sharp, well-lit close-up gives the model the best chance of a useful result.",
    anchor="detect"
)

up_col, tips_col = st.columns([1.7, 1], gap="medium")

with up_col:
    uploaded_file = st.file_uploader(
        "Choose a leaf photograph",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with tips_col:
    html("""
    <div class="tips">
        <h4>Photography tips</h4>
        <ol>
            <li>Use natural daylight and avoid harsh shadows.</li>
            <li>Fill the frame with a single leaf.</li>
            <li>Keep the leaf in focus against a plain background.</li>
            <li>Photograph the affected area up close.</li>
        </ol>
    </div>
    """)

st.caption(
    "Supported categories: Tomato (Healthy, Early Blight, Late Blight), "
    "Potato (Healthy, Early Blight, Late Blight), and Bell Pepper "
    "(Healthy, Bacterial Spot)."
)


# ==================================================
# 7. PREDICTION HISTORY
# ==================================================

if "history" not in st.session_state:
    st.session_state.history = []

# Additive, display-only companion list (thumbnail + time recorded when a
# prediction is made). The original "history" records are unchanged.
if "history_extra" not in st.session_state:
    st.session_state.history_extra = []


# ==================================================
# 8. IMAGE VALIDATION AND DISEASE DETECTION
# ==================================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:
            st.info(
                "Upload a clear image of a tomato, potato "
                "or bell pepper leaf for best results."
            )

            with st.container(border=True):
                st.markdown("### Optional demo verification")
                st.caption(
                    "If you already know the leaf's correct label (for example, "
                    "from a verified test image), you can select it here. This is a "
                    "manual demo correction and does not change the AI model."
                )

                demo_label_options = ["Use model prediction"] + list(disease_info.keys())

                def format_demo_label(label):
                    if label == "Use model prediction":
                        return label
                    label_info = disease_info[label]
                    return f"{label_info['crop']} — {label_info['status']} (manual)"

                selected_demo_label = st.selectbox(
                    "Verified label for demo (optional)",
                    demo_label_options,
                    format_func=format_demo_label,
                    key="demo_verified_label"
                )

            detect_clicked = st.button(
                "🔍 Detect Crop Disease",
                type="primary",
                use_container_width=True
            )

        if detect_clicked:

            # ------------------------------------------
            # STAGE 1: IMAGE VALIDATION
            # ------------------------------------------

            with st.spinner("Checking image..."):

                validator_image = image.resize(
                    VALIDATOR_SIZE
                )

                validator_array = np.asarray(
                    validator_image,
                    dtype=np.float32
                )

                validator_array = np.expand_dims(
                    validator_array,
                    axis=0
                )

                # Output is sigmoid probability:
                # Class 0 = crop_leaf
                # Class 1 = non_crop

                non_crop_probability = float(
                    validator_model.predict(
                        validator_array,
                        verbose=0
                    )[0][0]
                )

                crop_probability = 1.0 - non_crop_probability

            section(
                "02 / SCREENING", "Image", "validation.",
                "A preliminary screening signal produced before disease detection."
            )

            crop_bar = max(0.0, min(crop_probability * 100, 100.0))
            html(f"""
            <div class="screen">
                <div><small>Estimated crop-leaf score</small><b>{crop_probability * 100:.2f}%</b></div>
                <div class="track"><i style="width:{crop_bar:.2f}%"></i></div>
            </div>
            """)

            # The validator is a preliminary screening model, not a
            # reliable crop/disease authority. Do not let it block a
            # genuine leaf image from reaching the disease classifier.
            st.info(
                "Preliminary image screening completed. "
                "This score is only a screening signal; disease detection "
                "will still run so the model can evaluate the uploaded leaf."
            )

            if non_crop_probability >= VALIDATOR_THRESHOLD:
                st.warning(
                    "The image validator flagged this image as possibly "
                    "outside its training domain. The disease result may be "
                    "less reliable, so check that the photo clearly shows "
                    "a tomato, potato or bell pepper leaf."
                )

            # ------------------------------------------
            # STAGE 2: DISEASE DETECTION
            # ------------------------------------------

            with st.spinner("Detecting crop condition..."):

                disease_image = image.resize(
                    IMAGE_SIZE
                )

                disease_array = np.asarray(
                    disease_image,
                    dtype=np.float32
                )

                disease_array = np.expand_dims(
                    disease_array,
                    axis=0
                )

                predictions = disease_model.predict(
                    disease_array,
                    verbose=0
                )[0]

            predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)

            if (
                len(predictions) != len(class_names)
                or not np.all(np.isfinite(predictions))
                or np.any(predictions < 0)
            ):
                st.error(
                    "The model returned invalid prediction values or its "
                    "output does not match the class-name file."
                )
                st.stop()

            predicted_index = int(
                np.argmax(predictions)
            )

            predicted_class = class_names[predicted_index]

            confidence = float(
                predictions[predicted_index]
            )

            confidence_percent = confidence * 100

            ai_info = disease_info.get(predicted_class)

            if ai_info is None:
                st.error(
                    f"Unknown model class: {predicted_class}"
                )
                st.stop()

            # A selected demo label is explicitly manual. The raw AI output
            # remains unchanged and is still shown separately below.
            manual_correction_used = selected_demo_label != "Use model prediction"
            final_class = (
                selected_demo_label if manual_correction_used else predicted_class
            )
            display_info = disease_info[final_class]

            # ------------------------------------------
            # STAGE 3: DISPLAY RESULTS
            # ------------------------------------------

            section(
                "03 / ANALYSIS", "Diagnostic", "report.",
                "The raw output of the disease model for your uploaded leaf."
            )

            if ai_info["status"] == "Healthy":
                pill_class, pill_text = "ok", "✓ HEALTHY LEAF"
            elif ai_info["status"] == "Late Blight":
                pill_class, pill_text = "bad", "⚠ DISEASE DETECTED"
            else:
                pill_class, pill_text = "warn", "⚠ DISEASE DETECTED"

            ring_value = max(0.0, min(confidence, 1.0)) * 100

            r1, r2 = st.columns([1, 1.45], gap="large")

            with r1:
                st.image(image, caption="Analysed leaf", use_container_width=True)

            with r2:
                html(f"""
                <div class="lab">
                    <div class="lab-top">
                        <span class="status"><i></i>DETECTION COMPLETE</span>
                        <small>MODEL OUTPUT</small>
                    </div>
                    <div class="lab-main">
                        <div>
                            <small>Detected crop</small>
                            <div class="crop">🌿 AI model prediction: {ai_info['crop']}</div>
                            <small>Predicted condition</small>
                            <h3>AI predicted condition: {ai_info['status']}</h3>
                            <span class="pill {pill_class}">{pill_text}</span>
                        </div>
                        <div class="ring" style="--p:{ring_value:.2f}">
                            <div><b>{confidence_percent:.2f}%</b><small>CONFIDENCE</small></div>
                        </div>
                    </div>
                    <div class="facts">
                        <div><small>Crop-leaf score</small><b>{crop_probability * 100:.2f}%</b></div>
                        <div><small>Confidence threshold</small><b>{DISEASE_THRESHOLD * 100:.0f}%</b></div>
                        <div><small>Classes</small><b>{len(class_names)}</b></div>
                    </div>
                </div>
                """)

            st.write("")

            if manual_correction_used:
                st.info(
                    f"Demo-verified label (manual selection): "
                    f"{display_info['crop']} — {display_info['status']}. "
                    "This is a user-selected label, not the AI model's prediction."
                )
            else:
                st.caption("No manual correction selected; displaying the AI prediction.")

            if confidence < DISEASE_THRESHOLD:
                st.warning(
                    "Lower model confidence. Treat this as an uncertain "
                    "AI estimate; try a clearer, well-lit leaf photo."
                )
            else:
                st.info(
                    "The model produced this prediction with a higher "
                    "score. It is still an AI estimate, not a confirmed diagnosis."
                )

            # ------------------------------------------
            # STAGE 4: GUIDANCE
            # ------------------------------------------

            if is_kannada:
                sec_title, sec_accent = "ಲಕ್ಷಣಗಳು &", "ಸಲಹೆಗಳು"
                ov_title, ov_crop, ov_cond, ov_src = "ಅವಲೋಕನ", "ಬೆಳೆ", "ಸ್ಥಿತಿ", "ಮೂಲ"
                src_text = "ಕೈಯಾರೆ ಆಯ್ಕೆ" if manual_correction_used else "AI ಮಾದರಿ"
                sym_title, sym_text = "ಲಕ್ಷಣಗಳು", display_info["kn_symptoms"]
                gd_title, gd_text = "ಸಲಹೆಗಳು", display_info["kn_guidance"]
            else:
                sec_title, sec_accent = "Understanding the", "condition."
                ov_title, ov_crop, ov_cond, ov_src = "Overview", "Crop", "Condition", "Source"
                src_text = "Manual demo label" if manual_correction_used else "AI model"
                sym_title, sym_text = "Symptoms", display_info["symptoms"]
                gd_title, gd_text = "Prevention Guidance", display_info["guidance"]

            section(
                "04 / GUIDANCE", sec_title, sec_accent,
                "Practical information to help you decide what to check next."
            )

            html(f"""
            <div class="edit">
                <div class="panel overview">
                    <div class="ic">🌿</div>
                    <h4>{ov_title}</h4>
                    <div class="kv"><span>{ov_crop}</span><b>{display_info['crop']}</b></div>
                    <div class="kv"><span>{ov_cond}</span><b>{display_info['status']}</b></div>
                    <div class="kv"><span>{ov_src}</span><b>{src_text}</b></div>
                </div>
                <div class="stack">
                    <div class="panel symptoms">
                        <div class="ic">🔎</div>
                        <h4>{sym_title}</h4>
                        <p>{sym_text}</p>
                    </div>
                    <div class="panel guide">
                        <div class="ic">🛡️</div>
                        <h4>{gd_title}</h4>
                        <p>{gd_text}</p>
                    </div>
                </div>
            </div>
            """)

            st.write("")

            st.warning(
                "Important: a high confidence score does not guarantee a correct "
                "diagnosis. This model has previously confused healthy leaves with "
                "Late Blight. Do not use this result alone to decide treatment."
            )
            st.caption(
                "The top-three scores below are the model's actual outputs. "
                "A manual demo label, if selected, does not alter those scores "
                "or retrain the model."
            )

            # ------------------------------------------
            # STAGE 5: TOP THREE PREDICTIONS
            # ------------------------------------------

            section(
                "05 / RANKING", "Top 3", "predictions.",
                "The three highest-scoring classes, exactly as the model produced them."
            )

            top_indices = np.argsort(
                predictions
            )[-3:][::-1]

            cards = ""

            for rank, index in enumerate(top_indices, start=1):

                class_name = class_names[index]

                probability = float(
                    predictions[index]
                )

                candidate_info = disease_info.get(class_name, {})
                candidate_crop = candidate_info.get("crop", "Unknown crop")
                display_name = candidate_info.get("status", class_name)

                bar = max(0.0, min(probability, 1.0)) * 100
                first = " first" if rank == 1 else ""

                cards += f"""
                <div class="t3{first}">
                    <div class="rk">RANK 0{rank}</div>
                    <h5>{candidate_crop} — {display_name}</h5>
                    <div class="pc">{probability * 100:.2f}%</div>
                    <div class="track"><i style="width:{bar:.2f}%"></i></div>
                </div>
                """

            html(f'<div class="top3">{cards}</div>')

            # ------------------------------------------
            # STAGE 6: HISTORY
            # ------------------------------------------

            st.session_state.history.append({
                "AI crop prediction": ai_info["crop"],
                "AI condition prediction": ai_info["status"],
                "AI confidence": f"{confidence_percent:.2f}%",
                "Displayed demo label": display_info["status"],
                "Label source": "Manual verification" if manual_correction_used else "AI model"
            })

            # Display-only extras for the timeline (thumbnail and the time
            # this prediction was recorded). Not used by any model logic.
            thumb = image.copy()
            thumb.thumbnail((120, 120))
            thumb_buffer = io.BytesIO()
            thumb.save(thumb_buffer, format="JPEG", quality=80)
            st.session_state.history_extra.append({
                "thumb": base64.b64encode(thumb_buffer.getvalue()).decode("ascii"),
                "time": datetime.now().strftime("%d %b %Y, %H:%M"),
            })

    except Exception as error:

        st.error(
            f"Unable to process image: {error}"
        )


# ==================================================
# 9. PREDICTION HISTORY
# ==================================================

section(
    "06 / ACTIVITY", "Prediction", "history.",
    "Every prediction made during this session, newest first.",
    anchor="history"
)

if st.session_state.history:

    extras = st.session_state.history_extra
    items = ""

    for position in range(len(st.session_state.history) - 1, -1, -1):

        record = st.session_state.history[position]
        extra = extras[position] if position < len(extras) else {}

        if extra.get("thumb"):
            thumb_html = f'<img src="data:image/jpeg;base64,{extra["thumb"]}" alt="Uploaded leaf thumbnail">'
        else:
            thumb_html = '<div class="ph">🍃</div>'

        when = f' · {extra["time"]}' if extra.get("time") else ""

        items += f"""
        <div class="tl-item">
            {thumb_html}
            <div class="tl-body">
                <b>{record['AI crop prediction']} — {record['AI condition prediction']}</b>
                <span>Displayed: {record['Displayed demo label']}{when}</span>
            </div>
            <span class="tl-tag">{record['Label source'].upper()}</span>
            <div class="tl-conf">{record['AI confidence']}</div>
        </div>
        """

    html(f'<div class="tl">{items}</div>')

    with st.expander("View history as table"):
        st.dataframe(
            st.session_state.history,
            use_container_width=True
        )

    if st.button("Clear History"):
        st.session_state.history = []
        st.session_state.history_extra = []
        st.rerun()

else:
    html('<div class="empty">🍃 No predictions yet.</div>')


# ==================================================
# 10. FOOTER
# ==================================================

html("""
<div class="foot">
    <span><b>CropCare AI</b> &nbsp;|&nbsp; AI-Based Crop Disease Detection</span>
    <span>Educational Project</span>
</div>
""")
