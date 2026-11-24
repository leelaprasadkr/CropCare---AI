import base64
import io
import json
import math
from datetime import datetime
from html import escape as esc
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
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 1A. DESIGN SYSTEM: "NIGHT GREENHOUSE" (UI ONLY)
# ==================================================
# Dark botanical app shell, glass surfaces, ivory "paper" for the report data,
# lime used only for actions and live signals. Every Streamlit element that has
# changed names between versions is targeted with both its old and new test id.

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&family=Noto+Sans+Kannada:wght@400;500;600&display=swap');

:root {
    --bg: #03100b; --forest: #0d3b2c; --emerald: #12a36b; --mint: #2fd18b; --lime: #c6f432;
    --ivory: #f6f1e4; --paper: #fbf8f0; --ink: #10261d; --sage: #a9bfa8; --muted: #8fa896;
    --muted-ink: #55695d;
    --line: rgba(255,255,255,.10); --line-ink: rgba(13,59,44,.14);
    --glass: rgba(255,255,255,.055); --glass-2: rgba(255,255,255,.09);
    --amber: #f0b429; --rust: #e0643a;
    --r: 26px; --r-sm: 16px;
    --serif: 'Fraunces', Georgia, serif;
    --sans: 'Manrope', 'Noto Sans Kannada', system-ui, sans-serif;
    --ease: cubic-bezier(.22,.8,.3,1);
    --lift: 0 40px 80px -30px rgba(0,0,0,.75);
}

/* ---------- App shell ---------- */
html { scroll-behavior: smooth; }
html, body, .stApp, .stApp button, .stApp input, .stApp label, .stMarkdown { font-family: var(--sans); -webkit-font-smoothing: antialiased; }
.stApp {
    color: var(--ivory);
    background:
        radial-gradient(1100px 620px at 88% -8%, rgba(47,209,139,.17), transparent 60%),
        radial-gradient(900px 640px at -4% 46%, rgba(198,244,50,.07), transparent 60%),
        linear-gradient(180deg, #03100b, #05180f 55%, #03100b);
}
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 1240px !important; padding: 1.2rem 1.6rem 7rem !important; }
::selection { background: var(--lime); color: #06251a; }
a:focus-visible, button:focus-visible { outline: 3px solid var(--lime); outline-offset: 3px; }
.anchor { scroll-margin-top: 1rem; }

@keyframes enter { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: none; } }
@keyframes sway { 0%,100% { transform: rotate(-1.4deg); } 50% { transform: rotate(1.6deg); } }
@keyframes scanv { 0% { transform: translateY(-100%); } 100% { transform: translateY(560%); } }
@keyframes sweep { 0%,100% { transform: translateY(0); } 50% { transform: translateY(80px); } }
@keyframes pulse { 0%,100% { opacity: .35; } 50% { opacity: 1; } }
@keyframes halo { 0%,100% { box-shadow: 0 0 0 0 rgba(198,244,50,.5); } 50% { box-shadow: 0 0 0 10px rgba(198,244,50,0); } }

/* ---------- Sidebar: control centre ---------- */
[data-testid="stSidebar"] { background: linear-gradient(195deg, #041a13, #06251a 58%, #03100b) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebar"][aria-expanded="true"] { width: 292px !important; min-width: 292px !important; }
[data-testid="stSidebarContent"], [data-testid="stSidebar"] > div:first-child { padding: .4rem .3rem; }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapsedControl"] button, [data-testid="collapsedControl"] button { color: var(--lime) !important; }
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { background: rgba(6,37,26,.85); border: 1px solid var(--line); border-radius: 14px; backdrop-filter: blur(14px); margin: .6rem; }
.s-brand { display: flex; align-items: center; gap: .75rem; padding: .6rem .4rem 1.3rem; }
.s-brand svg { width: 34px; height: 34px; filter: drop-shadow(0 6px 14px rgba(47,209,139,.55)); }
.s-brand b { display: block; font-family: var(--serif); font-size: 1.35rem; font-weight: 600; letter-spacing: -.02em; color: #fff; line-height: 1.1; }
.s-brand b i { font-style: normal; color: var(--lime); }
.s-brand span { font-size: .76rem; color: var(--muted); }
.s-label { font-size: .76rem; font-weight: 700; color: var(--muted); margin: 1.2rem .4rem .5rem; letter-spacing: .02em; }
.snav { display: flex; align-items: center; gap: .8rem; padding: .7rem .9rem; margin-bottom: .2rem; border-radius: 14px; color: #dfeadb !important; text-decoration: none !important; font-weight: 600; font-size: .92rem; transition: background .25s var(--ease), transform .25s var(--ease); }
.snav .ico { width: 19px; height: 19px; color: var(--mint); flex: none; }
.snav:hover { background: var(--glass-2); transform: translateX(3px); color: #fff !important; }
.snav:hover .ico { color: var(--lime); }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background: rgba(255,255,255,.07) !important; border: 1px solid var(--line) !important; border-radius: 14px !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: var(--ivory) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: var(--lime) !important; }
.s-card { margin: 1rem .1rem 0; padding: 1.1rem 1.1rem 1rem; border-radius: 20px; background: var(--glass); border: 1px solid var(--line); box-shadow: inset 0 1px 0 rgba(255,255,255,.08); }
.s-card h6 { margin: 0 0 .8rem; font-size: .8rem; font-weight: 700; color: var(--muted); }
.s-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; }
.s-grid div { text-align: left; }
.s-grid b { display: block; font-family: var(--serif); font-size: 1.7rem; font-weight: 600; color: #fff; line-height: 1; font-variant-numeric: tabular-nums; }
.s-grid small { font-size: .72rem; color: var(--muted); }
.s-note { margin: 1rem .4rem 0; font-size: .76rem; line-height: 1.55; color: var(--muted); }

/* ---------- Floating dock ---------- */
.dock { position: fixed; left: 50%; bottom: 20px; transform: translateX(-50%); z-index: 99999; display: flex; gap: .2rem; padding: .4rem; border-radius: 999px; background: rgba(5,26,19,.72); backdrop-filter: blur(22px) saturate(1.6); -webkit-backdrop-filter: blur(22px) saturate(1.6); border: 1px solid rgba(255,255,255,.14); box-shadow: 0 24px 60px -12px rgba(0,0,0,.7), inset 0 1px 0 rgba(255,255,255,.12); }
.dock a { display: flex; align-items: center; gap: .5rem; padding: .62rem 1rem; border-radius: 999px; color: #dfeadb !important; text-decoration: none !important; font-size: .84rem; font-weight: 700; transition: background .25s var(--ease), color .25s var(--ease); }
.dock a .ico { width: 18px; height: 18px; }
.dock a:hover { background: var(--lime); color: #06251a !important; }
.dock a.primary { background: var(--lime); color: #06251a !important; }
.dock a.primary:hover { background: #fff; }

/* ---------- Cinematic hero ---------- */
.hero2 { position: relative; isolation: isolate; overflow: hidden; display: grid; grid-template-columns: minmax(0, 1.12fr) minmax(0, .88fr); grid-template-rows: 1fr auto; gap: 1.6rem 2rem; min-height: 680px; padding: clamp(1.6rem, 4vw, 3.4rem); border-radius: 38px; border: 1px solid var(--line); box-shadow: var(--lift); animation: enter 1s var(--ease) both; background: #03100b; }
.scene { position: absolute; inset: 0; z-index: -2; }
.scene svg, .scene img { width: 100%; height: 100%; display: block; object-fit: cover; }
.scene .sway { transform-origin: 0 0; animation: sway 14s ease-in-out infinite; will-change: transform; }
.hero2::after { content: ""; position: absolute; inset: 0; z-index: -1; pointer-events: none; background: linear-gradient(90deg, rgba(3,16,11,.9) 0%, rgba(3,16,11,.6) 40%, transparent 74%), linear-gradient(0deg, rgba(3,16,11,.85), transparent 42%); }
.hero2-copy { grid-column: 1; align-self: center; padding-top: 1rem; }
.badge { display: inline-flex; align-items: center; gap: .6rem; padding: .45rem 1rem; border-radius: 999px; font-size: .82rem; font-weight: 700; color: var(--lime); background: rgba(198,244,50,.07); border: 1px solid rgba(198,244,50,.3); backdrop-filter: blur(10px); }
.badge i { width: 7px; height: 7px; border-radius: 50%; background: var(--lime); animation: halo 2.6s ease-in-out infinite; }
.hero2 h1 { font-family: var(--serif); font-weight: 600; font-size: clamp(2.7rem, 5.6vw, 4.9rem); line-height: .98; letter-spacing: -.04em; color: #fbf8ee; margin: 1.3rem 0 1.3rem; padding: 0; font-variation-settings: 'opsz' 144; text-wrap: balance; }
.hero2 h1 span { color: var(--lime); }
.hero2 p { color: rgba(236,244,232,.8); font-size: 1.06rem; line-height: 1.75; max-width: 540px; margin: 0 0 2rem; }
.actions { display: flex; flex-wrap: wrap; gap: .8rem; }
.cta { display: inline-flex; align-items: center; gap: .7rem; background: var(--lime); color: #06251a !important; font-weight: 800; text-decoration: none !important; padding: 1rem 1.8rem; border-radius: 999px; box-shadow: 0 14px 40px -8px rgba(198,244,50,.55), inset 0 1px 0 rgba(255,255,255,.6); transition: transform .3s var(--ease), box-shadow .3s var(--ease); }
.cta:hover { transform: translateY(-3px); box-shadow: 0 22px 50px -8px rgba(198,244,50,.7), inset 0 1px 0 rgba(255,255,255,.6); }
.ghost { display: inline-flex; align-items: center; color: #eaf3e6 !important; font-weight: 700; text-decoration: none !important; padding: 1rem 1.6rem; border-radius: 999px; border: 1px solid rgba(255,255,255,.22); background: var(--glass); backdrop-filter: blur(12px); transition: background .25s var(--ease); }
.ghost:hover { background: var(--glass-2); }
.hud { grid-column: 2; grid-row: 1; align-self: end; justify-self: end; width: min(330px, 100%); padding: 1.3rem 1.4rem; border-radius: 24px; background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.16); backdrop-filter: blur(22px) saturate(1.5); -webkit-backdrop-filter: blur(22px) saturate(1.5); box-shadow: 0 30px 60px -20px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.18); }
.hud h6 { margin: 0 0 .9rem; font-size: .8rem; font-weight: 700; color: var(--lime); display: flex; align-items: center; gap: .5rem; }
.hud h6::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--lime); box-shadow: 0 0 10px var(--lime); }
.hud div { display: flex; justify-content: space-between; gap: 1rem; padding: .55rem 0; border-top: 1px solid rgba(255,255,255,.1); font-size: .86rem; }
.hud div span { color: var(--sage); }
.hud div b { color: #fff; text-align: right; font-weight: 700; }
.hud small { display: block; margin-top: .7rem; font-size: .72rem; color: var(--muted); }
.hero2-stats { grid-column: 1 / -1; display: grid; grid-template-columns: repeat(3, 1fr); border-radius: 22px; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12); backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); overflow: hidden; }
.hero2-stats div { padding: 1.1rem 1.6rem; border-right: 1px solid rgba(255,255,255,.1); }
.hero2-stats div:last-child { border-right: none; }
.hero2-stats b { display: block; font-family: var(--serif); font-weight: 600; font-size: 2.4rem; color: #fff; line-height: 1; letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
.hero2-stats small { display: block; margin-top: .4rem; font-size: .82rem; font-weight: 600; color: var(--sage); }

/* ---------- Section headings ---------- */
.sec { margin: 5.2rem 0 1.7rem; }
.eyebrow { display: inline-flex; align-items: center; gap: .7rem; font-size: .86rem; font-weight: 700; color: var(--lime); }
.eyebrow::before { content: ""; width: 30px; height: 2px; background: var(--lime); border-radius: 2px; }
.sec h2 { font-family: var(--serif); font-weight: 600; font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.02; letter-spacing: -.035em; margin: .8rem 0 .7rem; padding: 0; color: #fbf8ee; font-variation-settings: 'opsz' 144; }
.sec p { max-width: 520px; margin: 0; color: var(--muted); font-size: 1rem; line-height: 1.65; }
.subhead { margin: 2.4rem 0 1rem; font-family: var(--serif); font-size: 1.5rem; font-weight: 600; letter-spacing: -.02em; color: #fbf8ee; display: flex; align-items: center; gap: .9rem; }
.subhead::after { content: ""; flex: 1; height: 1px; background: var(--line); }

/* ---------- Process rail ---------- */
.rail { position: relative; display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
.rail::before { content: ""; position: absolute; left: 12%; right: 12%; top: 2.45rem; border-top: 1px dashed rgba(198,244,50,.4); }
.node { position: relative; padding: 1.5rem 1.6rem 1.7rem; border-radius: var(--r); background: var(--glass); border: 1px solid var(--line); backdrop-filter: blur(14px); box-shadow: inset 0 1px 0 rgba(255,255,255,.08); transition: transform .4s var(--ease), border-color .4s var(--ease); }
.node:hover { transform: translateY(-5px); border-color: rgba(198,244,50,.4); }
.node .ic { position: relative; z-index: 1; display: grid; place-items: center; width: 54px; height: 54px; border-radius: 18px; margin-bottom: 1.3rem; color: #06251a; background: linear-gradient(140deg, var(--lime), var(--mint)); box-shadow: 0 12px 30px -6px rgba(198,244,50,.5); }
.node .ic .ico { width: 24px; height: 24px; }
.node h3 { font-family: var(--serif); font-size: 1.35rem; font-weight: 600; letter-spacing: -.02em; margin: 0 0 .5rem; padding: 0; color: #fff; }
.node p { margin: 0; font-size: .95rem; line-height: 1.65; color: var(--muted); }

/* ---------- Field index (crops) ---------- */
.crops { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
.crop { position: relative; overflow: hidden; padding: 1.5rem 1.5rem 1.6rem; border-radius: var(--r); border: 1px solid var(--line); background: radial-gradient(240px 170px at 100% 0%, color-mix(in srgb, var(--c, #12a36b) 38%, transparent), transparent 72%), var(--glass); box-shadow: inset 0 1px 0 rgba(255,255,255,.08); transition: transform .45s var(--ease), border-color .45s var(--ease); }
.crop:hover { transform: translateY(-6px); border-color: color-mix(in srgb, var(--c, #12a36b) 70%, white); }
.crop-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; margin-bottom: 1.5rem; }
.crop-top .badge-ic { display: grid; place-items: center; width: 52px; height: 52px; border-radius: 17px; background: rgba(255,255,255,.08); border: 1px solid var(--line); }
.crop-top svg { width: 30px; height: 30px; }
.crop-top span { font-family: var(--serif); font-size: 2.3rem; font-weight: 600; line-height: 1; color: #fff; letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
.crop-top span small { display: block; font-family: var(--sans); font-size: .74rem; font-weight: 600; color: var(--muted); letter-spacing: 0; margin-top: .25rem; text-align: right; }
.crop h4 { font-family: var(--serif); font-size: 1.5rem; font-weight: 600; letter-spacing: -.02em; margin: 0 0 .8rem; padding: 0; color: #fff; }
.chips { display: flex; flex-wrap: wrap; gap: .35rem; }
.chip { font-size: .76rem; font-weight: 600; padding: .27rem .7rem; border-radius: 999px; background: rgba(255,255,255,.08); color: #d2e0cf; }
.chip.ok { background: var(--lime); color: #06251a; }

/* ---------- Analysis workspace ---------- */
[data-testid="stHorizontalBlock"]:has(.ws-left) { align-items: stretch !important; gap: 1.2rem !important; padding: 1.3rem; border-radius: 34px; border: 1px solid rgba(255,255,255,.13); background: radial-gradient(520px 260px at 0% 0%, rgba(47,209,139,.14), transparent 70%), linear-gradient(165deg, rgba(255,255,255,.08), rgba(255,255,255,.02)); box-shadow: var(--lift), inset 0 1px 0 rgba(255,255,255,.14); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); }
[data-testid="stColumn"]:has(.ws-right), [data-testid="column"]:has(.ws-right) { padding: 1.4rem 1.4rem 1.2rem; border-radius: 26px; background: rgba(0,0,0,.3); border: 1px solid var(--line); }
.stepper { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; margin: 0 0 1rem; }
.stepper .st { display: flex; align-items: center; gap: .55rem; padding: .4rem .9rem .4rem .45rem; border-radius: 999px; font-size: .82rem; font-weight: 700; color: var(--muted); background: rgba(255,255,255,.04); border: 1px solid var(--line); transition: all .4s var(--ease); }
.stepper .st i { display: grid; place-items: center; width: 22px; height: 22px; border-radius: 50%; font-style: normal; font-size: .72rem; background: rgba(255,255,255,.1); }
.stepper .st.now { color: #fff; border-color: rgba(198,244,50,.5); background: rgba(198,244,50,.1); }
.stepper .st.now i { background: var(--lime); color: #06251a; animation: halo 2.4s ease-in-out infinite; }
.stepper .st.done { color: var(--lime); }
.stepper .st.done i { background: var(--mint); color: #06251a; }
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"] {
    flex-direction: column; justify-content: center; align-items: center; gap: .9rem; min-height: 330px; border: 1px solid var(--line); border-radius: 24px; transition: all .35s var(--ease);
    background:
        linear-gradient(var(--lime), var(--lime)) left 14px top 14px / 26px 2px no-repeat, linear-gradient(var(--lime), var(--lime)) left 14px top 14px / 2px 26px no-repeat,
        linear-gradient(var(--lime), var(--lime)) right 14px top 14px / 26px 2px no-repeat, linear-gradient(var(--lime), var(--lime)) right 14px top 14px / 2px 26px no-repeat,
        linear-gradient(var(--lime), var(--lime)) left 14px bottom 14px / 26px 2px no-repeat, linear-gradient(var(--lime), var(--lime)) left 14px bottom 14px / 2px 26px no-repeat,
        linear-gradient(var(--lime), var(--lime)) right 14px bottom 14px / 26px 2px no-repeat, linear-gradient(var(--lime), var(--lime)) right 14px bottom 14px / 2px 26px no-repeat,
        radial-gradient(360px 200px at 50% 0%, rgba(198,244,50,.12), transparent 70%), rgba(0,0,0,.28);
}
[data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploadDropzone"]:hover { border-color: rgba(198,244,50,.55); box-shadow: 0 0 60px -10px rgba(198,244,50,.35); transform: translateY(-2px); }
[data-testid="stFileUploaderDropzoneInstructions"], [data-testid="stFileDropzoneInstructions"] { display: none !important; }
[data-testid="stFileUploaderDropzone"]::before, [data-testid="stFileUploadDropzone"]::before {
    content: "Drop your leaf image here"; display: block; text-align: center; font-family: var(--serif); font-size: 1.8rem; font-weight: 600; letter-spacing: -.02em; color: #fff; padding-top: 86px;
    background: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><circle cx='32' cy='32' r='30' fill='%23c6f432'/><path d='M32 44V21M22.5 30.5L32 21l9.5 9.5' stroke='%2306251a' stroke-width='4' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>") no-repeat top center / 64px;
}
[data-testid="stFileUploaderDropzone"]::after, [data-testid="stFileUploadDropzone"]::after { content: "JPG, JPEG or PNG"; font-size: .85rem; color: var(--muted); }
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploadDropzone"] button { background: rgba(255,255,255,.1) !important; color: #fff !important; border: 1px solid rgba(255,255,255,.22) !important; border-radius: 999px !important; padding: .6rem 1.6rem; font-weight: 700; backdrop-filter: blur(10px); transition: all .25s var(--ease); }
[data-testid="stFileUploaderDropzone"] button:hover, [data-testid="stFileUploadDropzone"] button:hover { background: var(--lime) !important; color: #06251a !important; border-color: var(--lime) !important; }
[data-testid="stFileUploaderFile"], [data-testid="stUploadedFile"] { color: var(--ivory); }
[data-testid="stFileUploaderFile"] *, [data-testid="stUploadedFile"] * { color: var(--ivory) !important; }

.viewport { position: relative; overflow: hidden; border-radius: 22px; background: #000; border: 1px solid var(--line); box-shadow: var(--lift); margin-top: 1rem; }
.viewport img { display: block; width: 100%; max-height: 470px; object-fit: cover; }
.vc { position: absolute; width: 28px; height: 28px; border: 0 solid var(--lime); pointer-events: none; }
.vc.tl { left: 14px; top: 14px; border-width: 2px 0 0 2px; border-top-left-radius: 8px; }
.vc.tr { right: 14px; top: 14px; border-width: 2px 2px 0 0; border-top-right-radius: 8px; }
.vc.bl { left: 14px; bottom: 54px; border-width: 0 0 2px 2px; border-bottom-left-radius: 8px; }
.vc.br { right: 14px; bottom: 54px; border-width: 0 2px 2px 0; border-bottom-right-radius: 8px; }
.scanband { position: absolute; left: 0; right: 0; top: 0; height: 18%; pointer-events: none; background: linear-gradient(180deg, transparent, rgba(198,244,50,.22) 80%, rgba(198,244,50,.6)); border-bottom: 1px solid var(--lime); animation: scanv 3.4s ease-in-out infinite; }
.viewport.still .scanband { display: none; }
.vp-meta { position: absolute; left: 0; right: 0; bottom: 0; display: flex; justify-content: space-between; gap: 1rem; padding: .8rem 1.1rem; font-size: .8rem; font-weight: 600; color: #e9f1e6; background: rgba(3,16,11,.62); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); border-top: 1px solid rgba(255,255,255,.12); }
.console-head { display: flex; align-items: center; justify-content: space-between; gap: .8rem; margin-bottom: 1.1rem; }
.console-head h4 { font-family: var(--serif); font-size: 1.45rem; font-weight: 600; letter-spacing: -.02em; margin: 0; padding: 0; color: #fff; }
.live { display: inline-flex; align-items: center; gap: .5rem; font-size: .78rem; font-weight: 700; padding: .3rem .8rem; border-radius: 999px; color: var(--amber); background: rgba(240,180,41,.1); border: 1px solid rgba(240,180,41,.3); }
.live i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; animation: pulse 1.6s ease-in-out infinite; }
.live.ready { color: var(--lime); background: rgba(198,244,50,.1); border-color: rgba(198,244,50,.35); }
.tips { margin: 0; padding: 0; list-style: none; }
.tips li { display: flex; gap: .8rem; padding: .85rem 0; border-top: 1px solid var(--line); font-size: .93rem; line-height: 1.55; color: #cfdccb; }
.tips li:first-child { border-top: none; }
.tips li::before { content: ""; flex: none; width: 8px; height: 8px; margin-top: .5rem; border-radius: 50%; background: var(--lime); box-shadow: 0 0 10px rgba(198,244,50,.8); }
.demo-h { font-family: var(--serif); font-size: 1.15rem; font-weight: 600; color: #fff; margin: 0 0 .3rem; }

/* ---------- Native widgets on dark ---------- */
.stButton > button, [data-testid="stBaseButton-secondary"] { border-radius: 999px; font-weight: 700; padding: .7rem 1.5rem; color: var(--ivory); border: 1px solid rgba(255,255,255,.2); background: var(--glass-2); backdrop-filter: blur(12px); transition: all .25s var(--ease); }
.stButton > button:hover, [data-testid="stBaseButton-secondary"]:hover { border-color: var(--lime); color: var(--lime); transform: translateY(-2px); }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] { background: linear-gradient(135deg, #d6ff4a, #9fe870) !important; color: #06251a !important; border: none !important; padding: 1.1rem 1.4rem; font-size: 1.08rem; box-shadow: 0 18px 44px -10px rgba(198,244,50,.6), inset 0 1px 0 rgba(255,255,255,.65); }
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { transform: translateY(-3px); box-shadow: 0 26px 56px -10px rgba(198,244,50,.8), inset 0 1px 0 rgba(255,255,255,.65); color: #06251a !important; }
.stButton > button p, [data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-secondary"] p { color: inherit !important; font-weight: 800; }
[data-testid="stMain"] [data-baseweb="select"] > div, .stMain [data-baseweb="select"] > div, section.main [data-baseweb="select"] > div { border-radius: 14px; background: rgba(255,255,255,.07); border: 1px solid var(--line); }
[data-testid="stMain"] [data-baseweb="select"] *, .stMain [data-baseweb="select"] *, section.main [data-baseweb="select"] * { color: var(--ivory) !important; }
[data-baseweb="select"] svg { fill: var(--lime) !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid var(--line) !important; border-radius: 20px !important; background: rgba(255,255,255,.04); }
[data-testid="stAlert"] { border-radius: 18px; border: 1px solid var(--line); border-left: 4px solid var(--lime); background: rgba(255,255,255,.06) !important; backdrop-filter: blur(12px); }
[data-testid="stAlert"] [data-baseweb="notification"] { background: transparent !important; }
[data-testid="stAlert"] *, [data-testid="stAlertContainer"] * { color: #e7f0e3 !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
[data-testid="stExpander"] details { border-radius: 18px; border: 1px solid var(--line); background: var(--glass); }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--ivory) !important; }
[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }

/* ---------- Loading ---------- */
.loader { display: flex; align-items: center; gap: 1.5rem; padding: 1.5rem 1.8rem; margin: 1.4rem 0; border-radius: 28px; color: #e9f1e6; border: 1px solid rgba(198,244,50,.2); background: radial-gradient(360px 200px at 0% 0%, rgba(47,209,139,.28), transparent 70%), linear-gradient(150deg, #031610, var(--forest)); box-shadow: var(--lift); }
.scan { position: relative; width: 84px; height: 84px; flex: none; border-radius: var(--r-sm); background: var(--glass); border: 1px solid var(--line); overflow: hidden; }
.scan svg { width: 100%; height: 100%; display: block; padding: 10px; box-sizing: border-box; }
.scan::after { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px; background: var(--lime); box-shadow: 0 0 18px 2px var(--lime); animation: sweep 1.6s ease-in-out infinite; }
.loader b { display: block; font-family: var(--serif); font-size: 1.4rem; font-weight: 600; color: #fff; margin-bottom: .25rem; }
.loader span { color: var(--sage); font-size: .92rem; }
.dots { display: flex; gap: .4rem; margin-top: .8rem; }
.dots i { width: 32px; height: 5px; border-radius: 99px; background: rgba(255,255,255,.18); }
.dots i.on { background: var(--lime); animation: pulse 1.2s ease-in-out infinite; }
.dots i.done { background: var(--lime); }

/* ---------- Results dashboard ---------- */
.rep { display: grid; grid-template-columns: minmax(0, .85fr) minmax(0, 1.35fr); gap: 1rem; animation: enter .8s var(--ease) both; }
.rep .viewport { margin-top: 0; height: 100%; }
.rep .viewport img { height: 100%; max-height: none; min-height: 340px; }
.verdict { --accent: var(--lime); position: relative; overflow: hidden; isolation: isolate; padding: 2.1rem 2.2rem; border-radius: 32px; color: #eef5ea; border: 1px solid var(--line); background: radial-gradient(520px 320px at 100% 0%, color-mix(in srgb, var(--accent) 30%, transparent), transparent 70%), linear-gradient(155deg, #031610, #0a3024); box-shadow: var(--lift); }
.verdict::before { content: ""; position: absolute; inset: 0; z-index: -1; background-image: linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px); background-size: 36px 36px; -webkit-mask-image: linear-gradient(160deg, #000, transparent 70%); mask-image: linear-gradient(160deg, #000, transparent 70%); }
.verdict::after { content: ""; position: absolute; left: 2.2rem; top: 0; width: 76px; height: 4px; border-radius: 0 0 6px 6px; background: var(--accent); box-shadow: 0 0 26px 3px var(--accent); }
.verdict.warn { --accent: var(--amber); }
.verdict.bad { --accent: var(--rust); }
.v-top { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; }
.status { font-size: .84rem; font-weight: 700; color: var(--accent); display: inline-flex; align-items: center; gap: .55rem; }
.status i { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 12px var(--accent); }
.verdict small { font-size: .82rem; font-weight: 600; color: var(--sage); }
.v-main { display: grid; grid-template-columns: 1fr auto; gap: 1.6rem; align-items: center; margin: 1.5rem 0 .5rem; }
.crop-line { font-size: 1.05rem; font-weight: 700; color: #fff; margin: .25rem 0 1.1rem; }
.verdict h3 { font-family: var(--serif); font-weight: 600; font-size: 2.35rem; line-height: 1.05; letter-spacing: -.035em; margin: .35rem 0 1.2rem; color: #fff; padding: 0; }
.pill { display: inline-block; padding: .42rem 1.05rem; border-radius: 999px; font-weight: 800; font-size: .82rem; }
.pill.ok { background: var(--lime); color: #06251a; box-shadow: 0 6px 22px -4px rgba(198,244,50,.6); }
.pill.warn { background: var(--amber); color: #2b1c00; box-shadow: 0 6px 22px -4px rgba(240,180,41,.6); }
.pill.bad { background: var(--rust); color: #fff; box-shadow: 0 6px 22px -4px rgba(224,100,58,.65); }
.ring { --p: 0; width: 184px; height: 184px; border-radius: 50%; display: grid; place-items: center; background: conic-gradient(var(--accent) calc(var(--p) * 1%), rgba(255,255,255,.1) 0); filter: drop-shadow(0 0 24px color-mix(in srgb, var(--accent) 55%, transparent)); }
.ring div { width: 150px; height: 150px; border-radius: 50%; background: radial-gradient(circle at 30% 25%, #0f3a2b, #04170f); display: flex; flex-direction: column; align-items: center; justify-content: center; }
.ring b { font-family: var(--serif); font-weight: 600; font-size: 2.2rem; letter-spacing: -.02em; color: #fff; line-height: 1; font-variant-numeric: tabular-nums; }
.gauge { position: relative; height: 8px; background: rgba(255,255,255,.12); border-radius: 99px; margin: 1.7rem 0 .5rem; }
.gauge i { display: block; height: 100%; background: var(--accent); border-radius: 99px; box-shadow: 0 0 14px var(--accent); }
.gauge u { position: absolute; top: -6px; width: 2px; height: 20px; border-radius: 2px; background: #fff; text-decoration: none; }
.gauge-cap { display: flex; justify-content: space-between; gap: .5rem; font-size: .78rem; color: var(--sage); }
.notice { margin: 1.3rem 0 0; font-size: .86rem; line-height: 1.55; color: rgba(233,241,230,.78); border-top: 1px solid rgba(255,255,255,.12); padding-top: 1rem; }
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }
.metric { padding: 1.3rem 1.4rem; border-radius: 22px; background: var(--glass); border: 1px solid var(--line); box-shadow: inset 0 1px 0 rgba(255,255,255,.08); }
.metric small { font-size: .8rem; font-weight: 600; color: var(--muted); }
.metric b { display: block; margin: .35rem 0 .2rem; font-family: var(--serif); font-size: 2.1rem; font-weight: 600; letter-spacing: -.03em; color: #fff; font-variant-numeric: tabular-nums; line-height: 1.1; }
.metric b.sm { font-size: 1.25rem; padding: .55rem 0; letter-spacing: -.01em; }
.metric span { font-size: .78rem; color: var(--sage); }
.screen { display: grid; grid-template-columns: auto 1fr; gap: 1.4rem; align-items: center; padding: 1.1rem 1.6rem; border-radius: 22px; background: var(--glass); border: 1px solid var(--line); }
.screen small { font-size: .82rem; font-weight: 600; color: var(--muted); }
.screen b { font-family: var(--serif); font-size: 1.9rem; font-weight: 600; letter-spacing: -.02em; color: #fff; display: block; font-variant-numeric: tabular-nums; }
.track { height: 8px; background: rgba(255,255,255,.12); border-radius: 99px; overflow: hidden; }
.track i { display: block; height: 100%; background: linear-gradient(90deg, var(--emerald), var(--lime)); border-radius: 99px; }
.rep-body { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 1rem; align-items: start; }
.paper { padding: 1.9rem 2rem; border-radius: 30px; background: var(--paper); color: var(--ink); box-shadow: var(--lift); }
.paper h4 { font-family: var(--serif); font-size: 1.5rem; font-weight: 600; letter-spacing: -.025em; margin: 0 0 .3rem; padding: 0; color: var(--ink); }
.paper > p { margin: 0 0 1.5rem; font-size: .9rem; color: var(--muted-ink); }
.bars { position: relative; display: grid; gap: 1.3rem; }
.bar-row .lbl { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; margin-bottom: .45rem; }
.bar-row .lbl span { font-weight: 700; font-size: .95rem; color: var(--ink); }
.bar-row .lbl span em { font-style: normal; display: inline-block; width: 1.5rem; color: var(--emerald); font-weight: 800; }
.bar-row .lbl b { font-family: var(--serif); font-size: 1.5rem; font-weight: 600; letter-spacing: -.02em; color: var(--ink); font-variant-numeric: tabular-nums; }
.bar { height: 14px; border-radius: 99px; background: repeating-linear-gradient(90deg, rgba(13,59,44,.08) 0, rgba(13,59,44,.08) calc(25% - 1px), rgba(13,59,44,.2) calc(25% - 1px), rgba(13,59,44,.2) 25%); overflow: hidden; }
.bar i { display: block; height: 100%; border-radius: 99px; background: #9cb9a6; }
.bar-row.first .bar i { background: linear-gradient(90deg, var(--emerald), #8fe14a); box-shadow: 0 0 16px rgba(18,163,107,.4); }
.scale { display: flex; justify-content: space-between; margin-top: 1.1rem; font-size: .74rem; color: var(--muted-ink); }
.stack { display: grid; gap: 1rem; }
.panel { padding: 1.6rem 1.8rem; border-radius: 26px; background: var(--glass); border: 1px solid var(--line); box-shadow: inset 0 1px 0 rgba(255,255,255,.08); }
.panel h4 { font-family: var(--serif); font-size: 1.35rem; font-weight: 600; letter-spacing: -.02em; margin: 0 0 .7rem; padding: 0; color: #fff; }
.panel p { margin: 0; font-size: 1.02rem; line-height: 1.75; color: #dbe7d7; }
.panel .kv { display: flex; justify-content: space-between; gap: 1rem; padding: .7rem 0; border-top: 1px solid var(--line); font-size: .95rem; color: var(--sage); }
.panel .kv b { color: #fff; text-align: right; }
.panel.symptoms { border-left: 5px solid var(--amber); }
.panel.guide { background: linear-gradient(135deg, #d6ff4a, #8fe14a); border: none; box-shadow: 0 26px 60px -20px rgba(198,244,50,.45); }
.panel.guide h4, .panel.guide p { color: #06251a; }

/* ---------- Session log ---------- */
.hsum { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem; }
.hsum div { padding: 1.1rem 1.4rem; border-radius: 20px; background: var(--glass); border: 1px solid var(--line); }
.hsum b { display: block; font-family: var(--serif); font-size: 2rem; font-weight: 600; color: #fff; line-height: 1; letter-spacing: -.02em; }
.hsum small { font-size: .8rem; font-weight: 600; color: var(--muted); }
.ledger { display: grid; gap: .7rem; }
.row { display: grid; grid-template-columns: 62px minmax(0, 1fr) auto 150px; align-items: center; gap: 1.2rem; padding: .85rem 1.3rem .85rem .85rem; border-radius: 22px; background: var(--glass); border: 1px solid var(--line); transition: transform .3s var(--ease), border-color .3s var(--ease); }
.row:hover { transform: translateX(5px); border-color: rgba(198,244,50,.4); }
.row img, .row .ph { width: 62px; height: 62px; border-radius: 16px; object-fit: cover; background: rgba(255,255,255,.08); display: grid; place-items: center; }
.row .ph svg { width: 28px; height: 28px; }
.row .who b { display: block; font-size: 1rem; color: #fff; }
.row .who span { display: block; margin-top: .15rem; font-size: .8rem; color: var(--muted); }
.row .tag { font-size: .75rem; font-weight: 700; padding: .28rem .75rem; border-radius: 99px; background: rgba(255,255,255,.08); color: #d2e0cf; white-space: nowrap; }
.row .cf b { display: block; text-align: right; font-family: var(--serif); font-size: 1.4rem; font-weight: 600; color: var(--lime); letter-spacing: -.02em; font-variant-numeric: tabular-nums; margin-bottom: .35rem; }
.empty { text-align: center; padding: 2.8rem; border: 1.5px dashed rgba(198,244,50,.3); border-radius: var(--r); color: var(--muted); background: var(--glass); }

.foot { margin-top: 4rem; padding: 1.6rem 2.2rem; border-radius: 26px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: .6rem 2rem; color: var(--muted); font-size: .86rem; background: var(--glass); border: 1px solid var(--line); }
.foot b { color: var(--lime); font-family: var(--serif); font-size: 1.1rem; font-weight: 600; }

/* ---------- Responsive ---------- */
@media (max-width: 980px) {
    .block-container { padding: 1rem .9rem 7rem !important; }
    .hero2 { grid-template-columns: 1fr; grid-template-rows: auto; min-height: 0; border-radius: 28px; }
    .hud { grid-column: 1; grid-row: auto; justify-self: stretch; width: 100%; }
    .hero2::after { background: linear-gradient(180deg, rgba(3,16,11,.55), rgba(3,16,11,.9)); }
    .hero2-stats b { font-size: 1.8rem; }
    .hero2-stats div { padding: .9rem 1rem; }
    .rail, .crops, .rep, .rep-body, .v-main { grid-template-columns: 1fr; }
    .rail::before { display: none; }
    .crops { grid-template-columns: repeat(2, 1fr); }
    .metrics { grid-template-columns: repeat(2, 1fr); }
    .ring { margin: 0 auto; }
    .verdict { padding: 1.7rem 1.4rem; }
    .verdict h3 { font-size: 1.85rem; }
    .row { grid-template-columns: 56px minmax(0, 1fr); }
    .row .tag, .row .cf { grid-column: 2; }
    .row .cf b { text-align: left; }
    .sec { margin-top: 3.6rem; }
    .dock a span { display: none; }
    .dock a { padding: .7rem .85rem; }
}
@media (max-width: 560px) { .crops, .hsum { grid-template-columns: 1fr; } .hero2-stats { grid-template-columns: 1fr; } .hero2-stats div { border-right: none; border-bottom: 1px solid rgba(255,255,255,.1); } .hero2-stats div:last-child { border-bottom: none; } .gauge-cap span:nth-child(2) { display: none; } }
@media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto; }
    .hero2, .rep, .scene .sway, .scanband, .scan::after, .dots i.on, .badge i, .live i, .stepper .st.now i { animation: none !important; }
    * { transition: none !important; }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==================================================
# 1B. UI HELPERS
# ==================================================

def html(markup, target=None):
    """Render HTML. Lines are joined with spaces so Markdown never sees code blocks."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines())
    (target if target is not None else st).markdown(flat, unsafe_allow_html=True)


def section(eyebrow, title, blurb, anchor=""):
    anchor_html = f'<div id="{anchor}" class="anchor"></div>' if anchor else ""
    html(f"""
    {anchor_html}
    <div class="sec">
        <span class="eyebrow">{eyebrow}</span>
        <h2>{title}</h2>
        <p>{blurb}</p>
    </div>
    """)


def subhead(text):
    html(f'<div class="subhead">{text}</div>')


def leaf_icon(color="#12a36b"):
    return (
        "<svg viewBox='0 0 32 32' xmlns='http://www.w3.org/2000/svg' aria-hidden='true'>"
        f"<path d='M16 30C6 22 4 11 16 2c12 9 10 20 0 28z' fill='{color}'/>"
        "<path d='M16 30V9' stroke='rgba(255,255,255,.65)' stroke-width='1.6' fill='none'/></svg>"
    )


ICON_PATHS = {
    "home": "M3 11l9-8 9 8v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1z",
    "flow": "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5",
    "grid": "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
    "scan": "M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3M4 12h16",
    "clock": "M12 7v5l3 2M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z",
    "upload": "M12 16V4m0 0l-4 4m4-4l4 4M4 20h16",
    "report": "M6 3h9l4 4v14H6zM14 3v5h5M9 13h6M9 17h6",
}


def ico(name):
    return (
        "<svg class='ico' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'>"
        f"<path d='{ICON_PATHS[name]}'/></svg>"
    )


def image_to_data_uri(img, max_side=900, quality=82):
    """Display-only copy of an image for custom HTML frames."""
    preview = img.copy()
    preview.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def viewport(src, left_text, right_text="", still=False):
    cls = "viewport still" if still else "viewport"
    return (
        f'<div class="{cls}"><img src="{src}" alt="Leaf image">'
        '<i class="vc tl"></i><i class="vc tr"></i><i class="vc bl"></i><i class="vc br"></i>'
        '<span class="scanband"></span>'
        f'<div class="vp-meta"><span>{left_text}</span><span>{right_text}</span></div></div>'
    )


# ==================================================
# 1C. FLOATING DOCK
# ==================================================

html(f"""
<nav class="dock" aria-label="Quick navigation">
    <a href="#home" target="_self">{ico('home')}<span>Home</span></a>
    <a href="#how" target="_self">{ico('flow')}<span>Process</span></a>
    <a href="#crops" target="_self">{ico('grid')}<span>Crops</span></a>
    <a href="#detect" target="_self" class="primary">{ico('scan')}<span>Analyze</span></a>
    <a href="#history" target="_self">{ico('clock')}<span>History</span></a>
</nav>
""")


# ==================================================
# 1D. CINEMATIC HERO
# ==================================================

def leaf_shape(x, y, length, width, rot, fill, blur="", opacity=1.0, veins=True, sway=False):
    """Procedural botanical leaf. Used only to paint the hero scene."""
    L, W = length, width
    body = (
        f"M0 0C{L * .22:.0f} {-W:.0f} {L * .72:.0f} {-W * .95:.0f} {L:.0f} 0"
        f"C{L * .72:.0f} {W * .95:.0f} {L * .22:.0f} {W:.0f} 0 0Z"
    )
    vein_markup = ""
    if veins:
        paths = f"<path d='M0 0L{L:.0f} 0'/>"
        for i in range(1, 8):
            px = L * i / 8.5
            span = W * 0.78 * math.sin(math.pi * i / 8.5)
            paths += f"<path d='M{px:.0f} 0Q{px + L * .06:.0f} {-span * .55:.0f} {px + L * .15:.0f} {-span:.0f}'/>"
            paths += f"<path d='M{px:.0f} 0Q{px + L * .06:.0f} {span * .55:.0f} {px + L * .15:.0f} {span:.0f}'/>"
        vein_markup = (
            f"<g stroke='rgba(236,255,200,.38)' stroke-width='{max(1.2, L / 200):.1f}' "
            f"fill='none' stroke-linecap='round'>{paths}</g>"
        )
    inner_class = ' class="sway"' if sway else ""
    filter_attr = f' filter="url(#{blur})"' if blur else ""
    return (
        f"<g transform='translate({x} {y}) rotate({rot})' opacity='{opacity}'{filter_attr}>"
        f"<g{inner_class}><path d='{body}' fill='url(#{fill})'/>{vein_markup}</g></g>"
    )


def dew_drop(cx, cy, r):
    return (
        f"<circle cx='{cx:.0f}' cy='{cy:.0f}' r='{r}' fill='url(#dew)' stroke='rgba(255,255,255,.5)' stroke-width='1'/>"
        f"<ellipse cx='{cx - r * .3:.0f}' cy='{cy - r * .38:.0f}' rx='{r * .3:.1f}' ry='{r * .18:.1f}' fill='rgba(255,255,255,.85)'/>"
    )


def build_hero_scene():
    theta = math.radians(-38)
    ox, oy, length = 520, 650, 780
    drops = ""
    for t, off, r in [(.30, -26, 13), (.42, 34, 9), (.55, -42, 16), (.66, 22, 8), (.74, -20, 11), (.86, 14, 7)]:
        dx = ox + t * length * math.cos(theta) - off * math.sin(theta)
        dy = oy + t * length * math.sin(theta) + off * math.cos(theta)
        drops += dew_drop(dx, dy, r)

    return f"""
    <svg viewBox="0 0 1200 700" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Macro illustration of a dew-covered leaf in golden light">
      <defs>
        <linearGradient id="bgG" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#021510"/><stop offset=".55" stop-color="#0a3024"/><stop offset="1" stop-color="#14523b"/></linearGradient>
        <radialGradient id="sun" cx=".84" cy=".06" r=".75"><stop offset="0" stop-color="#f6ecaa" stop-opacity=".7"/><stop offset=".35" stop-color="#c6f432" stop-opacity=".16"/><stop offset="1" stop-color="#c6f432" stop-opacity="0"/></radialGradient>
        <linearGradient id="ray" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff6c8" stop-opacity=".5"/><stop offset="1" stop-color="#fff6c8" stop-opacity="0"/></linearGradient>
        <linearGradient id="gA" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6df0a8"/><stop offset=".55" stop-color="#16a869"/><stop offset="1" stop-color="#0a5a3e"/></linearGradient>
        <linearGradient id="gB" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#2fd18b"/><stop offset="1" stop-color="#0a4a34"/></linearGradient>
        <linearGradient id="gC" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#c3ec62"/><stop offset="1" stop-color="#2b8a4a"/></linearGradient>
        <linearGradient id="gD" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0d4a35"/><stop offset="1" stop-color="#031c13"/></linearGradient>
        <radialGradient id="dew" cx=".5" cy=".5" r=".5"><stop offset=".6" stop-color="#b4ffe0" stop-opacity=".1"/><stop offset="1" stop-color="#ffffff" stop-opacity=".4"/></radialGradient>
        <radialGradient id="vig" cx=".55" cy=".45" r=".8"><stop offset=".5" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".6"/></radialGradient>
        <filter id="blurL" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="16"/></filter>
        <filter id="blurM" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6"/></filter>
        <filter id="grain"><feTurbulence type="fractalNoise" baseFrequency=".85" numOctaves="2" stitchTiles="stitch"/><feColorMatrix type="saturate" values="0"/></filter>
      </defs>
      <rect width="1200" height="700" fill="url(#bgG)"/>
      <rect width="1200" height="700" fill="url(#sun)"/>
      <polygon points="880,0 1010,0 560,700 360,700" fill="url(#ray)" opacity=".55"/>
      <polygon points="1010,0 1100,0 760,700 640,700" fill="url(#ray)" opacity=".4"/>
      <polygon points="760,0 830,0 380,700 300,700" fill="url(#ray)" opacity=".3"/>
      {leaf_shape(-70, 730, 760, 280, -22, 'gD', blur='blurL', opacity=.95, veins=False)}
      {leaf_shape(1230, 50, 700, 250, 148, 'gD', blur='blurL', opacity=.9, veins=False)}
      {leaf_shape(420, -50, 460, 160, 72, 'gB', blur='blurM', opacity=.5, veins=False)}
      {leaf_shape(660, 720, 560, 170, -72, 'gC', blur='blurM', opacity=.55, veins=False, sway=True)}
      {leaf_shape(520, 650, 780, 215, -38, 'gA', veins=True, sway=True)}
      {drops}
      {leaf_shape(1270, 780, 680, 230, -152, 'gD', blur='blurL', opacity=.96, veins=False)}
      <rect width="1200" height="700" filter="url(#grain)" opacity=".12"/>
      <rect width="1200" height="700" fill="url(#vig)"/>
    </svg>
    """


@st.cache_data(show_spinner=False)
def load_hero_photo(path_text):
    """Optional: drop a photograph at assets/hero.jpg to replace the painted scene."""
    photo_path = Path(path_text)
    if not photo_path.exists():
        return None
    try:
        photo = Image.open(photo_path).convert("RGB")
        photo.thumbnail((1600, 1600))
        buffer = io.BytesIO()
        photo.save(buffer, format="JPEG", quality=80)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


hero_photo = load_hero_photo(str(Path(__file__).resolve().parent / "assets" / "hero.jpg"))

if hero_photo:
    hero_scene = f'<img src="data:image/jpeg;base64,{hero_photo}" alt="Healthy crop leaves in the field">'
else:
    hero_scene = build_hero_scene()

html(f"""
<div id="home" class="anchor"></div>
<section class="hero2">
    <div class="scene">{hero_scene}</div>
    <div class="hero2-copy">
        <span class="badge"><i></i>AI-powered crop care</span>
        <h1>Protect every harvest <span>before the damage shows.</span></h1>
        <p>CropCare AI studies a photograph of a single leaf from nine supported crops and
        estimates its condition, with a confidence score, symptoms and general management
        guidance in English and ಕನ್ನಡ. It supports local agricultural expertise and never replaces it.</p>
        <div class="actions">
            <a class="cta" href="#detect" target="_self">Analyze Crop &rarr;</a>
            <a class="ghost" href="#how" target="_self">See how it works</a>
        </div>
    </div>
    <aside class="hud">
        <h6>Inside every analysis</h6>
        <div><span>Image validator</span><b>Preliminary screening</b></div>
        <div><span>Disease model</span><b>33 classes</b></div>
        <div><span>Model input</span><b>160 x 160 px</b></div>
        <div><span>Report</span><b>Top 3, symptoms, guidance</b></div>
    </aside>
    <div class="hero2-stats">
        <div><b>9</b><small>Supported crops</small></div>
        <div><b>33</b><small>Classification classes</small></div>
        <div><b>2</b><small>Languages</small></div>
    </div>
</section>
""")


# ==================================================
# 2. PATHS AND SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
EXPANDED_MODEL_DIR = BASE_DIR / "expanded_training_output"

DISEASE_MODEL_PATH = EXPANDED_MODEL_DIR / "cropcare_33class.keras"
CLASS_NAMES_PATH = EXPANDED_MODEL_DIR / "class_names.json"
VALIDATOR_MODEL_PATH = MODEL_DIR / "crop_validator.keras"

# Expanded model input dimensions
IMAGE_SIZE = (160, 160)
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

# Descriptions for the 25 additional classes in the expanded model.
EXPANDED_INFO = {
    "Apple___Apple_scab": ("Apple", "Apple Scab", "Olive-green to dark scabby spots may appear on leaves.", "Remove fallen infected leaves, improve airflow and seek local crop advice."),
    "Apple___Black_rot": ("Apple", "Black Rot", "Purple or brown leaf spots may develop and enlarge.", "Remove infected plant material and avoid leaving diseased debris near trees."),
    "Apple___Cedar_apple_rust": ("Apple", "Cedar Apple Rust", "Yellow-orange spots may appear on the upper leaf surface.", "Monitor nearby host plants and ask a local horticulture expert about management."),
    "Apple___healthy": ("Apple", "Healthy", "No visible disease symptoms are identified by this model.", "Continue regular watering, orchard sanitation and routine monitoring."),
    "Cherry_(including_sour)___Powdery_mildew": ("Cherry", "Powdery Mildew", "A pale, powdery fungal growth may appear on leaf surfaces.", "Improve airflow, remove badly affected material and seek local crop advice."),
    "Cherry_(including_sour)___healthy": ("Cherry", "Healthy", "No visible disease symptoms are identified by this model.", "Continue routine monitoring and appropriate orchard care."),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("Corn", "Cercospora / Gray Leaf Spot", "Elongated gray or tan lesions may develop on corn leaves.", "Use crop rotation, residue management and locally recommended resistant varieties."),
    "Corn_(maize)___Common_rust_": ("Corn", "Common Rust", "Reddish-brown raised pustules may appear on leaf surfaces.", "Monitor spread and consult local agricultural guidance for suitable management."),
    "Corn_(maize)___healthy": ("Corn", "Healthy", "No visible disease symptoms are identified by this model.", "Maintain balanced nutrition, suitable irrigation and regular field scouting."),
    "Corn_(maize)___Northern_Leaf_Blight": ("Corn", "Northern Leaf Blight", "Long, cigar-shaped gray-green lesions may appear on leaves.", "Rotate crops, manage infected residue and use locally recommended resistant hybrids."),
    "Grape___Black_rot": ("Grape", "Black Rot", "Brown circular leaf lesions may develop with dark margins.", "Remove infected berries and leaves and seek local vineyard management advice."),
    "Grape___Esca_(Black_Measles)": ("Grape", "Esca (Black Measles)", "Leaves may show interveinal discoloration or tiger-stripe patterns.", "Prune carefully in dry conditions and consult a vineyard disease specialist."),
    "Grape___healthy": ("Grape", "Healthy", "No visible disease symptoms are identified by this model.", "Continue canopy management, sanitation and regular vineyard monitoring."),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("Grape", "Leaf Blight", "Brown leaf spots may enlarge and cause affected tissue to dry.", "Remove infected fallen leaves and maintain good canopy airflow."),
    "Peach___Bacterial_spot": ("Peach", "Bacterial Spot", "Small dark, angular or shot-hole-like lesions may appear on leaves.", "Avoid working with wet foliage and seek local orchard disease advice."),
    "Peach___healthy": ("Peach", "Healthy", "No visible disease symptoms are identified by this model.", "Continue routine orchard inspection and appropriate plant care."),
    "Strawberry___healthy": ("Strawberry", "Healthy", "No visible disease symptoms are identified by this model.", "Maintain good airflow, appropriate irrigation and routine monitoring."),
    "Strawberry___Leaf_scorch": ("Strawberry", "Leaf Scorch", "Small purple-red spots may develop and merge into scorched areas.", "Remove old infected leaves, reduce prolonged leaf wetness and seek local advice."),
    "Tomato___Bacterial_spot": ("Tomato", "Bacterial Spot", "Small dark, water-soaked spots may appear on leaves.", "Avoid overhead irrigation, remove infected debris and seek local advice."),
    "Tomato___Leaf_Mold": ("Tomato", "Leaf Mold", "Yellow patches may appear above leaves with olive-green growth underneath.", "Improve ventilation and avoid prolonged leaf humidity."),
    "Tomato___Septoria_leaf_spot": ("Tomato", "Septoria Leaf Spot", "Small circular spots with pale centers and dark borders may appear.", "Remove affected lower leaves and avoid splashing water onto foliage."),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Tomato", "Spider Mites", "Fine stippling or bronzing may appear, sometimes with delicate webbing.", "Inspect leaf undersides and seek locally appropriate integrated pest advice."),
    "Tomato___Target_Spot": ("Tomato", "Target Spot", "Brown lesions may show concentric rings resembling a target.", "Improve airflow, avoid prolonged leaf wetness and remove affected debris."),
    "Tomato___Tomato_mosaic_virus": ("Tomato", "Tomato Mosaic Virus", "Leaves may show mottling, mosaic patterns or distortion.", "Remove suspected infected plants and disinfect tools; seek expert confirmation."),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Tomato", "Tomato Yellow Leaf Curl Virus", "Leaves may curl upward and become yellow or stunted.", "Monitor whiteflies and seek local integrated pest-management advice."),
}

for class_key, (crop, status, symptoms, guidance) in EXPANDED_INFO.items():
    disease_info.setdefault(class_key, {
        "crop": crop,
        "status": status,
        "symptoms": symptoms,
        "guidance": guidance,
        "kn_symptoms": "ಈ ಲಕ್ಷಣಗಳು ಕಂಡುಬಂದರೆ ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರಿಂದ ದೃಢೀಕರಣ ಪಡೆಯಿರಿ.",
        "kn_guidance": "ಬಾಧಿತ ಭಾಗಗಳನ್ನು ಗಮನಿಸಿ ಮತ್ತು ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ."
    })

missing_info = [name for name in class_names if name not in disease_info]
if missing_info:
    st.error("Missing disease descriptions for: " + ", ".join(missing_info))
    st.stop()


# ==================================================
# 5. SIDEBAR: CONTROL CENTRE AND LANGUAGE
# ==================================================

with st.sidebar:
    html(f"""
    <div class="s-brand">{leaf_icon('#2fd18b')}<div><b>CropCare<i>AI</i></b><span>Crop disease intelligence</span></div></div>
    <div class="s-label">Navigate</div>
    <a class="snav" href="#home" target="_self">{ico('home')}Overview</a>
    <a class="snav" href="#how" target="_self">{ico('flow')}How it works</a>
    <a class="snav" href="#crops" target="_self">{ico('grid')}Supported crops</a>
    <a class="snav" href="#detect" target="_self">{ico('scan')}Analysis workspace</a>
    <a class="snav" href="#history" target="_self">{ico('clock')}Session history</a>
    <div class="s-label">Language / ಭಾಷೆ</div>
    """, target=st.sidebar)

    language = st.selectbox(
        "Choose Language / ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        ["English", "ಕನ್ನಡ"],
        label_visibility="collapsed"
    )

    sidebar_slot = st.empty()

    html("""
    <p class="s-note">Predictions are AI-generated suggestions. Please verify them with an agricultural expert before acting.</p>
    """, target=st.sidebar)

is_kannada = language == "ಕನ್ನಡ"


# ==================================================
# 5A. HOW IT WORKS
# ==================================================

section(
    "Process",
    "From leaf to report in three steps",
    "Everything runs on a single photograph. Nothing is stored beyond this browser session.",
    anchor="how"
)

html(f"""
<div class="rail">
    <div class="node"><div class="ic">{ico('upload')}</div>
        <h3>Upload a leaf photo</h3>
        <p>Add a clear, well-lit photograph of one leaf from a supported crop. JPG, JPEG and PNG files work.</p></div>
    <div class="node"><div class="ic">{ico('scan')}</div>
        <h3>Screen and classify</h3>
        <p>An image validator gives a preliminary screening score, then the disease model scores all 33 classes.</p></div>
    <div class="node"><div class="ic">{ico('report')}</div>
        <h3>Review the report</h3>
        <p>See the predicted crop and condition, the confidence, the top three predictions, symptoms and general guidance.</p></div>
</div>
""")


# ==================================================
# 5B. SUPPORTED CROPS  (built from the real class list)
# ==================================================

CROP_ORDER = [
    "Apple", "Cherry", "Corn", "Grape", "Peach",
    "Bell Pepper", "Potato", "Strawberry", "Tomato"
]

CROP_COLORS = {
    "Apple": "#e0553f", "Cherry": "#b3315a", "Corn": "#e3b02a",
    "Grape": "#8b62b8", "Peach": "#ee9468", "Bell Pepper": "#4fb35c",
    "Potato": "#b8905f", "Strawberry": "#e8466a", "Tomato": "#e0643a",
}

crop_conditions = {crop_name: [] for crop_name in CROP_ORDER}
for class_name in class_names:
    entry = disease_info[class_name]
    crop_conditions.setdefault(entry["crop"], []).append(entry["status"])

crop_tiles = ""
for crop_name, conditions in crop_conditions.items():
    chips = "".join(
        f'<span class="chip{" ok" if cond == "Healthy" else ""}">{esc(cond)}</span>'
        for cond in sorted(conditions, key=lambda c: (c != "Healthy", c))
    )
    color = CROP_COLORS.get(crop_name, "#12a36b")
    crop_tiles += f"""
    <div class="crop" style="--c:{color}">
        <div class="crop-top">
            <div class="badge-ic">{leaf_icon(color)}</div>
            <span>{len(conditions)}<small>classes</small></span>
        </div>
        <h4>{esc(crop_name)}</h4>
        <div class="chips">{chips}</div>
    </div>
    """

section(
    "Field index",
    "Nine crops, 33 classes",
    "Every class below comes from the trained model, healthy and diseased leaves included.",
    anchor="crops"
)
html(f'<div class="crops">{crop_tiles}</div>')
st.caption(
    "Orange is not included because healthy orange leaves were not present in the training dataset."
)


# ==================================================
# 7. SESSION STATE AND HISTORY
# ==================================================

if "history" not in st.session_state:
    st.session_state.history = []

# Additive, display-only companion list (thumbnail + time recorded when a
# prediction is made). The original "history" records are unchanged.
if "history_extra" not in st.session_state:
    st.session_state.history_extra = []

# The most recent analysis, kept so the report survives a language change.
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def loader_markup(title, detail, stage):
    dots = "".join(
        f'<i class="{"done" if n < stage else "on" if n == stage else ""}"></i>'
        for n in (1, 2)
    )
    return (
        '<div class="loader" role="status" aria-live="polite">'
        f'<div class="scan">{leaf_icon("#2fd18b")}</div>'
        f'<div><b>{title}</b><span>{detail}</span><div class="dots">{dots}</div></div>'
        '</div>'
    )


# ==================================================
# 6. ANALYSIS WORKSPACE
# ==================================================

def stepper(stage):
    items = ""
    for number, label in enumerate(["Upload", "Analyse", "Report"], start=1):
        state = "done" if number < stage else "now" if number == stage else ""
        items += f'<div class="st {state}"><i>{number}</i>{label}</div>'
    return f'<div class="stepper">{items}</div>'


section(
    "Workspace",
    "Analyze a leaf",
    "A sharp, well-lit close-up gives the model the best chance of a useful result.",
    anchor="detect"
)

image = None
detect_clicked = False
selected_demo_label = "Use model prediction"

ws_left, ws_right = st.columns([1.5, 1], gap="large")

with ws_left:
    html('<span class="ws-left"></span>')
    stepper_slot = st.empty()

    uploaded_file = st.file_uploader(
        "Choose a leaf photograph",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            html(viewport(
                image_to_data_uri(image),
                f"Uploaded Image · {esc(uploaded_file.name)}",
                f"{image.size[0]} x {image.size[1]} px"
            ))
        except Exception as error:
            st.error(
                f"Unable to process image: {error}"
            )

with ws_right:
    ready_class = " ready" if image is not None else ""
    ready_text = "Image ready" if image is not None else "Waiting for an image"
    html(f"""
    <span class="ws-right"></span>
    <div class="console-head"><h4>Analysis console</h4><span class="live{ready_class}"><i></i>{ready_text}</span></div>
    """)

    if image is None:
        html("""
        <ul class="tips">
            <li>Use natural daylight and avoid harsh shadows.</li>
            <li>Fill the frame with a single leaf.</li>
            <li>Keep the leaf in focus against a plain background.</li>
            <li>Photograph the affected area up close.</li>
        </ul>
        """)
    else:
        st.info(
            "Upload a clear leaf image from Apple, Cherry, Corn, Grape, Peach, "
            "Bell Pepper, Potato, Strawberry or Tomato."
        )

        with st.container(border=True):
            html('<div class="demo-h">Optional demo verification</div>')
            st.caption(
                "If you already know the leaf's correct label (for example, "
                "from a verified test image), you can select it here. This is a "
                "manual demo correction and does not change the AI model."
            )

            demo_label_options = ["Use model prediction"] + sorted(
                disease_info.keys(),
                key=lambda k: (disease_info[k]["crop"], disease_info[k]["status"])
            )

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
            "Detect Crop Disease",
            type="primary",
            use_container_width=True
        )

    st.caption(
        "Supported crops: Apple, Cherry, Corn, Grape, Peach, Bell Pepper, "
        "Potato, Strawberry and Tomato (33 trained categories)."
    )


# ==================================================
# 8. IMAGE VALIDATION AND DISEASE DETECTION
# ==================================================

if image is not None:

    try:

        loader_slot = st.empty()

        if detect_clicked:

            # ------------------------------------------
            # STAGE 1: IMAGE VALIDATION
            # ------------------------------------------

            loader_slot.markdown(
                loader_markup(
                    "Checking the image",
                    "Running the preliminary screening model.",
                    1
                ),
                unsafe_allow_html=True
            )

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

            # ------------------------------------------
            # STAGE 2: DISEASE DETECTION
            # ------------------------------------------

            loader_slot.markdown(
                loader_markup(
                    "Detecting crop condition",
                    "Scoring the leaf against 33 trained classes.",
                    2
                ),
                unsafe_allow_html=True
            )

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

            loader_slot.empty()

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
            # remains unchanged and is still shown separately in the report.
            manual_correction_used = selected_demo_label != "Use model prediction"
            final_class = (
                selected_demo_label if manual_correction_used else predicted_class
            )
            display_info = disease_info[final_class]

            top_indices = np.argsort(
                predictions
            )[-3:][::-1]

            # ------------------------------------------
            # STORE RESULT FOR THE REPORT
            # ------------------------------------------

            analysed_at = datetime.now()

            st.session_state.last_result = {
                "image": image,
                "time": analysed_at.strftime("%d %b %Y, %H:%M"),
                "crop_probability": crop_probability,
                "non_crop_probability": non_crop_probability,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "manual": manual_correction_used,
                "final_class": final_class,
                "top3": [
                    (class_names[int(index)], float(predictions[index]))
                    for index in top_indices
                ],
            }

            # ------------------------------------------
            # HISTORY
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
                "time": analysed_at.strftime("%d %b %Y, %H:%M"),
            })

    except Exception as error:

        st.error(
            f"Unable to process image: {error}"
        )


# Update the stepper now that this run's analysis (if any) has finished.
if st.session_state.last_result is not None and image is not None:
    workspace_stage = 4
elif image is not None:
    workspace_stage = 2
else:
    workspace_stage = 1
stepper_slot.markdown(stepper(workspace_stage), unsafe_allow_html=True)


# ==================================================
# 9. DIAGNOSTIC REPORT DASHBOARD
# ==================================================

def render_report(result):
    crop_probability = result["crop_probability"]
    non_crop_probability = result["non_crop_probability"]
    confidence = result["confidence"]
    confidence_percent = confidence * 100
    ai_info = disease_info[result["predicted_class"]]
    display_info = disease_info[result["final_class"]]
    manual_correction_used = result["manual"]
    top3 = result["top3"]

    if len(top3) > 1:
        runner_gap = (top3[0][1] - top3[1][1]) * 100
    else:
        runner_gap = 0.0

    if ai_info["status"] == "Healthy":
        tone, pill_text = "ok", "Healthy leaf"
    elif ai_info["status"] == "Late Blight":
        tone, pill_text = "bad", "Disease detected"
    else:
        tone, pill_text = "warn", "Disease detected"

    ring_value = max(0.0, min(confidence, 1.0)) * 100
    threshold_pos = DISEASE_THRESHOLD * 100
    when = esc(result["time"])

    section(
        "Report",
        "Diagnostic report",
        f"The raw output of the disease model for your uploaded leaf. Analysed {when}."
    )

    # ---- Verdict: image + diagnosis ----
    html(f"""
    <div class="rep">
        <div>{viewport(image_to_data_uri(result['image']), 'Analysed leaf', when, still=True)}</div>
        <div class="verdict {tone}">
            <div class="v-top">
                <span class="status"><i></i>Detection complete</span>
                <small>Analysed {when}</small>
            </div>
            <div class="v-main">
                <div>
                    <small>Detected crop</small>
                    <div class="crop-line">AI model prediction: {esc(ai_info['crop'])}</div>
                    <small>Predicted condition</small>
                    <h3>AI predicted condition: {esc(ai_info['status'])}</h3>
                    <span class="pill {tone}">{pill_text}</span>
                </div>
                <div class="ring" style="--p:{ring_value:.2f}">
                    <div><b>{confidence_percent:.2f}%</b><small>Confidence</small></div>
                </div>
            </div>
            <div class="gauge"><i style="width:{ring_value:.2f}%"></i><u style="left:{threshold_pos:.0f}%"></u></div>
            <div class="gauge-cap"><span>0%</span><span>White mark: {DISEASE_THRESHOLD * 100:.0f}% review threshold</span><span>100%</span></div>
            <p class="notice">This is an AI-generated suggestion, not a confirmed diagnosis. Please verify it with an agricultural expert before acting.</p>
        </div>
    </div>
    """)

    # ---- Metrics ----
    if confidence >= DISEASE_THRESHOLD:
        threshold_text, threshold_sub = "Above", f"the {DISEASE_THRESHOLD * 100:.0f}% review threshold"
    else:
        threshold_text, threshold_sub = "Below", f"the {DISEASE_THRESHOLD * 100:.0f}% review threshold"

    html(f"""
    <div class="metrics">
        <div class="metric"><small>Model confidence</small><b>{confidence_percent:.2f}%</b><span>for the top class</span></div>
        <div class="metric"><small>Lead over runner-up</small><b>{runner_gap:.2f}</b><span>percentage points</span></div>
        <div class="metric"><small>Crop-leaf score</small><b>{crop_probability * 100:.2f}%</b><span>preliminary screening</span></div>
        <div class="metric"><small>Review threshold</small><b class="sm">{threshold_text}</b><span>{threshold_sub}</span></div>
    </div>
    """)

    # ---- Screening ----
    subhead("Image screening")

    crop_bar = max(0.0, min(crop_probability * 100, 100.0))
    html(f"""
    <div class="screen">
        <div><small>Estimated crop-leaf score</small><b>{crop_probability * 100:.2f}%</b></div>
        <div class="track"><i style="width:{crop_bar:.2f}%"></i></div>
    </div>
    """)

    # The validator is a preliminary screening model, not a reliable
    # crop/disease authority. It never blocks the disease classifier.
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
            "a leaf from one of the supported crops. This validator is only "
            "a preliminary screening signal and was trained separately."
        )

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

    # ---- Guidance (English / Kannada) ----
    if is_kannada:
        sec_title = "ಲಕ್ಷಣಗಳು & ಸಲಹೆಗಳು"
        ov_title, ov_crop, ov_cond, ov_src = "ಅವಲೋಕನ", "ಬೆಳೆ", "ಸ್ಥಿತಿ", "ಮೂಲ"
        src_text = "ಕೈಯಾರೆ ಆಯ್ಕೆ" if manual_correction_used else "AI ಮಾದರಿ"
        sym_title, sym_text = "ಲಕ್ಷಣಗಳು", display_info["kn_symptoms"]
        gd_title, gd_text = "ಸಲಹೆಗಳು", display_info["kn_guidance"]
    else:
        sec_title = "Understanding the condition"
        ov_title, ov_crop, ov_cond, ov_src = "Overview", "Crop", "Condition", "Source"
        src_text = "Manual demo label" if manual_correction_used else "AI model"
        sym_title, sym_text = "Symptoms", display_info["symptoms"]
        gd_title, gd_text = "General management guidance", display_info["guidance"]

    # ---- Top 3 chart ----
    bars = ""
    for rank, (class_name, probability) in enumerate(top3, start=1):
        candidate_info = disease_info.get(class_name, {})
        candidate_crop = candidate_info.get("crop", "Unknown crop")
        display_name = candidate_info.get("status", class_name)
        width = max(0.0, min(probability, 1.0)) * 100
        first = " first" if rank == 1 else ""
        bars += f"""
        <div class="bar-row{first}">
            <div class="lbl"><span><em>{rank}</em>{esc(candidate_crop)} — {esc(display_name)}</span><b>{probability * 100:.2f}%</b></div>
            <div class="bar"><i style="width:{width:.2f}%"></i></div>
        </div>
        """

    subhead(sec_title)

    html(f"""
    <div class="rep-body">
        <div class="paper">
            <h4>Top 3 predictions</h4>
            <p>The three highest-scoring classes, exactly as the model produced them.</p>
            <div class="bars">{bars}</div>
            <div class="scale"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
        </div>
        <div class="stack">
            <div class="panel">
                <h4>{ov_title}</h4>
                <div class="kv"><span>{ov_crop}</span><b>{esc(display_info['crop'])}</b></div>
                <div class="kv"><span>{ov_cond}</span><b>{esc(display_info['status'])}</b></div>
                <div class="kv"><span>{ov_src}</span><b>{src_text}</b></div>
            </div>
            <div class="panel symptoms"><h4>{sym_title}</h4><p>{sym_text}</p></div>
            <div class="panel guide"><h4>{gd_title}</h4><p>{gd_text}</p></div>
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
        "The top-three scores above are the model's actual outputs. "
        "A manual demo label, if selected, does not alter those scores "
        "or retrain the model."
    )


if st.session_state.last_result is not None:
    render_report(st.session_state.last_result)


# ==================================================
# 10. PREDICTION HISTORY
# ==================================================

section(
    "Session log",
    "Prediction history",
    "Every prediction made during this session, newest first.",
    anchor="history"
)

if st.session_state.history:

    history_total = len(st.session_state.history)
    history_healthy = sum(
        1 for record in st.session_state.history
        if record["AI condition prediction"] == "Healthy"
    )

    html(f"""
    <div class="hsum">
        <div><b>{history_total}</b><small>Analyses this session</small></div>
        <div><b>{history_healthy}</b><small>Predicted healthy</small></div>
        <div><b>{history_total - history_healthy}</b><small>Predicted with a condition</small></div>
    </div>
    """)

    extras = st.session_state.history_extra
    items = ""

    for position in range(len(st.session_state.history) - 1, -1, -1):

        record = st.session_state.history[position]
        extra = extras[position] if position < len(extras) else {}

        if extra.get("thumb"):
            thumb_html = f'<img src="data:image/jpeg;base64,{extra["thumb"]}" alt="Uploaded leaf thumbnail">'
        else:
            thumb_html = f'<div class="ph">{leaf_icon()}</div>'

        when_text = f' · {extra["time"]}' if extra.get("time") else ""

        try:
            conf_width = max(0.0, min(float(record["AI confidence"].rstrip("%")), 100.0))
        except ValueError:
            conf_width = 0.0

        items += f"""
        <div class="row">
            {thumb_html}
            <div class="who">
                <b>{esc(record['AI crop prediction'])} — {esc(record['AI condition prediction'])}</b>
                <span>Displayed: {esc(record['Displayed demo label'])}{when_text}</span>
            </div>
            <span class="tag">{esc(record['Label source'])}</span>
            <div class="cf"><b>{record['AI confidence']}</b><div class="track"><i style="width:{conf_width:.2f}%"></i></div></div>
        </div>
        """

    html(f'<div class="ledger">{items}</div>')

    st.write("")

    with st.expander("View history as table"):
        st.dataframe(
            st.session_state.history,
            use_container_width=True
        )

    if st.button("Clear History"):
        st.session_state.history = []
        st.session_state.history_extra = []
        st.session_state.last_result = None
        st.rerun()

else:
    html('<div class="empty">No predictions yet. Upload a leaf above to run your first analysis.</div>')


# ==================================================
# 11. FOOTER AND SIDEBAR SESSION CARD
# ==================================================

html("""
<div class="foot">
    <span><b>CropCare AI</b> &nbsp;|&nbsp; AI-based crop disease detection</span>
    <span>Predictions are AI-generated suggestions. Verify with an agricultural expert. Educational project.</span>
</div>
""")

session_total = len(st.session_state.history)
session_healthy = sum(
    1 for record in st.session_state.history
    if record["AI condition prediction"] == "Healthy"
)
sidebar_slot.markdown(
    f"""<div class="s-card"><h6>This session</h6><div class="s-grid">
    <div><b>{session_total}</b><small>Analyses</small></div>
    <div><b>{session_healthy}</b><small>Healthy</small></div>
    <div><b>{session_total - session_healthy}</b><small>Flagged</small></div>
    </div></div>""",
    unsafe_allow_html=True
)
