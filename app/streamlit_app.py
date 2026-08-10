# app/streamlit_app.py
import json
import math
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

API = "https://e-commerce-fraud-detection-system.onrender.com"

st.set_page_config(
    page_title="FraudShield · Risk Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Clash+Display:wght@500;600;700&family=Bricolage+Grotesque:wght@300;400;500;600;700&display=swap');
      @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,300;12..96,400;12..96,500;12..96,600;12..96,700&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

      :root {
        --bg:        #08090d;
        --bg2:       #0d0f16;
        --surface:   #111420;
        --surface2:  #161928;
        --surface3:  #1c2035;
        --border:    #1f2438;
        --border2:   #2a2f4a;
        --border3:   #353c5c;
        --text:      #e9ecf5;
        --text2:     #b0b8d4;
        --muted:     #636d8e;
        --faint:     #3a4060;

        --blue:      #4d8cfa;
        --blue-dim:  rgba(77,140,250,0.12);
        --blue-glow: rgba(77,140,250,0.25);
        --violet:    #8b6ef5;
        --violet-dim:rgba(139,110,245,0.1);

        --low:        #2dd87a;
        --low-dim:    rgba(45,216,122,0.08);
        --low-border: rgba(45,216,122,0.22);
        --low-glow:   rgba(45,216,122,0.15);

        --med:        #f5a623;
        --med-dim:    rgba(245,166,35,0.08);
        --med-border: rgba(245,166,35,0.22);
        --med-glow:   rgba(245,166,35,0.15);

        --high:        #f0455a;
        --high-dim:    rgba(240,69,90,0.08);
        --high-border: rgba(240,69,90,0.22);
        --high-glow:   rgba(240,69,90,0.2);

        --font-display: 'Bricolage Grotesque', 'SF Pro Display', sans-serif;
        --font-mono:    'Space Mono', 'Fira Code', monospace;
        --font-body:    'Bricolage Grotesque', sans-serif;
        --radius:    10px;
        --radius-lg: 16px;
        --radius-xl: 20px;
      }

      /* ── Base Reset ───────────────────────────────────────── */
      html, body, [data-testid="stApp"] {
        background: var(--bg) !important;
        font-family: var(--font-body) !important;
        color: var(--text) !important;
      }
      #MainMenu, footer,
      [data-testid="stToolbar"],
      [data-testid="stDecoration"],
      [data-testid="stHeader"] { display: none !important; }
      .block-container {
        padding: 2.5rem 3rem 5rem !important;
        max-width: 1440px !important;
      }

      /* ── Background grid texture ──────────────────────────── */
      [data-testid="stApp"]::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
          linear-gradient(rgba(77,140,250,0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(77,140,250,0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
      }

      /* ── App Header ───────────────────────────────────────── */
      .app-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 0 0 2rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2.5rem;
        position: relative;
      }
      .app-header::after {
        content: '';
        position: absolute;
        bottom: -1px;
        left: 0;
        width: 120px;
        height: 1px;
        background: linear-gradient(90deg, var(--blue), transparent);
      }
      .app-logo {
        width: 44px; height: 44px;
        background: linear-gradient(135deg, var(--blue) 0%, var(--violet) 100%);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px;
        box-shadow: 0 0 24px var(--blue-glow), inset 0 1px 0 rgba(255,255,255,0.12);
        flex-shrink: 0;
      }
      .app-title {
        font-family: var(--font-display);
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--text);
        margin: 0;
        line-height: 1.1;
      }
      .app-subtitle {
        font-size: 0.72rem;
        color: var(--muted);
        font-family: var(--font-mono);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 3px 0 0 0;
      }
      .live-badge {
        margin-left: auto;
        display: flex; align-items: center; gap: 7px;
        background: var(--low-dim);
        border: 1px solid var(--low-border);
        padding: 6px 14px;
        border-radius: 999px;
        font-family: var(--font-mono);
        font-size: 0.7rem;
        color: var(--low);
        letter-spacing: 0.05em;
      }
      .live-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--low);
        box-shadow: 0 0 6px var(--low);
        animation: pulse-dot 2.4s ease-in-out infinite;
      }
      @keyframes pulse-dot {
        0%,100% { opacity:1; box-shadow: 0 0 6px var(--low); }
        50% { opacity:0.35; box-shadow: 0 0 2px var(--low); }
      }

      /* ── Cards ────────────────────────────────────────────── */
      .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        padding: 28px;
        height: 100%;
        position: relative;
        overflow: hidden;
      }
      .card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(77,140,250,0.3), transparent);
      }
      .card-title {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin: 0 0 20px 0;
      }

      /* ── Risk Band Blocks ─────────────────────────────────── */
      .band-wrap {
        border-radius: var(--radius-lg);
        padding: 20px 22px;
        margin-bottom: 14px;
        position: relative;
        overflow: hidden;
      }
      .band-wrap::after {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 120px; height: 120px;
        border-radius: 50%;
        opacity: 0.08;
        filter: blur(40px);
      }
      .band-low  { background: var(--low-dim);  border: 1px solid var(--low-border);  }
      .band-med  { background: var(--med-dim);  border: 1px solid var(--med-border);  }
      .band-high { background: var(--high-dim); border: 1px solid var(--high-border); }
      .band-low::after  { background: var(--low);  }
      .band-med::after  { background: var(--med);  }
      .band-high::after { background: var(--high); }

      .band-label {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 6px;
        display: flex; align-items: center; gap: 6px;
      }
      .band-label-low  { color: var(--low);  }
      .band-label-med  { color: var(--med);  }
      .band-label-high { color: var(--high); }

      .band-score {
        font-family: var(--font-display);
        font-size: 3.8rem;
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.05em;
      }
      .score-low  { color: var(--low);  text-shadow: 0 0 30px var(--low-glow);  }
      .score-med  { color: var(--med);  text-shadow: 0 0 30px var(--med-glow);  }
      .score-high { color: var(--high); text-shadow: 0 0 30px var(--high-glow); }

      .band-name {
        font-family: var(--font-mono);
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        margin-top: 8px;
        text-transform: uppercase;
        opacity: 0.85;
      }

      /* Score bar */
      .score-bar-track {
        height: 4px;
        background: rgba(255,255,255,0.06);
        border-radius: 999px;
        margin: 12px 0 6px;
        overflow: hidden;
      }
      .score-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.8s cubic-bezier(.22,.68,0,1.2);
        position: relative;
      }
      .score-bar-fill::after {
        content: '';
        position: absolute;
        right: 0; top: -2px;
        width: 8px; height: 8px;
        border-radius: 50%;
        transform: translateX(50%);
      }
      .fill-low  { background: linear-gradient(90deg, rgba(45,216,122,0.3), var(--low)); }
      .fill-low::after  { background: var(--low); box-shadow: 0 0 6px var(--low); }
      .fill-med  { background: linear-gradient(90deg, rgba(245,166,35,0.3), var(--med)); }
      .fill-med::after  { background: var(--med); box-shadow: 0 0 6px var(--med); }
      .fill-high { background: linear-gradient(90deg, rgba(240,69,90,0.3), var(--high)); }
      .fill-high::after { background: var(--high); box-shadow: 0 0 6px var(--high); }

      .action-text {
        font-size: 0.82rem;
        color: var(--text2);
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid rgba(255,255,255,0.06);
        font-weight: 500;
        letter-spacing: 0.01em;
      }

      /* ── Reason tags ──────────────────────────────────────── */
      .reason-tag {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        background: var(--surface3);
        border: 1px solid var(--border2);
        font-size: 0.73rem;
        color: var(--text2);
        margin: 3px 3px 3px 0;
        font-family: var(--font-mono);
        letter-spacing: 0.02em;
        transition: border-color 0.2s;
      }
      .reason-tag:hover { border-color: var(--border3); }

      /* ── Extra metric mini-cards ──────────────────────────── */
      .extra-metric-card {
        flex: 1; min-width: 120px;
        background: var(--surface2);
        border: 1px solid var(--border2);
        border-radius: var(--radius);
        padding: 12px 16px;
        transition: border-color 0.2s;
      }
      .extra-metric-card:hover { border-color: var(--border3); }
      .extra-metric-label {
        font-family: var(--font-mono);
        font-size: 0.66rem;
        text-transform: uppercase;
        letter-spacing: .1em;
        color: var(--muted);
        margin-bottom: 5px;
      }
      .extra-metric-value {
        font-family: var(--font-display);
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.02em;
      }

      /* ── Pills ────────────────────────────────────────────── */
      .pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 12px;
        border-radius: 999px;
        border: 1px solid var(--border2);
        background: var(--surface2);
        font-size: 0.75rem;
        color: var(--muted);
        font-family: var(--font-mono);
        letter-spacing: 0.02em;
      }
      .pill-accent {
        border-color: rgba(77,140,250,0.25);
        background: var(--blue-dim);
        color: var(--blue);
      }

      /* ── Native Streamlit overrides ───────────────────────── */
      [data-testid="stMetric"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 14px 18px !important;
        transition: border-color 0.2s !important;
      }
      [data-testid="stMetric"]:hover {
        border-color: var(--border2) !important;
      }
      [data-testid="stMetricLabel"] {
        font-family: var(--font-mono) !important;
        font-size: 0.68rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: var(--muted) !important;
      }
      [data-testid="stMetricValue"] {
        font-family: var(--font-display) !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--text) !important;
        letter-spacing: -0.03em !important;
      }

      /* Inputs */
      [data-testid="stTextInput"] input,
      [data-testid="stNumberInput"] input {
        background: var(--surface2) !important;
        border-color: var(--border2) !important;
        color: var(--text) !important;
        border-radius: var(--radius) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.85rem !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
      }
      [data-testid="stTextInput"] input:focus,
      [data-testid="stNumberInput"] input:focus {
        border-color: rgba(77,140,250,0.5) !important;
        box-shadow: 0 0 0 3px rgba(77,140,250,0.08) !important;
        outline: none !important;
      }
      [data-testid="stSelectbox"] > div > div {
        background: var(--surface2) !important;
        border-color: var(--border2) !important;
        color: var(--text) !important;
        border-radius: var(--radius) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.85rem !important;
      }
      [data-testid="stTextInput"] label,
      [data-testid="stNumberInput"] label,
      [data-testid="stSelectbox"] label,
      [data-testid="stRadio"] label {
        color: var(--muted) !important;
        font-size: 0.72rem !important;
        font-family: var(--font-mono) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-weight: 400 !important;
      }

      /* Primary button */
      [data-testid="stButton"] > button {
        background: linear-gradient(135deg, var(--blue) 0%, var(--violet) 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: var(--radius) !important;
        font-family: var(--font-display) !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        padding: 0.55rem 1.6rem !important;
        transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
        box-shadow: 0 4px 16px rgba(77,140,250,0.2) !important;
        font-size: 0.9rem !important;
      }
      [data-testid="stButton"] > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(77,140,250,0.3) !important;
      }
      [data-testid="stButton"] > button:active {
        transform: translateY(0) !important;
      }

      /* Tabs */
      [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 1px solid var(--border) !important;
        gap: 2px !important;
        background: transparent !important;
      }
      [data-testid="stTabs"] button[role="tab"] {
        font-family: var(--font-mono) !important;
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: var(--muted) !important;
        background: transparent !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 8px 8px 0 0 !important;
        transition: color 0.2s !important;
      }
      [data-testid="stTabs"] button[role="tab"]:hover {
        color: var(--text2) !important;
      }
      [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--blue) !important;
        border-bottom: 2px solid var(--blue) !important;
        background: var(--blue-dim) !important;
      }

      /* Dataframe */
      [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        overflow: hidden !important;
      }

      /* Expander */
      [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        background: var(--surface) !important;
      }
      [data-testid="stExpander"] summary {
        font-family: var(--font-mono) !important;
        font-size: 0.78rem !important;
        color: var(--muted) !important;
        letter-spacing: 0.04em !important;
      }

      /* Radio */
      [data-testid="stRadio"] > div {
        gap: 12px !important;
      }
      [data-testid="stRadio"] label {
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        border-radius: var(--radius) !important;
        padding: 8px 14px !important;
        transition: border-color 0.2s !important;
      }
      [data-testid="stRadio"] label:has(input:checked) {
        border-color: rgba(77,140,250,0.4) !important;
        background: var(--blue-dim) !important;
      }

      /* ── Section headings ─────────────────────────────────── */
      .section-heading {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--faint);
        margin: 24px 0 12px 0;
        display: flex; align-items: center; gap: 10px;
      }
      .section-heading::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border), transparent);
      }

      /* ── Stat cards ───────────────────────────────────────── */
      .stat-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s, transform 0.2s;
      }
      .stat-card:hover {
        border-color: var(--border2);
        transform: translateY(-2px);
      }
      .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(77,140,250,0.2), transparent);
      }
      .stat-number {
        font-family: var(--font-display);
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.05em;
        line-height: 1;
      }
      .stat-label {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted);
        margin-top: 6px;
      }

      /* ── Placeholder ──────────────────────────────────────── */
      .result-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 16px;
        padding: 60px 20px;
        color: var(--muted);
        text-align: center;
      }
      .placeholder-icon {
        font-size: 3rem;
        opacity: 0.25;
        filter: grayscale(0.5);
      }
      .placeholder-text {
        font-family: var(--font-mono);
        font-size: 0.8rem;
        line-height: 1.8;
        color: var(--faint);
        max-width: 260px;
      }
      .placeholder-text strong {
        color: var(--muted);
        font-weight: 700;
      }

      /* ── Alerts ───────────────────────────────────────────── */
      [data-testid="stAlert"] {
        border-radius: var(--radius) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.82rem !important;
      }

      /* ── Scrollbar ────────────────────────────────────────── */
      ::-webkit-scrollbar { width: 6px; height: 6px; }
      ::-webkit-scrollbar-track { background: var(--bg); }
      ::-webkit-scrollbar-thumb {
        background: var(--border2);
        border-radius: 999px;
      }
      ::-webkit-scrollbar-thumb:hover { background: var(--border3); }

      /* ── Spinner ──────────────────────────────────────────── */
      [data-testid="stSpinner"] {
        color: var(--blue) !important;
      }

      /* ── Code block ───────────────────────────────────────── */
      [data-testid="stCode"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
      }

      /* Caption */
      [data-testid="stCaptionContainer"] p {
        font-family: var(--font-mono) !important;
        font-size: 0.72rem !important;
        color: var(--muted) !important;
        letter-spacing: 0.04em !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
      <div class="app-logo">🛡️</div>
      <div>
        <div class="app-title">FraudShield</div>
        <div class="app-subtitle">E-commerce Risk Intelligence</div>
      </div>
      <div class="live-badge">
        <div class="live-dot"></div>
        Hybrid Scoring Active
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def call_score(payload: dict):
    return requests.post(f"{API}/score", json={"payload": payload}, timeout=60)


def call_score_batch(payloads: list[dict]):
    return requests.post(f"{API}/score_batch", json={"payloads": payloads}, timeout=300)


def friendly_action(band: str):
    if band == "HIGH":
        return "Manual review recommended"
    if band == "MEDIUM":
        return "Step-up verification recommended (OTP / 3DS / KYC)"
    return "Approve (low risk)"


@st.cache_data
def load_sample_df():
    df = pd.read_parquet("data/processed/X_test.parquet")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def json_safe_payload_from_series(s: pd.Series) -> dict:
    s = s.replace([np.inf, -np.inf], np.nan)
    d = s.to_dict()
    out = {}
    for k, v in d.items():
        if pd.isna(v):
            out[k] = None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def df_to_json_safe_records(df: pd.DataFrame) -> list[dict]:
    df = df.replace([np.inf, -np.inf], np.nan).astype("object")

    def to_py(v):
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, (np.floating,)):
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(v, (np.integer,)):
            return int(v)
        return v

    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")
    return [{k: to_py(v) for k, v in r.items()} for r in records]


def _band_color_class(band: str) -> str:
    return {"HIGH": "high", "MEDIUM": "med", "LOW": "low"}.get(band, "low")


def _band_icon(band: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(band, "⚪")


def render_risk_result(title: str, score: int, band: str, reasons: list[str] | None = None, extra_metrics: dict | None = None):
    c = _band_color_class(band)
    icon = _band_icon(band)

    reasons_html = ""
    if reasons:
        tags = "".join(f'<span class="reason-tag">{r}</span>' for r in reasons[:8])
        reasons_html = f'<div style="margin-top:14px;">{tags}</div>'

    extra_html = ""
    if extra_metrics:
        items = "".join(
            f'<div class="extra-metric-card">'
            f'<div class="extra-metric-label">{k}</div>'
            f'<div class="extra-metric-value">{v}</div>'
            f'</div>'
            for k, v in extra_metrics.items()
        )
        extra_html = f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;">{items}</div>'

    st.markdown(
        f"""
        <div class="band-wrap band-{c}">
          <div class="band-label band-label-{c}">{icon} {title}</div>
          <div class="band-score score-{c}">{score}</div>
          <div class="score-bar-track"><div class="score-bar-fill fill-{c}" style="width:{score}%;"></div></div>
          <div class="band-name" style="color:var(--{c});">{band} RISK</div>
          <div class="action-text">⟶ {friendly_action(band)}</div>
          {reasons_html}
          {extra_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_form_state():
    defaults = {
        "tx_amt": 125.0,
        "tx_dt": 100000,
        "product_cd": "W",
        "card4": "visa",
        "card6": "debit",
        "addr1": 315,
        "addr2": 87,
        "p_email": "gmail.com",
        "r_email": "gmail.com",
        "device_type": "desktop",
        "device_info": "Windows",
        "id_30": "Windows 10",
        "id_31": "chrome 120",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def load_preset(name: str):
    presets = {
        "low": {
            "tx_amt": 42.0, "tx_dt": 36000, "product_cd": "W", "card4": "visa", "card6": "debit",
            "addr1": 315, "addr2": 87, "p_email": "gmail.com", "r_email": "gmail.com",
            "device_type": "desktop", "device_info": "Windows", "id_30": "Windows 10", "id_31": "chrome 120",
        },
        "medium": {
            "tx_amt": 420.0, "tx_dt": 83000, "product_cd": "C", "card4": "mastercard", "card6": "credit",
            "addr1": 299, "addr2": 60, "p_email": "gmail.com", "r_email": "outlook.com",
            "device_type": "mobile", "device_info": "iPhone", "id_30": "iOS 16", "id_31": "chrome 120",
        },
        "high": {
            "tx_amt": 2450.0, "tx_dt": 10000, "product_cd": "H", "card4": "discover", "card6": "credit",
            "addr1": 0, "addr2": 0, "p_email": "mail-temp.cc", "r_email": "unknown-mail.cc",
            "device_type": "mobile", "device_info": "selenium-headless", "id_30": "Windows 10", "id_31": "python-requests",
        },
    }
    for k, v in presets[name].items():
        st.session_state[k] = v


init_form_state()

tab1, tab2, tab3 = st.tabs(["⬡  Single Transaction", "⬡  Batch Scoring", "⬡  Admin / History"])

with tab1:
    st.markdown('<div class="section-heading">Input Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Choose how to provide input",
        ["User-friendly form (recommended)", "Pick a real transaction from dataset"],
        horizontal=True,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1.15, 1], gap="large")
    payload: dict = {}
    score_btn = False

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if mode == "Pick a real transaction from dataset":
            st.markdown(
                '<span class="pill pill-accent">📊 Full IEEE row → scored by the trained XGBoost bundle</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            df_samples = load_sample_df()
            cA, cB = st.columns([1, 1])
            with cA:
                idx = st.number_input("Row index", min_value=0, max_value=len(df_samples) - 1, value=0, step=1)
            with cB:
                st.markdown("<br>", unsafe_allow_html=True)
                random_pick = st.button("🎲 Random row", use_container_width=True)

            show_cols = st.multiselect(
                "Preview columns",
                options=[
                    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
                    "card1", "card2", "card3", "card4", "card5", "card6",
                    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
                    "DeviceType", "DeviceInfo", "id_30", "id_31",
                ],
                default=["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "P_emaildomain", "DeviceType"],
            )

            if random_pick:
                idx = int(np.random.randint(0, len(df_samples)))

            row = df_samples.iloc[int(idx)]
            st.markdown('<div class="section-heading">Row Preview</div>', unsafe_allow_html=True)
            preview = row[show_cols].to_frame("value") if show_cols else row.head(20).to_frame("value")
            st.dataframe(preview, use_container_width=True)
            payload = json_safe_payload_from_series(row)
        else:
            st.markdown(
                '<span class="pill pill-accent">✨ Interactive mode → API uses a rule-based demo scorer designed for user-enterable fields</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            p1, p2, p3 = st.columns(3)
            with p1:
                if st.button("Load Low Risk Example", use_container_width=True):
                    load_preset("low")
            with p2:
                if st.button("Load Medium Risk Example", use_container_width=True):
                    load_preset("medium")
            with p3:
                if st.button("Load High Risk Example", use_container_width=True):
                    load_preset("high")

            st.markdown('<div class="section-heading">Transaction Details</div>', unsafe_allow_html=True)
            cA, cB, cC = st.columns(3)
            with cA:
                amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=1.0, key="tx_amt")
            with cB:
                tx_dt = st.number_input("TransactionDT (seconds)", min_value=0, step=1000, key="tx_dt")
            with cC:
                product_cd = st.selectbox("Product Type (ProductCD)", ["W", "C", "R", "H", "S"], key="product_cd")

            st.markdown('<div class="section-heading">Payment Information</div>', unsafe_allow_html=True)
            cA, cB, cC = st.columns(3)
            with cA:
                card4 = st.selectbox("Card Brand", ["visa", "mastercard", "american express", "discover", "other"], key="card4")
            with cB:
                card6 = st.selectbox("Card Type", ["debit", "credit", "charge card", "other"], key="card6")
            with cC:
                device_type = st.selectbox("Device Type", ["desktop", "mobile", "other"], key="device_type")

            st.markdown('<div class="section-heading">Customer / Device Signals</div>', unsafe_allow_html=True)
            cA, cB = st.columns(2)
            with cA:
                p_email = st.text_input("Purchaser Email Domain", key="p_email", placeholder="gmail.com / NotFound")
                addr1 = st.number_input("Address Region ID 1 (addr1)", min_value=0, step=1, key="addr1")
                device_info = st.text_input("Device Info", key="device_info", placeholder="Windows / iPhone / NotFound")
            with cB:
                r_email = st.text_input("Recipient Email Domain", key="r_email", placeholder="gmail.com / NotFound")
                addr2 = st.number_input("Address Region ID 2 (addr2)", min_value=0, step=1, key="addr2")
                id_30 = st.text_input("Operating System (id_30)", key="id_30", placeholder="Windows 10 / iOS 16")

            id_31 = st.text_input("Browser / Client (id_31)", key="id_31", placeholder="chrome 120 / safari / python-requests")

            payload = {
                "TransactionDT": int(tx_dt),
                "TransactionAmt": float(amount),
                "ProductCD": product_cd,
                "card4": card4,
                "card6": card6,
                "DeviceType": device_type,
            }
            if p_email.strip():
                payload["P_emaildomain"] = p_email.strip()
            if r_email.strip():
                payload["R_emaildomain"] = r_email.strip()
            if addr1 > 0:
                payload["addr1"] = int(addr1)
            if addr2 > 0:
                payload["addr2"] = int(addr2)
            if device_info.strip():
                payload["DeviceInfo"] = device_info.strip()
            if id_30.strip():
                payload["id_30"] = id_30.strip()
            if id_31.strip():
                payload["id_31"] = id_31.strip()

        st.markdown("<br>", unsafe_allow_html=True)
        score_btn = st.button("⟶ Score Transaction", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Risk Analysis</div>', unsafe_allow_html=True)

        if score_btn:
            with st.spinner("Calling scoring engine…"):
                try:
                    resp = call_score(payload)
                    if resp.status_code != 200:
                        st.error(f"API error ({resp.status_code}): {resp.text}")
                    else:
                        out = resp.json()
                        render_risk_result(
                            title="Final Risk Score",
                            score=out["risk_score"],
                            band=out["risk_band"],
                            reasons=out.get("reasons", []),
                            extra_metrics={
                                "Fraud Probability": f"{out['fraud_proba']:.6f}",
                                "Scoring Engine": out.get("model_used", "n/a"),
                            },
                        )
                        with st.expander("Raw API response", expanded=False):
                            st.json(out)
                except Exception as e:
                    st.error(f"Request failed: {e}")
        else:
            st.markdown(
                """
                <div class="result-placeholder">
                  <div class="placeholder-icon">🛡️</div>
                  <div class="placeholder-text">
                    Fill the form or select a dataset row,<br>
                    then click <strong>Score Transaction</strong>.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔍 Built payload (debug)", expanded=False):
        st.code(json.dumps(payload, indent=2), language="json")

with tab2:
    st.markdown('<div class="section-heading">CSV Upload</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV with interactive columns or full IEEE-style transaction columns.")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.markdown('<div class="section-heading">Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)

        cA, _ = st.columns([1, 3])
        with cA:
            max_rows = st.number_input("Max rows to score", min_value=1, max_value=2000, value=200)

        if st.button("⟶ Score Batch", type="primary"):
            try:
                df_small = df.head(int(max_rows))
                payloads = df_to_json_safe_records(df_small)
                with st.spinner("Scoring…"):
                    r = call_score_batch(payloads)

                if r.status_code != 200:
                    st.error(f"API error ({r.status_code}): {r.text}")
                    st.stop()

                out = r.json()
                st.success("✅ Batch scored successfully")

                counts = out.get("counts", {})
                cA, cB, cC = st.columns(3)
                for col, band, color in [(cA, "LOW", "var(--low)"), (cB, "MEDIUM", "var(--med)"), (cC, "HIGH", "var(--high)")]:
                    with col:
                        st.markdown(
                            f'<div class="stat-card"><div class="stat-number" style="color:{color};">{counts.get(band, 0)}</div><div class="stat-label">{band} Risk</div></div>',
                            unsafe_allow_html=True,
                        )

                res_df = pd.DataFrame(out["results"])
                joined = pd.concat([df_small.reset_index(drop=True), res_df], axis=1)
                st.markdown('<div class="section-heading">Scored Results</div>', unsafe_allow_html=True)
                st.dataframe(joined.head(50), use_container_width=True)

                csv_bytes = joined.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download scored CSV", data=csv_bytes, file_name="batch_scored.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch scoring failed: {e}")

    st.markdown("---")
    st.markdown('<div class="section-heading">Score from Dataset</div>', unsafe_allow_html=True)
    st.caption("Fast test: score real rows from X_test.parquet using the trained XGBoost bundle")

    cA, cB, _ = st.columns([1, 1, 2])
    with cA:
        n = st.number_input("Rows from X_test.parquet", min_value=1, max_value=2000, value=200, step=50)
    with cB:
        st.markdown("<br>", unsafe_allow_html=True)
        random_rows = st.checkbox("Pick random rows", value=True)

    if st.button("⟶ Score dataset rows", type="primary"):
        try:
            df_samples = load_sample_df()
            df_pick = df_samples.sample(int(n), random_state=int(np.random.randint(1, 10_000))) if random_rows else df_samples.head(int(n))
            payloads = df_to_json_safe_records(df_pick)
            with st.spinner("Scoring…"):
                r = call_score_batch(payloads)

            if r.status_code != 200:
                st.error(f"API error ({r.status_code}): {r.text}")
                st.stop()

            out = r.json()
            st.success("✅ Dataset batch scored successfully")

            counts = out.get("counts", {})
            cA, cB, cC = st.columns(3)
            for col, band, color in [(cA, "LOW", "var(--low)"), (cB, "MEDIUM", "var(--med)"), (cC, "HIGH", "var(--high)")]:
                with col:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-number" style="color:{color};">{counts.get(band, 0)}</div><div class="stat-label">{band} Risk</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="section-heading">Top 10 Riskiest Transactions</div>', unsafe_allow_html=True)
            st.json(out["top10"])
        except Exception as e:
            st.error(f"Dataset batch scoring failed: {e}")

with tab3:
    st.markdown('<div class="section-heading">Scoring History</div>', unsafe_allow_html=True)
    cA, _ = st.columns([1, 4])
    with cA:
        limit = st.number_input("Rows to fetch", min_value=10, max_value=500, value=50, step=10)
        refresh = st.button("⟳ Refresh", type="primary", use_container_width=True)

    if refresh:
        try:
            with st.spinner("Loading history…"):
                r = requests.get(f"{API}/admin/recent?limit={int(limit)}", timeout=60)

            if r.status_code != 200:
                st.error(f"API error ({r.status_code}): {r.text}")
                st.stop()

            items = r.json().get("items", [])
            if not items:
                st.info("No history yet. Score something first.")
                st.stop()

            rows = []
            for it in items:
                rows.append(
                    {
                        "created_at": it["created_at"],
                        "fraud_proba": it["fraud_proba"],
                        "risk_score": it["risk_score"],
                        "risk_band": it["risk_band"],
                        "payload": json.dumps(it["payload"]),
                    }
                )

            hist = pd.DataFrame(rows)
            counts = hist["risk_band"].value_counts()
            cA, cB, cC = st.columns(3)
            for col, band, color in [(cA, "LOW", "var(--low)"), (cB, "MEDIUM", "var(--med)"), (cC, "HIGH", "var(--high)")]:
                with col:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-number" style="color:{color};">{counts.get(band, 0)}</div><div class="stat-label">{band} Risk</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="section-heading">History Table</div>', unsafe_allow_html=True)
            st.dataframe(hist, use_container_width=True)
            st.markdown('<div class="section-heading">Band Distribution</div>', unsafe_allow_html=True)
            st.bar_chart(hist["risk_band"].value_counts())
        except Exception as e:
            st.error(f"Failed to load history: {e}")
