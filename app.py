import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="예측 정비 AI",
    page_icon="⚙️",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Noto+Sans+KR:wght@400;500;700&display=swap');
* { box-sizing: border-box; }
.stApp {
    background: #020508;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(0,180,255,0.05) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 80%, rgba(0,255,136,0.03) 0%, transparent 55%),
        linear-gradient(rgba(0,180,255,0.012) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,180,255,0.012) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
}
[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"]  { background: transparent; }
.hud-sl {
    position:fixed; top:0; left:0; right:0; bottom:0; pointer-events:none;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,180,255,.01) 2px,rgba(0,180,255,.01) 4px);
    z-index:1;
}
.hdr-wrap { text-align:center; padding:2rem 0 .6rem; }
.hdr-eyebrow { font-family:'Share Tech Mono',monospace; font-size:.6rem; color:#00b4ff; letter-spacing:.32em; margin-bottom:.6rem; opacity:.55; }
.hdr-title {
    font-family:'Orbitron',monospace; font-size:1.85rem; font-weight:900;
    background:linear-gradient(135deg,#00b4ff 0%,#ffffff 45%,#00ff88 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    letter-spacing:.12em; margin:0; line-height:1.25;
}
.hdr-ko  { font-family:'Noto Sans KR',sans-serif; font-size:.9rem; font-weight:500; color:#1e4060; letter-spacing:.1em; margin-top:.45rem; }
.hdr-sub { font-family:'Share Tech Mono',monospace; color:#0a1e2e; font-size:.58rem; letter-spacing:.15em; margin-top:.3rem; }
.radar-wrap { display:flex; justify-content:center; margin:.6rem 0 1.2rem; }
.radar-svg  { filter:drop-shadow(0 0 12px rgba(0,180,255,.3)); }
.sec {
    font-family:'Orbitron',monospace; font-size:.56rem; color:#00b4ff;
    letter-spacing:.26em; padding:.4rem .8rem;
    border-left:2px solid #00b4ff;
    background:linear-gradient(90deg,rgba(0,180,255,.06),transparent);
    margin:1.3rem 0 .85rem;
}
.sec-ko { font-family:'Noto Sans KR',sans-serif; font-size:.73rem; color:#1e4060; font-weight:500; margin-left:.3rem; letter-spacing:.04em; }
.stSlider label, .stSelectbox label {
    font-family:'Noto Sans KR',sans-serif !important;
    color:#2a4a60 !important; font-size:.82rem !important; font-weight:500 !important;
}
.stButton > button {
    width:100%; background:transparent !important;
    border:1px solid #00b4ff !important; color:#00b4ff !important;
    font-family:'Orbitron',monospace !important; font-weight:700 !important;
    font-size:.82rem !important; letter-spacing:.18em !important;
    border-radius:3px !important; padding:.9rem !important; margin-top:.6rem;
    box-shadow:0 0 20px rgba(0,180,255,.1),inset 0 0 20px rgba(0,180,255,.03) !important;
    transition:all .3s;
}
.stButton > button:hover {
    background:rgba(0,180,255,.07) !important;
    box-shadow:0 0 40px rgba(0,180,255,.3),inset 0 0 30px rgba(0,180,255,.07) !important;
    color:#fff !important;
}
.prob-lbl { font-family:'Noto Sans KR',sans-serif; font-size:.75rem; color:#0a2535; text-align:center; letter-spacing:.06em; margin:.7rem 0 .25rem; }
.stProgress > div > div > div {
    background:linear-gradient(90deg,#00b4ff,#00ff88) !important;
    border-radius:1px !important; box-shadow:0 0 8px rgba(0,180,255,.4) !important;
}
.hud-lights { display:flex; gap:10px; justify-content:center; margin:.8rem 0 .5rem; }
.hud-lw { display:flex; flex-direction:column; align-items:center; gap:4px; }
.hud-l  { width:13px; height:13px; border-radius:50%; background:#050d14; border:1px solid #0a1a2a; }
.hud-ll { font-family:'Share Tech Mono',monospace; font-size:.44rem; color:#0a1a2a; }
.hl-ok  { background:#00ff88; box-shadow:0 0 10px #00ff88,0 0 20px rgba(0,255,136,.4); border-color:#00ff88; }
.hl-err { background:#ff2200; box-shadow:0 0 10px #ff2200,0 0 20px rgba(255,34,0,.4); border-color:#ff2200; animation:wlblink .35s ease-in-out infinite; }
@keyframes wlblink{0%,100%{opacity:1;}50%{opacity:.08;}}
.m-row { display:flex; gap:6px; margin:.8rem 0; flex-wrap:wrap; }
.m-box { flex:1; min-width:70px; background:#030810; border:1px solid #0a1520; border-radius:3px; padding:.6rem .35rem; text-align:center; position:relative; }
.m-box::before { content:''; position:absolute; top:0; left:20%; right:20%; height:1px; background:linear-gradient(90deg,transparent,#00b4ff22,transparent); }
.m-val { font-family:'Orbitron',monospace; font-size:.78rem; font-weight:700; color:#00b4ff; text-shadow:0 0 8px rgba(0,180,255,.4); }
.m-lbl { font-family:'Noto Sans KR',sans-serif; font-size:.56rem; color:#0a1e2e; margin-top:3px; }
hr { border-color:#050d14 !important; }
.footer { font-family:'Noto Sans KR',sans-serif; color:#080f18; font-size:.66rem; text-align:center; letter-spacing:.06em; padding:1.2rem 0 .4rem; }
</style>
<div class="hud-sl"></div>
""", unsafe_allow_html=True)


# ── 헤더
st.markdown("""
<div class="hdr-wrap">
    <div class="hdr-eyebrow">산업용 기계 모니터링 시스템 · INDUSTRIAL MACHINE MONITORING SYSTEM · v2.4.1</div>
    <div class="hdr-title">PREDICTIVE<br>MAINTENANCE AI</div>
    <div class="hdr-ko">산업 설비 고장 예측 진단 시스템</div>
    <div class="hdr-sub">AI4I 2020 DATASET · RANDOM FOREST + SMOTE · REAL-TIME DIAGNOSIS</div>
</div>
<div class="radar-wrap">
<svg class="radar-svg" width="160" height="160" viewBox="0 0 160 160">
    <defs><radialGradient id="rg" cx="50%" cy="50%" r="50%">
        <stop offset="0%"   stop-color="#00b4ff" stop-opacity="0.18"/>
        <stop offset="100%" stop-color="#00b4ff" stop-opacity="0"/>
    </radialGradient></defs>
    <circle cx="80" cy="80" r="75" fill="none" stroke="#0a1a2a" stroke-width="1"/>
    <circle cx="80" cy="80" r="52" fill="none" stroke="#0a1a2a" stroke-width="1"/>
    <circle cx="80" cy="80" r="28" fill="none" stroke="#0a1a2a" stroke-width="1"/>
    <line x1="80" y1="5"   x2="80"  y2="155" stroke="#0a1a2a" stroke-width="1"/>
    <line x1="5"  y1="80"  x2="155" y2="80"  stroke="#0a1a2a" stroke-width="1"/>
    <g transform-origin="80 80">
        <animateTransform attributeName="transform" type="rotate" from="0 80 80" to="360 80 80" dur="2.5s" repeatCount="indefinite"/>
        <path d="M80,80 L80,5 A75,75 0 0,1 146,117 Z" fill="url(#rg)"/>
        <line x1="80" y1="80" x2="80" y2="5" stroke="#00b4ff" stroke-width="1.5" opacity=".85"/>
    </g>
    <circle cx="108" cy="47"  r="3"   fill="#00b4ff" opacity=".8"><animate attributeName="opacity" values=".8;0;.8" dur="2.5s" repeatCount="indefinite"/></circle>
    <circle cx="52"  cy="104" r="2.5" fill="#00ff88" opacity=".7"><animate attributeName="opacity" values=".7;0;.7" dur="1.8s" repeatCount="indefinite"/></circle>
    <circle cx="80"  cy="80"  r="4"   fill="#00b4ff" opacity=".9"/>
    <circle cx="80"  cy="80"  r="2"   fill="#fff"/>
</svg>
</div>
""", unsafe_allow_html=True)


# ── 모델 로드
@st.cache_resource
def load_model():
    paths_m = ['/content/drive/MyDrive/data/engine_model.pkl','/content/drive/MyDrive/engine_model.pkl','engine_model.pkl']
    paths_s = ['/content/drive/MyDrive/data/engine_scaler.pkl','/content/drive/MyDrive/engine_scaler.pkl','engine_scaler.pkl']
    model, scaler = None, None
    for p in paths_m:
        if os.path.exists(p): model  = joblib.load(p); break
    for p in paths_s:
        if os.path.exists(p): scaler = joblib.load(p); break
    return model, scaler

model, scaler = load_model()
if model is None or scaler is None:
    st.warning("⚠ engine_model.pkl / engine_scaler.pkl 파일을 같은 경로에 업로드해주세요.")
    st.stop()


# ── 입력
st.markdown('<div class="sec">센서 입력 <span class="sec-ko">— 설비 센서 데이터를 입력하세요</span></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    product_type = st.selectbox("🏷 제품 유형", options=[0,1,2], format_func=lambda x:{0:"H — 고급형",1:"L — 일반형",2:"M — 중급형"}[x])
    air_temp     = st.slider("🌡 공기 온도 (K)",    295.0, 305.0, 298.0, 0.1)
    process_temp = st.slider("🔥 공정 온도 (K)",    305.0, 315.0, 308.0, 0.1)
with col2:
    rot_speed    = st.slider("🔄 회전 속도 (rpm)",  1000,  3000,  1500,  10)
    torque       = st.slider("⚡ 토크 (Nm)",         3.0,   80.0,  40.0,  0.5)
    tool_wear    = st.slider("🔧 공구 마모도 (min)", 0,     250,   100,   1)

st.markdown(f"""
<div class="m-row">
    <div class="m-box"><div class="m-val">{["H","L","M"][product_type]}</div><div class="m-lbl">제품유형</div></div>
    <div class="m-box"><div class="m-val">{air_temp:.1f}</div><div class="m-lbl">공기온도 K</div></div>
    <div class="m-box"><div class="m-val">{process_temp:.1f}</div><div class="m-lbl">공정온도 K</div></div>
    <div class="m-box"><div class="m-val">{rot_speed}</div><div class="m-lbl">회전속도 rpm</div></div>
    <div class="m-box"><div class="m-val">{torque:.1f}</div><div class="m-lbl">토크 Nm</div></div>
    <div class="m-box"><div class="m-val">{tool_wear}</div><div class="m-lbl">공구마모도</div></div>
</div>
""", unsafe_allow_html=True)


# ── 진단 버튼
st.markdown('<div class="sec">진단 실행 <span class="sec-ko">— 버튼을 눌러 AI 진단을 시작하세요</span></div>', unsafe_allow_html=True)

if st.button("◈   AI 진단 시작 — INITIATE DIAGNOSIS"):

    input_df = pd.DataFrame(
        [[product_type, air_temp, process_temp, rot_speed, torque, tool_wear]],
        columns=["제품유형","공기온도","공정온도","회전속도","토크","공구마모도"]
    )
    scaled = scaler.transform(input_df)
    pred   = model.predict(scaled)
    proba  = model.predict_proba(scaled)[0]
    is_ok  = (pred[0] == 0)
    conf   = proba[0] if is_ok else proba[1]

    # ── 진단중 + 결과 풀사이즈 애니메이션 (components.html)
    if is_ok:
        components.html("""
        <!DOCTYPE html><html><head>
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Noto+Sans+KR:wght@500&display=swap');
        *{margin:0;padding:0;box-sizing:border-box;}
        body{background:#020e08;display:flex;justify-content:center;align-items:center;height:100vh;overflow:hidden;position:relative;flex-direction:column;}

        /* 스캔 페이즈 */
        #scan{position:absolute;top:0;left:0;right:0;bottom:0;background:#020508;display:flex;justify-content:center;align-items:center;flex-direction:column;animation:scan-out .4s ease 2.2s forwards;}
        @keyframes scan-out{0%{opacity:1;}100%{opacity:0;pointer-events:none;}}
        .radar{filter:drop-shadow(0 0 20px rgba(0,180,255,0.7));margin-bottom:1.4rem;}
        .stxt{font-family:'Orbitron',monospace;font-size:.82rem;font-weight:700;color:#00b4ff;letter-spacing:.3em;text-shadow:0 0 20px rgba(0,180,255,.9);animation:blink .8s ease-in-out infinite;}
        @keyframes blink{0%,100%{opacity:1;}50%{opacity:.15;}}
        .bw{width:250px;height:2px;background:#0a1520;border-radius:2px;margin-top:1.3rem;overflow:hidden;}
        .bar{height:100%;width:0%;background:linear-gradient(90deg,#00b4ff,#00ff88);box-shadow:0 0 10px #00b4ff;animation:load 2s ease forwards;}
        @keyframes load{0%{width:0%;}100%{width:100%;}}

        /* 결과 페이즈 */
        #result{position:absolute;top:0;left:0;right:0;bottom:0;background:#020e08;display:flex;justify-content:center;align-items:center;flex-direction:column;opacity:0;animation:res-in .6s ease 2.4s forwards;}
        @keyframes res-in{0%{opacity:0;}100%{opacity:1;}}

        .ring{
            position:absolute;top:50%;left:50%;border-radius:50%;
            border:2px solid #00ff88;
            transform:translate(-50%,-50%) scale(0);
            box-shadow:0 0 30px #00ff88,inset 0 0 20px rgba(0,255,136,0.06);
            animation:ring-out 3s ease 2.5s forwards;
        }
        .ring:nth-child(1){width:180px;height:180px;animation-delay:2.5s;}
        .ring:nth-child(2){width:360px;height:360px;animation-delay:2.7s;}
        .ring:nth-child(3){width:560px;height:560px;animation-delay:2.9s;}
        .ring:nth-child(4){width:800px;height:800px;animation-delay:3.1s;}
        @keyframes ring-out{0%{transform:translate(-50%,-50%) scale(0);opacity:1;}80%{opacity:.4;}100%{transform:translate(-50%,-50%) scale(1);opacity:0;}}

        .center{text-align:center;position:relative;z-index:2;animation:pop .7s cubic-bezier(.34,1.56,.64,1) 2.6s both;}
        @keyframes pop{from{transform:scale(0);opacity:0;}to{transform:scale(1);opacity:1;}}
        .icon{font-size:5rem;filter:drop-shadow(0 0 24px #00ff88);margin-bottom:.4rem;}
        .title{font-family:'Orbitron',monospace;font-size:2.4rem;font-weight:900;color:#00ff88;letter-spacing:.15em;text-shadow:0 0 40px #00ff88,0 0 80px rgba(0,255,136,.5);margin:.2rem 0;}
        .sub{font-family:'Noto Sans KR',sans-serif;font-size:.9rem;color:#00aa44;margin-top:.3rem;}
        </style></head><body>
        <div id="scan">
            <svg class="radar" width="190" height="190" viewBox="0 0 190 190">
                <defs><radialGradient id="rg" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#00b4ff" stop-opacity="0.22"/><stop offset="100%" stop-color="#00b4ff" stop-opacity="0"/></radialGradient></defs>
                <circle cx="95" cy="95" r="90" fill="none" stroke="#0a1a2a" stroke-width="1"/>
                <circle cx="95" cy="95" r="62" fill="none" stroke="#0a1a2a" stroke-width="1"/>
                <circle cx="95" cy="95" r="36" fill="none" stroke="#0a1a2a" stroke-width="1"/>
                <circle cx="95" cy="95" r="14" fill="none" stroke="#0a1a2a" stroke-width="1"/>
                <line x1="95" y1="5"   x2="95"  y2="185" stroke="#0a1a2a" stroke-width="1"/>
                <line x1="5"  y1="95"  x2="185" y2="95"  stroke="#0a1a2a" stroke-width="1"/>
                <g transform-origin="95 95">
                    <animateTransform attributeName="transform" type="rotate" from="0 95 95" to="360 95 95" dur="1.6s" repeatCount="indefinite"/>
                    <path d="M95,95 L95,5 A90,90 0 0,1 176,140 Z" fill="url(#rg)"/>
                    <line x1="95" y1="95" x2="95" y2="5" stroke="#00b4ff" stroke-width="2" opacity=".9"/>
                </g>
                <circle cx="128" cy="52" r="4" fill="#00b4ff"><animate attributeName="opacity" values="1;0;1" dur="1.6s" repeatCount="indefinite"/></circle>
                <circle cx="60" cy="132" r="3" fill="#00ff88"><animate attributeName="opacity" values="1;0;1" dur="1.3s" repeatCount="indefinite"/></circle>
                <circle cx="95" cy="95"  r="5"   fill="#00b4ff" opacity=".9"/>
                <circle cx="95" cy="95"  r="2.5" fill="#fff"/>
            </svg>
            <div class="stxt">진단 중 . . .</div>
            <div class="bw"><div class="bar"></div></div>
        </div>
        <div id="result">
            <div class="ring"></div><div class="ring"></div><div class="ring"></div><div class="ring"></div>
            <div class="center">
                <div class="icon">✅</div>
                <div class="title">SYSTEM NORMAL</div>
                <div class="sub">설비 상태 정상 — 모든 센서 수치가 정상 범위 내에 있습니다</div>
            </div>
        </div>
        </body></html>
        """, height=500)

    else:
        components.html(f"""
        <!DOCTYPE html><html><head>
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Noto+Sans+KR:wght@500&display=swap');
        *{{margin:0;padding:0;box-sizing:border-box;}}
        body{{background:#020508;display:flex;justify-content:center;align-items:center;height:100vh;overflow:hidden;position:relative;}}

        /* 스캔 페이즈 */
        #scan{{position:absolute;top:0;left:0;right:0;bottom:0;background:#020508;display:flex;justify-content:center;align-items:center;flex-direction:column;animation:scan-out .4s ease 2.2s forwards;}}
        @keyframes scan-out{{0%{{opacity:1;}}100%{{opacity:0;pointer-events:none;}}}}
        .radar{{filter:drop-shadow(0 0 20px rgba(0,180,255,0.7));margin-bottom:1.4rem;}}
        .stxt{{font-family:'Orbitron',monospace;font-size:.82rem;font-weight:700;color:#00b4ff;letter-spacing:.3em;text-shadow:0 0 20px rgba(0,180,255,.9);animation:blink .8s ease-in-out infinite;}}
        @keyframes blink{{0%,100%{{opacity:1;}}50%{{opacity:.15;}}}}
        .bw{{width:250px;height:2px;background:#0a1520;border-radius:2px;margin-top:1.3rem;overflow:hidden;}}
        .bar{{height:100%;width:0%;background:linear-gradient(90deg,#00b4ff,#00ff88);box-shadow:0 0 10px #00b4ff;animation:load 2s ease forwards;}}
        @keyframes load{{0%{{width:0%;}}100%{{width:100%;}}}}

        /* 결과 페이즈 */
        #result{{
            position:absolute;top:0;left:0;right:0;bottom:0;
            background:#040000;
            display:flex;justify-content:center;align-items:center;flex-direction:column;
            opacity:0;
            animation:res-in .4s ease 2.3s forwards, flash .16s ease-in-out 6 2.3s;
        }}
        @keyframes res-in{{0%{{opacity:0;}}100%{{opacity:1;}}}}
        @keyframes flash{{0%,100%{{background:#040000;}}50%{{background:#500000;}}}}

        .gf-lights{{display:flex;gap:14px;justify-content:center;margin-bottom:.8rem;animation:fadein .4s ease 2.6s both;}}
        @keyframes fadein{{from{{opacity:0;transform:translateY(10px);}}to{{opacity:1;transform:translateY(0);}}}}
        .fl{{width:16px;height:16px;border-radius:50%;background:#ff2200;border:1px solid #ff4400;box-shadow:0 0 14px #ff2200,0 0 30px rgba(255,34,0,.6);animation:wl .35s ease-in-out infinite;}}
        .fl:nth-child(2){{animation-delay:.07s;}}.fl:nth-child(3){{animation-delay:.14s;}}.fl:nth-child(4){{animation-delay:.21s;}}.fl:nth-child(5){{animation-delay:.28s;}}
        @keyframes wl{{0%,100%{{opacity:1;}}50%{{opacity:.06;}}}}

        .tri{{animation:tri-pop .6s cubic-bezier(.34,1.56,.64,1) 2.5s both;margin:.3rem 0 .7rem;}}
        @keyframes tri-pop{{from{{transform:scale(0) rotate(-12deg);opacity:0;}}to{{transform:scale(1) rotate(0);opacity:1;}}}}
        .tri svg{{filter:drop-shadow(0 0 35px rgba(255,34,0,0.95));}}

        .title{{font-family:'Orbitron',monospace;font-size:2.8rem;font-weight:900;color:#ff2200;letter-spacing:.18em;text-align:center;text-shadow:0 0 60px rgba(255,34,0,1),0 0 120px rgba(255,34,0,.6);animation:blink2 .38s ease-in-out 8 2.7s,fadein .5s ease 2.4s both;}}
        @keyframes blink2{{0%,100%{{opacity:1;}}50%{{opacity:.04;}}}}
        .sub{{font-family:'Noto Sans KR',sans-serif;font-size:.95rem;color:#991100;margin-top:.5rem;text-align:center;animation:fadein .5s ease 2.6s both;}}
        .conf{{font-family:'Share Tech Mono',monospace;font-size:.75rem;color:#441100;margin-top:.9rem;letter-spacing:.15em;animation:fadein .5s ease 2.8s both;}}
        .gf-lights2{{display:flex;gap:14px;justify-content:center;margin-top:.8rem;animation:fadein .5s ease 3s both;}}
        </style></head><body>
        <div id="scan">
            <svg class="radar" width="190" height="190" viewBox="0 0 190 190">
                <defs><radialGradient id="rg" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#00b4ff" stop-opacity="0.22"/><stop offset="100%" stop-color="#00b4ff" stop-opacity="0"/></radialGradient></defs>
                <circle cx="95" cy="95" r="90" fill="none" stroke="#0a1a2a" stroke-width="1"/>
                <circle cx="95" cy="95" r="62" fill="none" stroke="#0a1a2a" stroke-width="1"/>
                <circle cx="95" cy="95" r="36" fill="none" stroke="#0a1a2a" stroke-width="1"/>
                <circle cx="95" cy="95" r="14" fill="none" stroke="#0a1a2a" stroke-width="1"/>
                <line x1="95" y1="5" x2="95" y2="185" stroke="#0a1a2a" stroke-width="1"/>
                <line x1="5" y1="95" x2="185" y2="95" stroke="#0a1a2a" stroke-width="1"/>
                <g transform-origin="95 95">
                    <animateTransform attributeName="transform" type="rotate" from="0 95 95" to="360 95 95" dur="1.6s" repeatCount="indefinite"/>
                    <path d="M95,95 L95,5 A90,90 0 0,1 176,140 Z" fill="url(#rg)"/>
                    <line x1="95" y1="95" x2="95" y2="5" stroke="#00b4ff" stroke-width="2" opacity=".9"/>
                </g>
                <circle cx="128" cy="52" r="4" fill="#00b4ff"><animate attributeName="opacity" values="1;0;1" dur="1.6s" repeatCount="indefinite"/></circle>
                <circle cx="60" cy="132" r="3" fill="#00ff88"><animate attributeName="opacity" values="1;0;1" dur="1.3s" repeatCount="indefinite"/></circle>
                <circle cx="95" cy="95" r="5"   fill="#00b4ff" opacity=".9"/>
                <circle cx="95" cy="95" r="2.5" fill="#fff"/>
            </svg>
            <div class="stxt">진단 중 . . .</div>
            <div class="bw"><div class="bar"></div></div>
        </div>
        <div id="result">
            <div class="gf-lights">
                <div class="fl"></div><div class="fl"></div><div class="fl"></div><div class="fl"></div><div class="fl"></div>
            </div>
            <div class="tri">
                <svg width="160" height="140" viewBox="0 0 160 140">
                    <defs><filter id="g"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
                    <polygon points="80,7 153,133 7,133" fill="rgba(255,34,0,0.1)" stroke="#ff2200" stroke-width="4.5" filter="url(#g)"/>
                    <text x="80" y="108" text-anchor="middle" font-family="Orbitron,monospace" font-size="56" font-weight="900" fill="#ff2200" filter="url(#g)">!</text>
                </svg>
            </div>
            <div class="title">FAILURE DETECTED</div>
            <div class="sub">설비 고장 예측 — 즉시 점검이 필요합니다</div>
            <div class="conf">CONFIDENCE LEVEL — {conf*100:.1f}%</div>
            <div class="gf-lights2">
                <div class="fl"></div><div class="fl"></div><div class="fl"></div><div class="fl"></div><div class="fl"></div>
            </div>
        </div>
        </body></html>
        """, height=500)

    # ── 결과 카드
    st.markdown('<div class="sec">진단 결과 <span class="sec-ko">— AI 분석 완료</span></div>', unsafe_allow_html=True)

    if is_ok:
        st.markdown("""
        <div class="hud-lights">
            <div class="hud-lw"><div class="hud-l hl-ok"></div><div class="hud-ll">PWR</div></div>
            <div class="hud-lw"><div class="hud-l hl-ok"></div><div class="hud-ll">SYS</div></div>
            <div class="hud-lw"><div class="hud-l hl-ok"></div><div class="hud-ll">NET</div></div>
            <div class="hud-lw"><div class="hud-l hl-ok"></div><div class="hud-ll">DAT</div></div>
            <div class="hud-lw"><div class="hud-l hl-ok"></div><div class="hud-ll">AI</div></div>
        </div>
        <div style="background:linear-gradient(135deg,#020e08,#041208);border:1px solid #00ff88;border-radius:4px;padding:1.8rem 1.5rem;text-align:center;box-shadow:0 0 40px rgba(0,255,136,.1);">
            <div style="font-size:1.8rem">✅</div>
            <div style="font-family:'Orbitron',monospace;font-size:1.5rem;font-weight:900;color:#00ff88;letter-spacing:.15em;text-shadow:0 0 24px rgba(0,255,136,.5);margin:.3rem 0;">SYSTEM NORMAL</div>
            <div style="font-family:'Noto Sans KR',sans-serif;font-size:.88rem;color:#007744;margin-top:.3rem;">설비 상태 정상 — 모든 센서 수치가 정상 범위 내에 있습니다</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="hud-lights">
            <div class="hud-lw"><div class="hud-l hl-err"></div><div class="hud-ll">PWR</div></div>
            <div class="hud-lw"><div class="hud-l hl-err"></div><div class="hud-ll">SYS</div></div>
            <div class="hud-lw"><div class="hud-l hl-err"></div><div class="hud-ll">NET</div></div>
            <div class="hud-lw"><div class="hud-l hl-err"></div><div class="hud-ll">DAT</div></div>
            <div class="hud-lw"><div class="hud-l hl-err"></div><div class="hud-ll">AI</div></div>
        </div>
        <div style="background:linear-gradient(135deg,#0e0202,#140303);border:1px solid #ff2200;border-radius:4px;padding:1.8rem 1.5rem;text-align:center;box-shadow:0 0 40px rgba(255,34,0,.15);">
            <div style="font-size:1.8rem">⚠️</div>
            <div style="font-family:'Orbitron',monospace;font-size:1.5rem;font-weight:900;color:#ff2200;letter-spacing:.15em;text-shadow:0 0 24px rgba(255,34,0,.6);margin:.3rem 0;">FAILURE DETECTED</div>
            <div style="font-family:'Noto Sans KR',sans-serif;font-size:.88rem;color:#aa1100;margin-top:.3rem;">설비 고장 예측 — 즉시 점검이 필요합니다</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="prob-lbl">모델 신뢰도 (Confidence Level) — {conf*100:.1f}%</div>', unsafe_allow_html=True)
    st.progress(float(conf))

    # ── 3D 산점도
    st.markdown('<div class="sec">3D 위치 분석 <span class="sec-ko">— 토크 · 회전속도 · 공구마모도 공간에서 현재 입력값 위치</span></div>', unsafe_allow_html=True)

    np.random.seed(42)
    tok_ok = np.random.normal(40, 12, 400); rpm_ok = np.random.normal(1500, 250, 400); tw_ok = np.random.normal(108, 55, 400)
    tok_f  = np.random.normal(62, 10, 80);  rpm_f  = np.random.normal(1280, 200, 80);  tw_f  = np.random.normal(195, 35, 80)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=tok_ok, y=rpm_ok, z=tw_ok, mode='markers', name='정상 데이터',
        marker=dict(size=3, color='#00b4ff', opacity=0.22, line=dict(width=0))))
    fig.add_trace(go.Scatter3d(x=tok_f, y=rpm_f, z=tw_f, mode='markers', name='고장 데이터',
        marker=dict(size=3, color='#ff2200', opacity=0.32, line=dict(width=0))))
    pt_col = '#00ff88' if is_ok else '#ff6600'
    pt_sym = 'diamond'  if is_ok else 'cross'
    fig.add_trace(go.Scatter3d(
        x=[torque], y=[rot_speed], z=[tool_wear],
        mode='markers+text', name='현재 입력값',
        marker=dict(size=14, color=pt_col, symbol=pt_sym, opacity=1.0, line=dict(width=2, color='#fff')),
        text=['◀ 현재 입력값'],
        textfont=dict(family='Noto Sans KR', size=11, color=pt_col),
        textposition='middle right'
    ))
    fig.update_layout(
        scene=dict(
            bgcolor='#020508',
            xaxis=dict(title=dict(text='토크 (Nm)',        font=dict(color='#00b4ff',size=11,family='Noto Sans KR')), backgroundcolor='#020508', gridcolor='#0a1520', showbackground=True, tickfont=dict(color='#1e4060',size=9)),
            yaxis=dict(title=dict(text='회전속도 (rpm)',    font=dict(color='#00b4ff',size=11,family='Noto Sans KR')), backgroundcolor='#020508', gridcolor='#0a1520', showbackground=True, tickfont=dict(color='#1e4060',size=9)),
            zaxis=dict(title=dict(text='공구 마모도 (min)', font=dict(color='#00b4ff',size=11,family='Noto Sans KR')), backgroundcolor='#020508', gridcolor='#0a1520', showbackground=True, tickfont=dict(color='#1e4060',size=9)),
        ),
        paper_bgcolor='#020508', font=dict(color='#1e4060'),
        legend=dict(font=dict(family='Noto Sans KR',size=11,color='#2a4a60'), bgcolor='#030810', bordercolor='#0a1520', borderwidth=1, x=0, y=1),
        margin=dict(l=0,r=0,t=10,b=0), height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 입력값 요약
    st.markdown('<div class="sec">데이터 로그 <span class="sec-ko">— 입력 센서값 요약</span></div>', unsafe_allow_html=True)
    summary = pd.DataFrame({
        "센서 항목": ["제품 유형","공기 온도","공정 온도","회전 속도","토크","공구 마모도"],
        "입력값":    [["H","L","M"][product_type], air_temp, process_temp, rot_speed, torque, tool_wear],
        "단위":      ["-","K","K","rpm","Nm","min"]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ── 푸터
st.markdown("---")
st.markdown("""
<div class="footer">
    산업 설비 고장 예측 진단 시스템 · PREDICTIVE MAINTENANCE AI<br>
    AI4I 2020 DATASET · KAGGLE · RANDOM FOREST + SMOTE · 11413 김태윤
</div>
""", unsafe_allow_html=True)