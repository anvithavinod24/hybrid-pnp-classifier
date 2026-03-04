"""
Streamlit App: Hybrid PnP V2 Image Classifier
Upload an image → get classification via Softmax baseline + Hybrid PnP V2 attention
"""

import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import json
import urllib.request
import time
import tracemalloc

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid PnP V2 Classifier",
    page_icon="🧠",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e6ff;
}

.stApp {
    background: #0a0a0f;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
}

.mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #12121a;
    border: 2px dashed #3d3a6b;
    border-radius: 12px;
    padding: 1.5rem;
    transition: border-color 0.3s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #7c6fff;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #16162a 0%, #1a1a30 100%);
    border: 1px solid #2d2b55;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #7a78a8;
    margin-bottom: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e8e6ff;
}

.metric-sub {
    font-size: 0.78rem;
    color: #5c5a88;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.2rem;
}

/* Winner badge */
.badge-win {
    display: inline-block;
    background: #2d6a4f;
    color: #52b788;
    border: 1px solid #52b788;
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

.badge-lose {
    display: inline-block;
    background: #3a1a1a;
    color: #e07070;
    border: 1px solid #a04040;
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

.badge-tie {
    display: inline-block;
    background: #2a2a1a;
    color: #c8b060;
    border: 1px solid #806040;
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

/* Head grid */
.head-grid {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 3px;
    margin-top: 0.5rem;
}

.head-cell {
    aspect-ratio: 1;
    border-radius: 3px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.55rem;
    font-family: 'JetBrains Mono', monospace;
}

.pnp-head { background: #3d2d7a; color: #b8a8ff; }
.softmax-head { background: #1a2a3d; color: #6090c0; }
.safe-head { background: #2a3d2a; color: #60c060; }

/* Section header */
.section-title {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #4a4880;
    margin-top: 2rem;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #1e1e38;
    padding-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Rank heatmap */
.rank-row {
    display: flex;
    gap: 3px;
    margin-bottom: 3px;
    align-items: center;
}
.rank-label {
    font-size: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    color: #4a4880;
    width: 20px;
    text-align: right;
    margin-right: 4px;
}
.rank-cell {
    width: 28px;
    height: 22px;
    border-radius: 3px;
    font-size: 0.55rem;
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #e8e6ff;
}

/* progress bar override */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #5040c0 0%, #9060ff 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d0d1a;
    border-right: 1px solid #1e1e38;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ──────────────────────────────────────────────────────────────────
THRESHOLD    = 64
M_MIN        = 16
M_MAX        = 64
LATE_LAYERS  = {9, 10, 11}
SAFE_SOFTMAX = 4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load ImageNet labels ───────────────────────────────────────────────────────
@st.cache_data
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception:
        return [f"class_{i}" for i in range(1000)]

# ─── Core logic (cached model creation) ────────────────────────────────────────
@st.cache_resource
def get_transform():
    base = timm.create_model('vit_base_patch16_224', pretrained=True)
    base.eval()
    config = resolve_data_config({}, model=base)
    return create_transform(**config)

# ─── Attention helpers (no external attn.py needed) ────────────────────────────
def landmark_pool(x, num_landmarks):
    B, H, N, D = x.shape
    if N <= num_landmarks:
        return x
    step  = N / num_landmarks
    idxs  = torch.tensor([int(i * step) for i in range(num_landmarks)],
                         device=x.device, dtype=torch.long)
    return x[:, :, idxs, :]

def moore_penrose_iter_pinv(x, iters=6):
    device = x.device
    abs_x  = torch.abs(x)
    col    = abs_x.sum(dim=-1, keepdim=True)
    row    = abs_x.sum(dim=-2, keepdim=True)
    z      = x.transpose(-1, -2) / (torch.max(col) * torch.max(row))
    I      = torch.eye(x.shape[-1], device=device).unsqueeze(0).unsqueeze(0)
    for _ in range(iters):
        xz = x @ z
        z  = 0.25 * z @ (13 * I - xz @ (15 * I - xz @ (7 * I - xz)))
    return z

def get_pnp_decision(layer_idx, rank_matrix_row):
    decisions = []
    if layer_idx in LATE_LAYERS:
        head_ranks         = [(h, rank_matrix_row[h]) for h in range(len(rank_matrix_row))]
        sorted_by_rank     = sorted(head_ranks, key=lambda x: x[1], reverse=True)
        forced_softmax     = {h for h, _ in sorted_by_rank[:SAFE_SOFTMAX]}
        for h, rank in head_ranks:
            if h in forced_softmax:
                decisions.append((False, 0))
            elif rank < THRESHOLD:
                decisions.append((True, int(min(max(rank, M_MIN), M_MAX))))
            else:
                decisions.append((False, 0))
    else:
        for rank in rank_matrix_row:
            decisions.append((True, int(min(max(rank, M_MIN), M_MAX)))
                             if rank < THRESHOLD else (False, 0))
    return decisions

# ─── Attention module ──────────────────────────────────────────────────────────
class HybridHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, decisions, pinv_iters=6):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.pinv_iters = pinv_iters
        self.decisions  = decisions
        self.qkv        = nn.Linear(dim, dim * 3, bias=True)
        self.proj       = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        outs = []
        for h in range(self.num_heads):
            q_h, k_h, v_h = q[:,h:h+1], k[:,h:h+1], v[:,h:h+1]
            use_pnp, M = self.decisions[h]
            if use_pnp:
                q_m, k_m = landmark_pool(q_h, M), landmark_pool(k_h, M)
                SA = q_h @ k_m.transpose(-2,-1)
                SC = q_m @ k_h.transpose(-2,-1)
                SB = q_m @ k_m.transpose(-2,-1)
                eA = torch.exp(SA - SA.amax(dim=-1, keepdim=True))
                eC = torch.exp(SC - SC.amax(dim=-1, keepdim=True))
                eB = torch.exp((SB - SC.amax(dim=-1, keepdim=True)).clamp(min=-88.0))
                v_aug = torch.cat([v_h, torch.ones_like(v_h[...,:1])], dim=-1)
                pi    = moore_penrose_iter_pinv(eB, self.pinv_iters)
                prod  = eA @ pi @ (eC @ v_aug)
                out_h = prod[...,:-1] / prod[...,-1:].clamp(min=1e-8)
            else:
                out_h = (q_h @ k_h.transpose(-2,-1)).softmax(dim=-1) @ v_h
            outs.append(out_h)
        ctx = torch.cat(outs, dim=1).permute(0,2,1,3).contiguous().view(B, N, C)
        return self.proj(ctx)

# ─── OG attention with hook for rank computation ───────────────────────────────
class OGAttentionWithHook(nn.Module):
    def __init__(self, old_attn):
        super().__init__()
        self.num_heads  = old_attn.num_heads
        self.head_dim   = old_attn.head_dim
        self.scale      = old_attn.scale
        self.qkv        = old_attn.qkv
        self.proj       = old_attn.proj
        self.attn_drop  = old_attn.attn_drop
        self.attention_probs = None

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2,-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attention_probs = attn.detach()
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        return self.proj(x)

def build_softmax_model():
    m = timm.create_model('vit_base_patch16_224', pretrained=True)
    m.eval().to(DEVICE)
    for block in m.blocks:
        block.attn = OGAttentionWithHook(block.attn)
    return m

def compute_rank_matrix(softmax_model, tensor):
    with torch.no_grad():
        softmax_model(tensor)
    rm = np.zeros((12,12), dtype=int)
    for i, block in enumerate(softmax_model.blocks):
        ap = block.attn.attention_probs
        for h in range(ap.shape[1]):
            S     = torch.linalg.svdvals(ap[0,h].float())
            S     = S / S.sum().clamp(min=1e-12)
            cumsum = torch.cumsum(S, dim=0)
            rm[i][h] = (cumsum < 0.95).sum().item() + 1
    return rm

def build_hybrid_model(rank_matrix):
    m = timm.create_model('vit_base_patch16_224', pretrained=True)
    m.eval().to(DEVICE)
    for i, block in enumerate(m.blocks):
        old   = block.attn
        dim   = old.qkv.weight.shape[1]
        nheads = old.num_heads
        decs  = get_pnp_decision(i, [int(rank_matrix[i][h]) for h in range(nheads)])
        new_a = HybridHeadAttention(dim=dim, num_heads=nheads, decisions=decs).to(DEVICE)
        new_a.qkv.weight.data  = old.qkv.weight.data.clone()
        new_a.qkv.bias.data    = old.qkv.bias.data.clone()
        new_a.proj.weight.data = old.proj.weight.data.clone()
        new_a.proj.bias.data   = old.proj.bias.data.clone()
        block.attn = new_a
    return m

def rank_to_color(r):
    # low rank (sparse) → purple, high rank (dense) → blue-grey
    t = min(r / 128.0, 1.0)
    r_c = int(30  + t * 20)
    g_c = int(20  + t * 35)
    b_c = int(80  + t * 60)
    return f"rgb({r_c},{g_c},{b_c})"

# ─── UI Layout ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 2rem;">
  <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
              color:#4a4880; letter-spacing:0.2em; margin-bottom:0.4rem;">
    ATTENTION MECHANISM RESEARCH
  </div>
  <h1 style="font-size:2.4rem; margin:0; color:#e8e6ff;">
    Hybrid PnP V2 Classifier
  </h1>
  <p style="color:#6060a0; margin-top:0.5rem; font-size:0.95rem;">
    Compares standard softmax attention against Hybrid Performer + Softmax (PnP V2)
    with clamped M and late-layer safeguards.
  </p>
</div>
""", unsafe_allow_html=True)

# Sidebar: settings display
with st.sidebar:
    st.markdown("### ⚙️ V2 Parameters")
    st.markdown(f"""
    <div class="mono" style="line-height:2;">
    THRESHOLD &nbsp;= {THRESHOLD}<br>
    M_MIN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {M_MIN}<br>
    M_MAX &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {M_MAX}<br>
    LATE_LAYERS &nbsp;= {{9,10,11}}<br>
    SAFE_SOFTMAX = {SAFE_SOFTMAX}<br>
    DEVICE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {DEVICE}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="mono" style="font-size:0.7rem; color:#4a4880; line-height:1.8;">
    🟣 PnP head (Performer)<br>
    🔵 Softmax head (standard)<br>
    🟢 Safe softmax (late layer)
    </div>
    """, unsafe_allow_html=True)

# ─── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop an image to classify",
    type=["jpg","jpeg","png","bmp","tiff","webp"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color:#3a3870;">
      <div style="font-size:3rem; margin-bottom:1rem;">🖼️</div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                  letter-spacing:0.1em;">Upload an image to begin classification</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Classification ────────────────────────────────────────────────────────────
img = Image.open(uploaded).convert("RGB")
transform  = get_transform()
labels     = load_imagenet_labels()
inp        = transform(img).unsqueeze(0).to(DEVICE)

col_img, col_results = st.columns([1, 2], gap="large")

with col_img:
    st.image(img, use_container_width=True,
             caption=f"{uploaded.name} · {img.size[0]}×{img.size[1]}px")

with col_results:
    progress_bar = st.progress(0, text="Building softmax model…")

    # ── Softmax pass ──
    tracemalloc.start()
    t_sm_start = time.time()

    sm_model = build_softmax_model()
    progress_bar.progress(20, text="Computing rank matrix…")

    rank_matrix = compute_rank_matrix(sm_model, inp)
    progress_bar.progress(40, text="Running softmax inference…")

    t0 = time.time()
    with torch.no_grad():
        sm_out = sm_model(inp)
    sm_infer_time = time.time() - t0

    sm_total_time = time.time() - t_sm_start
    _, sm_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    sm_peak_mem_mb = sm_peak_mem / 1024 / 1024

    sm_probs = torch.softmax(sm_out, dim=1)[0]
    s_prob, s_idx = sm_probs.max(dim=0)
    s_prob, s_idx = float(s_prob), int(s_idx)
    sm_top5 = torch.topk(sm_probs, 5)

    del sm_model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    progress_bar.progress(55, text="Building hybrid V2 model…")

    # ── Hybrid pass ──
    tracemalloc.start()
    t_hy_start = time.time()

    hy_model = build_hybrid_model(rank_matrix)
    progress_bar.progress(80, text="Running hybrid inference…")

    t0 = time.time()
    with torch.no_grad():
        hy_out = hy_model(inp)
    hy_infer_time = time.time() - t0

    hy_total_time = time.time() - t_hy_start
    _, hy_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    hy_peak_mem_mb = hy_peak_mem / 1024 / 1024

    hy_probs = torch.softmax(hy_out, dim=1)[0]
    h_prob, h_idx = hy_probs.max(dim=0)
    h_prob, h_idx = float(h_prob), int(h_idx)
    hy_top5 = torch.topk(hy_probs, 5)

    del hy_model
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # PnP stats
    pnp_heads   = sum(
        1 for li in range(12)
        for use_pnp, _ in get_pnp_decision(li, [int(rank_matrix[li][h]) for h in range(12)])
        if use_pnp
    )
    pnp_pct     = (pnp_heads / 144.0) * 100.0
    delta       = abs(s_prob - h_prob) * 100
    agree       = s_idx == h_idx
    progress_bar.progress(100, text="Done ✓")
    progress_bar.empty()

    # ── Result cards ──
    sm_label = labels[s_idx] if s_idx < len(labels) else f"class_{s_idx}"
    hy_label = labels[h_idx] if h_idx < len(labels) else f"class_{h_idx}"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">🔵 Softmax Baseline</div>
          <div class="metric-value">{sm_label}</div>
          <div class="metric-sub">conf: {s_prob*100:.2f}% · idx {s_idx}</div>
          <div style="margin-top:0.6rem; display:flex; gap:0.8rem; flex-wrap:wrap;">
            <div class="metric-sub">⏱ infer: <span style="color:#6090c0">{sm_infer_time*1000:.0f}ms</span></div>
            <div class="metric-sub">⏱ total: <span style="color:#4070a0">{sm_total_time*1000:.0f}ms</span></div>
            <div class="metric-sub">🧠 mem: <span style="color:#4070a0">{sm_peak_mem_mb:.1f}MB</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">🟣 Hybrid PnP V2</div>
          <div class="metric-value">{hy_label}</div>
          <div class="metric-sub">conf: {h_prob*100:.2f}% · idx {h_idx}</div>
          <div style="margin-top:0.6rem; display:flex; gap:0.8rem; flex-wrap:wrap;">
            <div class="metric-sub">⏱ infer: <span style="color:#b8a8ff">{hy_infer_time*1000:.0f}ms</span></div>
            <div class="metric-sub">⏱ total: <span style="color:#9060ff">{hy_total_time*1000:.0f}ms</span></div>
            <div class="metric-sub">🧠 mem: <span style="color:#9060ff">{hy_peak_mem_mb:.1f}MB</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Agreement + stats
    if agree:
        badge = '<span class="badge-win">✓ AGREE</span>'
    else:
        badge = '<span class="badge-lose">✗ DISAGREE</span>'

    st.markdown(f"""
    <div style="display:flex; gap:1.5rem; align-items:center; flex-wrap:wrap;
                background:#12121a; border:1px solid #1e1e38;
                border-radius:10px; padding:1rem 1.4rem; margin-bottom:1rem;">
      <div>{badge}</div>
      <div class="mono" style="color:#7a78a8;">
        Δ conf: <span style="color:#e8e6ff">{delta:.2f}%</span>
      </div>
      <div class="mono" style="color:#7a78a8;">
        PnP heads: <span style="color:#b8a8ff">{pnp_heads}/144</span> ({pnp_pct:.1f}%)
      </div>
      <div class="mono" style="color:#7a78a8;">
        🔵 infer: <span style="color:#6090c0">{sm_infer_time*1000:.0f}ms</span>
        · total: <span style="color:#4070a0">{sm_total_time*1000:.0f}ms</span>
        · mem: <span style="color:#4070a0">{sm_peak_mem_mb:.1f}MB</span>
      </div>
      <div class="mono" style="color:#7a78a8;">
        🟣 infer: <span style="color:#b8a8ff">{hy_infer_time*1000:.0f}ms</span>
        · total: <span style="color:#9060ff">{hy_total_time*1000:.0f}ms</span>
        · mem: <span style="color:#9060ff">{hy_peak_mem_mb:.1f}MB</span>
      </div>
      <div class="mono" style="color:#7a78a8;">
        infer speedup: <span style="color:#e8e6ff">{'%.2fx' % (sm_infer_time/hy_infer_time) if hy_infer_time > 0 else 'N/A'}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top-5 ──
    st.markdown('<div class="section-title">Top-5 Predictions</div>', unsafe_allow_html=True)
    t5c1, t5c2 = st.columns(2)
    with t5c1:
        st.markdown("**Softmax**", help="Standard attention baseline")
        for prob, idx in zip(sm_top5.values.tolist(), sm_top5.indices.tolist()):
            lbl = labels[idx] if idx < len(labels) else f"class_{idx}"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:0.3rem 0; border-bottom:1px solid #1a1a2e;">
              <span class="mono" style="color:#a0a0d0; font-size:0.8rem;">{lbl}</span>
              <span class="mono" style="color:#6080c0; font-size:0.78rem;">{prob*100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
    with t5c2:
        st.markdown("**Hybrid V2**", help="Performer + Softmax hybrid")
        for prob, idx in zip(hy_top5.values.tolist(), hy_top5.indices.tolist()):
            lbl = labels[idx] if idx < len(labels) else f"class_{idx}"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:0.3rem 0; border-bottom:1px solid #1a1a2e;">
              <span class="mono" style="color:#a080e0; font-size:0.8rem;">{lbl}</span>
              <span class="mono" style="color:#9060ff; font-size:0.78rem;">{prob*100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

# ─── Rank Matrix Heatmap ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">Rank Matrix · 12 Layers × 12 Heads (effective rank at 95% singular value energy)</div>',
            unsafe_allow_html=True)

heatmap_html = '<div style="display:flex; flex-direction:column; gap:2px;">'
heatmap_html += '<div style="display:flex; gap:3px; margin-left:28px; margin-bottom:2px;">'
for h in range(12):
    heatmap_html += f'<div class="rank-label" style="width:28px;text-align:center;">H{h}</div>'
heatmap_html += '</div>'

for li in range(12):
    decs = get_pnp_decision(li, [int(rank_matrix[li][h]) for h in range(12)])
    heatmap_html += f'<div class="rank-row"><div class="rank-label">L{li}</div>'
    for h in range(12):
        r = int(rank_matrix[li][h])
        use_pnp, M = decs[h]
        bg = rank_to_color(r)
        if li in LATE_LAYERS and not use_pnp and r < THRESHOLD:
            border = "2px solid #52b788"  # safe softmax (forced)
        elif use_pnp:
            border = "2px solid #7c6fff"
        else:
            border = "1px solid #2a2850"
        heatmap_html += f'<div class="rank-cell" style="background:{bg}; border:{border};">{r}</div>'
    heatmap_html += '</div>'
heatmap_html += '</div>'

st.markdown(heatmap_html, unsafe_allow_html=True)

st.markdown("""
<div class="mono" style="font-size:0.65rem; color:#3a3870; margin-top:0.5rem;">
  🟣 purple border = PnP head &nbsp;|&nbsp; 🟢 green border = forced softmax (late-layer safeguard) &nbsp;|&nbsp;
  darker = lower rank (more compressible)
</div>
""", unsafe_allow_html=True)

# ─── Head Decision Grid ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Head Decision Map</div>', unsafe_allow_html=True)

grid_html = '<div style="font-family:JetBrains Mono,monospace; font-size:0.6rem;">'
grid_html += '<div style="display:grid; grid-template-columns:30px repeat(12,32px); gap:2px; margin-bottom:4px;">'
grid_html += '<div></div>'
for h in range(12): grid_html += f'<div style="text-align:center;color:#4a4880;">H{h}</div>'
grid_html += '</div>'

for li in range(12):
    decs = get_pnp_decision(li, [int(rank_matrix[li][h]) for h in range(12)])
    grid_html += f'<div style="display:grid; grid-template-columns:30px repeat(12,32px); gap:2px; margin-bottom:2px;">'
    grid_html += f'<div style="color:#4a4880; line-height:22px;">L{li}</div>'
    for h, (use_pnp, M) in enumerate(decs):
        r = int(rank_matrix[li][h])
        if use_pnp:
            cls = "pnp-head"
            label = f"P{M}"
        elif li in LATE_LAYERS and r < THRESHOLD:
            cls = "safe-head"
            label = "SFX"
        else:
            cls = "softmax-head"
            label = "SFX"
        grid_html += f'<div class="head-cell {cls}" style="height:22px;">{label}</div>'
    grid_html += '</div>'
grid_html += '</div>'

st.markdown(grid_html, unsafe_allow_html=True)
st.markdown("""
<div class="mono" style="font-size:0.65rem; color:#3a3870; margin-top:0.5rem;">
  P{M} = Performer with M landmarks &nbsp;|&nbsp; SFX = Softmax attention
</div>
""", unsafe_allow_html=True)
