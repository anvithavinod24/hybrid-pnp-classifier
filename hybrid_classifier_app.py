"""
Streamlit App: Hybrid PnP V2 Image Classifier — Light Theme
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
import copy

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid PnP V2 Classifier",
    page_icon="🧠",
    layout="wide",
)

# ─── Custom CSS (Light Theme) ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #f5f4ff;
    color: #1a1840;
}

.stApp {
    background: #f5f4ff;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
    color: #1a1840;
}

.mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 2px dashed #b0a8e0;
    border-radius: 12px;
    padding: 1.5rem;
    transition: border-color 0.3s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #6c5ce7;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #ddd8f8;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(100,80,200,0.07);
}

.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8880b8;
    margin-bottom: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #1a1840;
}

.metric-sub {
    font-size: 0.78rem;
    color: #9890c0;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.2rem;
}

.badge-win {
    display: inline-block;
    background: #d4f5e5;
    color: #1a7a4a;
    border: 1px solid #52b788;
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

.badge-lose {
    display: inline-block;
    background: #fde8e8;
    color: #a02020;
    border: 1px solid #e07070;
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
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

.pnp-head     { background: #e0d8ff; color: #4020a0; }
.softmax-head { background: #dceeff; color: #1a4080; }
.safe-head    { background: #d4f5e5; color: #1a6040; }

.section-title {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #9890c0;
    margin-top: 2rem;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #e0dcf8;
    padding-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}

.rank-row {
    display: flex;
    gap: 3px;
    margin-bottom: 3px;
    align-items: center;
}
.rank-label {
    font-size: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    color: #9890c0;
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
    color: #1a1840;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #6c5ce7 0%, #a29bfe 100%);
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e0dcf8;
}

.stats-bar {
    display: flex;
    gap: 1.5rem;
    align-items: center;
    flex-wrap: wrap;
    background: #ffffff;
    border: 1px solid #ddd8f8;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(100,80,200,0.06);
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

@st.cache_resource
def get_transform():
    base = timm.create_model('vit_base_patch16_224', pretrained=True)
    base.eval()
    config = resolve_data_config({}, model=base)
    return create_transform(**config)

# ─── Attention helpers ──────────────────────────────────────────────────────────
def landmark_pool(x, num_landmarks):
    B, H, N, D = x.shape
    if N <= num_landmarks:
        return x
    idxs = torch.tensor([int(i * N / num_landmarks) for i in range(num_landmarks)],
                        device=x.device, dtype=torch.long)
    return x[:, :, idxs, :]

def moore_penrose_iter_pinv(x, iters=6):
    abs_x = torch.abs(x)
    col   = abs_x.sum(dim=-1, keepdim=True)
    row   = abs_x.sum(dim=-2, keepdim=True)
    z     = x.transpose(-1, -2) / (torch.max(col) * torch.max(row))
    I     = torch.eye(x.shape[-1], device=x.device).unsqueeze(0).unsqueeze(0)
    for _ in range(iters):
        xz = x @ z
        z  = 0.25 * z @ (13 * I - xz @ (15 * I - xz @ (7 * I - xz)))
    return z

def get_pnp_decision(layer_idx, rank_matrix_row):
    decisions = []
    if layer_idx in LATE_LAYERS:
        head_ranks     = [(h, rank_matrix_row[h]) for h in range(len(rank_matrix_row))]
        sorted_by_rank = sorted(head_ranks, key=lambda x: x[1], reverse=True)
        forced_softmax = {h for h, _ in sorted_by_rank[:SAFE_SOFTMAX]}
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

def rank_to_color(r):
    t   = min(r / 128.0, 1.0)
    r_c = int(210 - t * 40)
    g_c = int(200 - t * 20)
    b_c = int(245 - t * 20)
    return f"rgb({r_c},{g_c},{b_c})"

# ─── Model classes ─────────────────────────────────────────────────────────────
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

class OGAttentionWithHook(nn.Module):
    def __init__(self, old_attn):
        super().__init__()
        self.num_heads       = old_attn.num_heads
        self.head_dim        = old_attn.head_dim
        self.scale           = old_attn.scale
        self.qkv             = copy.deepcopy(old_attn.qkv)
        self.proj            = copy.deepcopy(old_attn.proj)
        self.attention_probs = None

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2,-1)
        attn = attn.softmax(dim=-1)
        self.attention_probs = attn.detach()
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        return self.proj(x)

def build_softmax_model():
    m = timm.create_model('vit_base_patch16_224', pretrained=True)
    m.eval().to(DEVICE).float()
    for block in m.blocks:
        block.attn = OGAttentionWithHook(block.attn)
    return m

def compute_rank_matrix(softmax_model, tensor):
    with torch.no_grad():
        softmax_model(tensor)
    rm = np.zeros((12, 12), dtype=int)
    for i, block in enumerate(softmax_model.blocks):
        ap = block.attn.attention_probs
        for h in range(ap.shape[1]):
            S      = torch.linalg.svdvals(ap[0, h].float())
            S      = S / S.sum().clamp(min=1e-12)
            cumsum = torch.cumsum(S, dim=0)
            rm[i][h] = (cumsum < 0.95).sum().item() + 1
    return rm

def build_hybrid_model(rank_matrix):
    m = timm.create_model('vit_base_patch16_224', pretrained=True)
    m.eval().to(DEVICE).float()
    for i, block in enumerate(m.blocks):
        old    = block.attn
        dim    = old.qkv.weight.shape[1]
        nheads = old.num_heads
        decs   = get_pnp_decision(i, [int(rank_matrix[i][h]) for h in range(nheads)])
        new_a  = HybridHeadAttention(dim=dim, num_heads=nheads, decisions=decs).to(DEVICE)
        new_a.qkv.weight.data  = old.qkv.weight.data.clone()
        new_a.qkv.bias.data    = old.qkv.bias.data.clone()
        new_a.proj.weight.data = old.proj.weight.data.clone()
        new_a.proj.bias.data   = old.proj.bias.data.clone()
        block.attn = new_a
    return m

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 2rem;">
  <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
              color:#9890c0; letter-spacing:0.2em; margin-bottom:0.4rem;">
    ATTENTION MECHANISM RESEARCH
  </div>
  <h1 style="font-size:2.4rem; margin:0; color:#1a1840;">
    Hybrid PnP V2 Classifier
  </h1>
  <p style="color:#7870a8; margin-top:0.5rem; font-size:0.95rem;">
    Compares standard softmax attention against Hybrid Performer + Softmax (PnP V2)
    with clamped M and late-layer safeguards.
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ V2 Parameters")
    st.markdown(f"""
    <div class="mono" style="line-height:2; color:#1a1840;">
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
    <div class="mono" style="font-size:0.7rem; color:#9890c0; line-height:1.8;">
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
    <div style="text-align:center; padding: 4rem 2rem; color:#c0b8e8;">
      <div style="font-size:3rem; margin-bottom:1rem;">🖼️</div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                  letter-spacing:0.1em;">Upload an image to begin classification</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Classification ────────────────────────────────────────────────────────────
img       = Image.open(uploaded).convert("RGB")
transform = get_transform()
labels    = load_imagenet_labels()
inp       = transform(img).unsqueeze(0).to(DEVICE).float()

col_img, col_results = st.columns([1, 2], gap="large")

with col_img:
    st.image(img, use_container_width=True,
             caption=f"{uploaded.name} · {img.size[0]}×{img.size[1]}px")

with col_results:
    progress_bar = st.progress(0, text="Building softmax model…")

    sm_model    = build_softmax_model()
    progress_bar.progress(20, text="Computing rank matrix…")
    rank_matrix = compute_rank_matrix(sm_model, inp)
    progress_bar.progress(40, text="Running softmax inference…")

    with torch.no_grad():
        sm_out = sm_model(inp)
    sm_probs      = torch.softmax(sm_out, dim=1)[0]
    s_prob, s_idx = sm_probs.max(dim=0)
    s_prob, s_idx = float(s_prob), int(s_idx)
    sm_top5       = torch.topk(sm_probs, 5)

    del sm_model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    progress_bar.progress(55, text="Building hybrid V2 model…")

    hy_model = build_hybrid_model(rank_matrix)
    progress_bar.progress(80, text="Running hybrid inference…")

    with torch.no_grad():
        hy_out = hy_model(inp)
    hy_probs      = torch.softmax(hy_out, dim=1)[0]
    h_prob, h_idx = hy_probs.max(dim=0)
    h_prob, h_idx = float(h_prob), int(h_idx)
    hy_top5       = torch.topk(hy_probs, 5)

    del hy_model
    if DEVICE == "cuda": torch.cuda.empty_cache()

    pnp_heads = sum(
        1 for li in range(12)
        for use_pnp, _ in get_pnp_decision(li, [int(rank_matrix[li][h]) for h in range(12)])
        if use_pnp
    )
    pnp_pct = (pnp_heads / 144.0) * 100.0
    delta   = abs(s_prob - h_prob) * 100
    agree   = s_idx == h_idx

    progress_bar.progress(100, text="Done ✓")
    progress_bar.empty()

    sm_label = labels[s_idx] if s_idx < len(labels) else f"class_{s_idx}"
    hy_label = labels[h_idx] if h_idx < len(labels) else f"class_{h_idx}"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">🔵 Softmax Baseline</div>
          <div class="metric-value">{sm_label}</div>
          <div class="metric-sub">conf: {s_prob*100:.2f}% · idx {s_idx}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">🟣 Hybrid PnP V2</div>
          <div class="metric-value">{hy_label}</div>
          <div class="metric-sub">conf: {h_prob*100:.2f}% · idx {h_idx}</div>
        </div>
        """, unsafe_allow_html=True)

    badge = '<span class="badge-win">✓ AGREE</span>' if agree else '<span class="badge-lose">✗ DISAGREE</span>'
    st.markdown(f"""
    <div class="stats-bar">
      <div>{badge}</div>
      <div class="mono" style="color:#7870a8;">
        Δ confidence: <span style="color:#1a1840; font-weight:700;">{delta:.2f}%</span>
      </div>
      <div class="mono" style="color:#7870a8;">
        PnP heads: <span style="color:#4020a0; font-weight:700;">{pnp_heads}/144</span>
        &nbsp;({pnp_pct:.1f}%)
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Top-5 Predictions</div>', unsafe_allow_html=True)
    t5c1, t5c2 = st.columns(2)
    with t5c1:
        st.markdown("**Softmax**")
        for prob, idx in zip(sm_top5.values.tolist(), sm_top5.indices.tolist()):
            lbl = labels[idx] if idx < len(labels) else f"class_{idx}"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:0.3rem 0; border-bottom:1px solid #e8e4f8;">
              <span class="mono" style="color:#2a2060; font-size:0.8rem;">{lbl}</span>
              <span class="mono" style="color:#4060c0; font-size:0.78rem;">{prob*100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
    with t5c2:
        st.markdown("**Hybrid V2**")
        for prob, idx in zip(hy_top5.values.tolist(), hy_top5.indices.tolist()):
            lbl = labels[idx] if idx < len(labels) else f"class_{idx}"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:0.3rem 0; border-bottom:1px solid #e8e4f8;">
              <span class="mono" style="color:#2a2060; font-size:0.8rem;">{lbl}</span>
              <span class="mono" style="color:#6040c0; font-size:0.78rem;">{prob*100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

# ─── Rank Matrix Heatmap ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">Rank Matrix · 12 Layers × 12 Heads (effective rank at 95% singular value energy)</div>',
            unsafe_allow_html=True)

heatmap_html  = '<div style="display:flex; flex-direction:column; gap:2px;">'
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
            border = "2px solid #52b788"
        elif use_pnp:
            border = "2px solid #6c5ce7"
        else:
            border = "1px solid #d0ccee"
        heatmap_html += f'<div class="rank-cell" style="background:{bg}; border:{border};">{r}</div>'
    heatmap_html += '</div>'
heatmap_html += '</div>'

st.markdown(heatmap_html, unsafe_allow_html=True)
st.markdown("""
<div class="mono" style="font-size:0.65rem; color:#b0a8d8; margin-top:0.5rem;">
  🟣 purple border = PnP head &nbsp;|&nbsp; 🟢 green border = forced softmax (late-layer safeguard) &nbsp;|&nbsp;
  darker = lower rank (more compressible)
</div>
""", unsafe_allow_html=True)

# ─── Head Decision Grid ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Head Decision Map</div>', unsafe_allow_html=True)

grid_html  = '<div style="font-family:JetBrains Mono,monospace; font-size:0.6rem;">'
grid_html += '<div style="display:grid; grid-template-columns:30px repeat(12,32px); gap:2px; margin-bottom:4px;">'
grid_html += '<div></div>'
for h in range(12):
    grid_html += f'<div style="text-align:center;color:#9890c0;">H{h}</div>'
grid_html += '</div>'

for li in range(12):
    decs = get_pnp_decision(li, [int(rank_matrix[li][h]) for h in range(12)])
    grid_html += '<div style="display:grid; grid-template-columns:30px repeat(12,32px); gap:2px; margin-bottom:2px;">'
    grid_html += f'<div style="color:#9890c0; line-height:22px;">L{li}</div>'
    for h, (use_pnp, M) in enumerate(decs):
        r = int(rank_matrix[li][h])
        if use_pnp:
            cls, lbl = "pnp-head", f"P{M}"
        elif li in LATE_LAYERS and r < THRESHOLD:
            cls, lbl = "safe-head", "SFX"
        else:
            cls, lbl = "softmax-head", "SFX"
        grid_html += f'<div class="head-cell {cls}" style="height:22px;">{lbl}</div>'
    grid_html += '</div>'
grid_html += '</div>'

st.markdown(grid_html, unsafe_allow_html=True)
st.markdown("""
<div class="mono" style="font-size:0.65rem; color:#b0a8d8; margin-top:0.5rem;">
  P{M} = Performer with M landmarks &nbsp;|&nbsp; SFX = Softmax attention
</div>
""", unsafe_allow_html=True)
