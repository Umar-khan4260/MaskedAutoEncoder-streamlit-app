"""
app.py — Streamlit App for Masked Autoencoder (MAE) Demo
Upload an image, choose a masking ratio, see the reconstruction in real time.
"""

import io
import os
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity  as ssim_metric

from model import MaskedAutoencoder, load_model

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "MAE — Masked Autoencoder Demo",
    page_icon   = "🎭",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT    = "best_model.pt"   # place your trained checkpoint here


# ── Model Loading (cached so it only loads once) ──────────────────────────────
@st.cache_resource(show_spinner="Loading MAE model weights...")
def get_model():
    """
    Load the trained MAE model.
    Cached by Streamlit so the 107M-param model loads only once per session.
    """
    if not os.path.exists(CHECKPOINT):
        return None   # Demo mode — no checkpoint available

    try:
        model = load_model(CHECKPOINT, device=DEVICE)
        return model
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        return None


# ── Image Processing Helpers ──────────────────────────────────────────────────
def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """Resize, crop, normalize PIL image → (1, 3, 224, 224) tensor."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(pil_img.convert("RGB")).unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization → values in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(tensor.device)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """(1, 3, H, W) float tensor → PIL Image."""
    arr = denormalize(tensor).squeeze(0).cpu().numpy()
    arr = (arr.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def compute_metrics(orig_tensor, recon_tensor):
    """Compute PSNR and SSIM between original and reconstruction."""
    orig  = denormalize(orig_tensor).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    recon = denormalize(recon_tensor).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    orig  = np.clip(orig,  0, 1)
    recon = np.clip(recon, 0, 1)
    psnr  = psnr_metric(orig, recon, data_range=1.0)
    ssim  = ssim_metric(orig, recon, data_range=1.0, channel_axis=2)
    return psnr, ssim


def run_inference(model, img_tensor, mask_ratio):
    """Run the MAE model and return (masked_pil, recon_pil, mask_tensor)."""
    img_tensor = img_tensor.to(DEVICE)
    with torch.no_grad():
        masked_t, recon_t, mask = model.reconstruct(img_tensor, mask_ratio=mask_ratio)
    return (
        tensor_to_pil(masked_t),
        tensor_to_pil(recon_t),
        tensor_to_pil(img_tensor),
        masked_t,
        recon_t,
        img_tensor,
        mask,
    )


# ── Demo Mode (no checkpoint) ─────────────────────────────────────────────────
def show_demo_mode():
    st.warning(
        "⚠️ **Demo Mode** — No trained checkpoint found (`best_model.pt`).\n\n"
        "To see real reconstructions:\n"
        "1. Train the model on Kaggle using the provided notebook\n"
        "2. Download `best_model.pt` from Kaggle outputs\n"
        "3. Place it in the same folder as `app.py`\n"
        "4. Re-run the app"
    )
    st.info(
        "**What this app does when a model is loaded:**\n"
        "- Splits your uploaded image into 196 patches (16×16 each)\n"
        "- Randomly hides the % you choose\n"
        "- The MAE model reconstructs the hidden patches from context\n"
        "- Shows side-by-side: Original | Masked | Reconstruction"
    )

    # Architecture diagram
    st.subheader("🏗️ Model Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Encoder — ViT-Base/16**
        | Parameter | Value |
        |---|---|
        | Hidden dim | 768 |
        | Transformer layers | 12 |
        | Attention heads | 12 |
        | Parameters | ~86M |
        | Input | 25% visible patches only |
        """)
    with col2:
        st.markdown("""
        **Decoder — ViT-Small/16**
        | Parameter | Value |
        |---|---|
        | Hidden dim | 384 |
        | Transformer layers | 12 |
        | Attention heads | 6 |
        | Parameters | ~22M |
        | Input | Encoder output + mask tokens |
        """)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def build_sidebar():
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")

    mask_pct = st.sidebar.slider(
        "🎭 Masking Ratio (%)",
        min_value = 10,
        max_value = 95,
        value     = 75,
        step      = 5,
        help      = "75% = MAE default. Higher = harder reconstruction task."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 What the metrics mean")
    st.sidebar.markdown("""
    **PSNR** (Peak Signal-to-Noise Ratio)
    - Measures pixel-level accuracy
    - Higher is better
    - > 25 dB = good quality

    **SSIM** (Structural Similarity)
    - Measures perceptual quality
    - Range: 0 → 1 (higher = better)
    - > 0.70 = visually similar
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 How MAE Works")
    st.sidebar.markdown("""
    1. Image split into **196 patches** (16×16)
    2. **75% randomly hidden** (147 patches)
    3. **Encoder** processes only 49 visible patches
    4. **Decoder** reconstructs all 196 patches
    5. Loss computed only on masked patches
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built with pure PyTorch | "
        "[GitHub](https://github.com) | "
        "Trained on TinyImageNet"
    )

    return mask_pct / 100.0


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🎭 Masked Autoencoder (MAE) — Live Demo")
    st.markdown(
        "Upload any image. The model hides a portion of it and reconstructs "
        "the missing parts — **with zero labels, no supervision**."
    )
    st.markdown("---")

    mask_ratio = build_sidebar()
    model      = get_model()

    if model is None:
        show_demo_mode()
        return

    # ── Upload ─────────────────────────────────────────────────────────────────
    col_upload, col_info = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "📤 Upload an image",
            type   = ["jpg", "jpeg", "png", "webp", "bmp"],
            help   = "Any image works — photos, drawings, screenshots"
        )

    with col_info:
        st.markdown("### Current Settings")
        st.metric("Masking Ratio",    f"{int(mask_ratio * 100)}%")
        st.metric("Visible Patches",  f"{int(196 * (1 - mask_ratio))} / 196")
        st.metric("Masked Patches",   f"{int(196 * mask_ratio)} / 196")
        st.metric("Device",           DEVICE.upper())

    if uploaded is None:
        # Show example instructions
        st.info(
            "👆 Upload an image above to get started.\n\n"
            "Adjust the **Masking Ratio** in the sidebar to control "
            "how much of the image gets hidden."
        )

        # Show example patches visualization
        st.markdown("### 🧩 How Patching Works")
        st.markdown(
            "A 224×224 image is divided into a **14×14 grid** of 16×16 patches "
            "(196 patches total). The selected percentage are randomly hidden "
            "(shown in gray), and the model reconstructs them."
        )
        return

    # ── Inference ──────────────────────────────────────────────────────────────
    pil_img = Image.open(uploaded).convert("RGB")
    img_tensor = preprocess(pil_img)

    with st.spinner("🔮 Reconstructing..."):
        try:
            masked_pil, recon_pil, orig_pil, masked_t, recon_t, orig_t, mask = \
                run_inference(model, img_tensor, mask_ratio)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            return

    # ── Results ────────────────────────────────────────────────────────────────
    st.markdown("### 🖼️ Results")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🖼️ Original**")
        st.image(orig_pil, use_column_width=True)

    with c2:
        st.markdown(f"**🎭 Masked Input ({int(mask_ratio*100)}% hidden)**")
        st.image(masked_pil, use_column_width=True)

    with c3:
        st.markdown("**🔮 Reconstruction**")
        st.image(recon_pil, use_column_width=True)

    # ── Metrics ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Quality Metrics")

    psnr, ssim = compute_metrics(orig_t, recon_t)
    masked_count = int(mask.sum().item())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        psnr_delta = "🟢 Good" if psnr > 25 else "🟡 Fair" if psnr > 20 else "🔴 Low"
        st.metric("PSNR", f"{psnr:.2f} dB", psnr_delta)
    with m2:
        ssim_delta = "🟢 Good" if ssim > 0.7 else "🟡 Fair" if ssim > 0.5 else "🔴 Low"
        st.metric("SSIM", f"{ssim:.4f}", ssim_delta)
    with m3:
        st.metric("Masked Patches", f"{masked_count} / 196")
    with m4:
        st.metric("Visible Patches", f"{196 - masked_count} / 196")

    # ── Download ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Download Results")

    d1, d2, d3 = st.columns(3)

    def pil_to_bytes(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    with d1:
        st.download_button(
            "⬇️ Download Original",
            data      = pil_to_bytes(orig_pil),
            file_name = "original.png",
            mime      = "image/png"
        )
    with d2:
        st.download_button(
            "⬇️ Download Masked",
            data      = pil_to_bytes(masked_pil),
            file_name = "masked.png",
            mime      = "image/png"
        )
    with d3:
        st.download_button(
            "⬇️ Download Reconstruction",
            data      = pil_to_bytes(recon_pil),
            file_name = "reconstruction.png",
            mime      = "image/png"
        )

    # ── Technical Details ──────────────────────────────────────────────────────
    with st.expander("🔬 Technical Details"):
        st.markdown(f"""
        **Inference Details**
        - Image resized to 256×256, then center-cropped to 224×224
        - Divided into 14×14 grid = **196 patches** of 16×16 pixels each
        - **{masked_count} patches masked** ({masked_count/196*100:.0f}%)
        - **{196-masked_count} patches visible** ({(196-masked_count)/196*100:.0f}%)
        - Running on: **{DEVICE.upper()}**

        **Model Architecture**
        - Encoder: ViT-Base/16 (~86M params) — processes only visible patches
        - Decoder: ViT-Small/16 (~22M params) — reconstructs all 196 patches
        - Total: ~108M parameters, built from pure PyTorch

        **Loss (training)**
        - MSE computed ONLY on masked patches
        - Per-patch normalization before loss (mean/std per patch)
        - Optimizer: AdamW (β₂=0.95, wd=0.05)
        - Schedule: 10-epoch linear warmup + cosine decay
        """)


if __name__ == "__main__":
    main()
