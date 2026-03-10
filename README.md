# 🎭 Masked Autoencoder (MAE) — Streamlit Demo

Self-supervised image reconstruction using a Vision Transformer-based Masked Autoencoder.
Upload any image, choose a masking ratio, and watch the model reconstruct the hidden patches.

## 🏗️ Architecture

| Component | Model | Params | Role |
|---|---|---|---|
| Encoder | ViT-Base/16 | ~86M | Processes only 25% visible patches |
| Decoder | ViT-Small/16 | ~22M | Reconstructs all 196 patches |
| **Total** | | **~108M** | Built from pure PyTorch |

**Key idea:** Hide 75% of image patches → force the model to reconstruct them → the encoder learns meaningful visual representations with zero labels.

---

## 📁 Project Structure

```
mae-streamlit-app/
├── app.py              ← Streamlit app (main entry point)
├── model.py            ← Full MAE architecture (pure PyTorch)
├── best_model.pt       ← Trained weights (you add this — not in repo)
├── requirements.txt    ← Python dependencies
├── packages.txt        ← System packages for Streamlit Cloud
├── .gitignore
└── README.md
```

---

## 🚀 Deployment Guide

### Step 1 — Get your trained checkpoint from Kaggle

After training completes on Kaggle:
1. Go to your notebook → **Output** panel (right side)
2. Find `mae_checkpoints/best_model.pt`
3. Click the **Download** button
4. Save it as `best_model.pt`

### Step 2 — Set up GitHub repository

```bash
# Clone or create your repo
git clone https://github.com/YOUR_USERNAME/mae-streamlit-app
cd mae-streamlit-app

# Copy all app files into this folder
# app.py, model.py, requirements.txt, packages.txt

# Add your checkpoint (NOT committed to git — too large)
cp /path/to/best_model.pt .

# Push code (without the .pt file — it's in .gitignore)
git add app.py model.py requirements.txt packages.txt README.md .gitignore
git commit -m "Add MAE Streamlit app"
git push origin main
```

### Step 3 — Host the checkpoint (pick one option)

Since `best_model.pt` (~400MB) is too large for GitHub, use one of:

**Option A: Hugging Face Hub (recommended — free)**
```python
# Install: pip install huggingface_hub
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="best_model.pt",
    path_in_repo="best_model.pt",
    repo_id="YOUR_USERNAME/mae-model",
    repo_type="model",
)
```
Then update `app.py` to download it:
```python
from huggingface_hub import hf_hub_download
CHECKPOINT = hf_hub_download(repo_id="YOUR_USERNAME/mae-model", filename="best_model.pt")
```

**Option B: Google Drive + gdown**
```python
# pip install gdown
import gdown
gdown.download(id="YOUR_GDRIVE_FILE_ID", output="best_model.pt")
```

**Option C: Keep it local (for local demo only)**
Just place `best_model.pt` next to `app.py` and run locally.

### Step 4 — Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Fill in:
   - **Repository**: `YOUR_USERNAME/mae-streamlit-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy!**

Streamlit Cloud will:
- Install everything in `requirements.txt`
- Install system packages from `packages.txt`
- Launch your app at `https://YOUR_USERNAME-mae-streamlit-app.streamlit.app`

---

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/mae-streamlit-app
cd mae-streamlit-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your trained checkpoint
cp /path/to/best_model.pt .

# 4. Run the app
streamlit run app.py

# Opens at http://localhost:8501
```

---

## 📊 What the Metrics Mean

| Metric | Range | Good Value | Description |
|---|---|---|---|
| PSNR | 0 → ∞ dB | > 25 dB | Pixel-level accuracy |
| SSIM | 0 → 1 | > 0.70 | Perceptual similarity |

---

## 🎓 Training Details

Trained on [TinyImageNet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet):
- 100,000 training images (200 classes, 500 per class)
- 60 epochs, batch size 64, 2× NVIDIA T4 GPUs
- AdamW optimizer (β₂=0.95, weight decay=0.05)
- 10-epoch linear warmup + cosine LR decay
- Mixed precision (float16) training
