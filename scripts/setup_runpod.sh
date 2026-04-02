#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# RunPod environment setup — run once after spinning up the instance.
#
# Recommended pod: RTX 4090 (24GB) or A100 40GB
# Template:        RunPod PyTorch 2.1 (has CUDA 12.1 pre-installed)
#
# Usage (run from /workspace or any directory outside the repo):
#   bash medvision-eval/scripts/setup_runpod.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit on first error

echo "============================================================"
echo " MedVision-Eval — RunPod Setup"
echo "============================================================"

# ── 1. Clone repo (skip if already cloned) ───────────────────────────────────
if [ ! -d "medvision-eval" ]; then
    echo "[1/7] Cloning repo..."
    git clone https://github.com/tanmaynaik11/medqaeval.git medvision-eval
else
    echo "[1/7] Repo already present — pulling latest..."
    cd medvision-eval && git pull && cd ..
fi

cd medvision-eval

# ── 2. Create virtual environment ────────────────────────────────────────────
# --system-site-packages: inherits RunPod's pre-built torch + CUDA drivers
# so we don't reinstall a 2GB torch from scratch.
# Everything we install goes into the venv, never touching system packages.
echo "[2/7] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python -m venv .venv --system-site-packages
    echo "  Virtual environment created at .venv"
else
    echo "  Virtual environment already exists — reusing"
fi

# Activate for the rest of this script
source .venv/bin/activate
echo "  Python: $(which python)  $(python --version)"

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo "[3/7] Installing dependencies..."
pip install --upgrade pip --quiet

# Install all GPU requirements into the venv (no system conflicts)
pip install -r requirements-gpu.txt --quiet

# FlashAttention-2 — needs --no-build-isolation so setup.py can see torch.
# Only meaningful on A100/H100. Skip on RTX 4090 (no speedup, slower compile).
# Uncomment if using A100 or H100:
# pip install flash-attn --no-build-isolation --quiet

echo "  Dependencies installed."

# ── 4. Set environment variables ─────────────────────────────────────────────
echo "[4/7] Setting environment variables..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
    echo "  Edit .env now — add HF_TOKEN and WANDB_API_KEY before continuing!"
fi

# Add venv activation to .bashrc so new terminals auto-activate
if ! grep -q "medvision-eval/.venv" ~/.bashrc 2>/dev/null; then
    echo "source /workspace/medvision-eval/.venv/bin/activate" >> ~/.bashrc
    echo "  Added venv auto-activation to ~/.bashrc"
fi

export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# ── 5. Download datasets ──────────────────────────────────────────────────────
echo "[5/7] Downloading datasets..."
python scripts/download_data.py
echo "  Datasets ready."

# ── 6. Create required directories ───────────────────────────────────────────
echo "[6/7] Creating directories..."
mkdir -p logs artifacts/checkpoints/stage1 artifacts/checkpoints/stage2 artifacts/model_cache

# ── 7. Verify GPU and CUDA ────────────────────────────────────────────────────
echo "[7/7] Verifying GPU..."
python -c "
import torch
print(f'  PyTorch       : {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU           : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import bitsandbytes as bnb
print(f'  bitsandbytes  : {bnb.__version__}')
"

echo ""
echo "============================================================"
echo " Setup complete. Next steps:"
echo ""
echo "  1. Edit .env — add HF_TOKEN and WANDB_API_KEY:"
echo "     nano .env"
echo ""
echo "  2. Smoke tests (validates full pipeline, ~5 min each):"
echo "     python src/training/train_stage1.py --smoke"
echo "     python src/training/train_stage2.py --smoke"
echo ""
echo "  3. Full training (run in background with nohup):"
echo "     nohup python src/training/train_stage1.py > logs/stage1_stdout.log 2>&1 &"
echo "     nohup python src/training/train_stage2.py > logs/stage2_stdout.log 2>&1 &"
echo ""
echo "  Monitor: tail -f logs/stage1_sft.log"
echo "============================================================"
