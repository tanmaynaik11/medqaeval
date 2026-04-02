#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# RunPod environment setup — run once after spinning up the instance.
#
# Recommended pod: RTX 4090 (24GB) or A100 40GB
# Template:        RunPod PyTorch 2.1 (has CUDA 12.1 pre-installed)
#
# Usage:
#   bash scripts/setup_runpod.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit on first error

echo "============================================================"
echo " MedVision-Eval — RunPod Setup"
echo "============================================================"

# ── 1. Clone repo (skip if already cloned) ────────────────────────────────────
if [ ! -d "medvision-eval" ]; then
    echo "[1/6] Cloning repo..."
    git clone https://github.com/tanmaynaik11/medqaeval.git medvision-eval
else
    echo "[1/6] Repo already present — pulling latest..."
    cd medvision-eval && git pull && cd ..
fi

cd medvision-eval

# ── 2. Install GPU dependencies ───────────────────────────────────────────────
echo "[2/6] Installing dependencies..."
pip install --upgrade pip --quiet

# Install base + GPU requirements (bitsandbytes, wandb, etc.)
pip install -r requirements-gpu.txt --quiet

# FlashAttention-2 (speeds up attention on A100/H100, skip on RTX 4090)
# Uncomment if using A100 or H100:
# pip install flash-attn --no-build-isolation --quiet

echo "Dependencies installed."

# ── 3. Set environment variables ──────────────────────────────────────────────
echo "[3/6] Setting environment variables..."

# Copy .env.example if no .env exists
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
    echo "  ⚠ Set HF_TOKEN and WANDB_API_KEY in .env before training!"
fi

# Suppress symlink warning (Windows only issue, harmless on Linux)
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# ── 4. Download datasets ──────────────────────────────────────────────────────
echo "[4/6] Downloading datasets..."
python scripts/download_data.py

echo "Datasets ready."

# ── 5. Create required directories ───────────────────────────────────────────
echo "[5/6] Creating directories..."
mkdir -p logs artifacts/checkpoints/stage1 artifacts/checkpoints/stage2 artifacts/model_cache

# ── 6. Verify GPU and CUDA ────────────────────────────────────────────────────
echo "[6/6] Verifying GPU..."
python -c "
import torch
print(f'  PyTorch      : {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU          : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import bitsandbytes as bnb
print(f'  bitsandbytes : {bnb.__version__}')
"

echo ""
echo "============================================================"
echo " Setup complete. Next steps:"
echo ""
echo "  1. Edit .env — add HF_TOKEN and WANDB_API_KEY"
echo ""
echo "  2. Smoke test (validates pipeline, ~5 min):"
echo "     python src/training/train_stage1.py --smoke"
echo "     python src/training/train_stage2.py --smoke"
echo ""
echo "  3. Full training:"
echo "     python src/training/train_stage1.py    (~6-8 hrs, RTX 4090)"
echo "     python src/training/train_stage2.py    (~3-5 hrs, RTX 4090)"
echo "============================================================"
