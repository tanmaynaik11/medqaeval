#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# RunPod environment setup — run once after spinning up the instance.
#
# Recommended pod: RTX 4090 (24GB) or A100 40GB
# Template:        RunPod PyTorch 2.1 (has CUDA 12.1 pre-installed)
#
# Usage (run from /workspace):
#   bash medvision-eval/scripts/setup_runpod.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "============================================================"
echo " MedVision-Eval — RunPod Setup"
echo "============================================================"

# ── 1. Clone repo (skip if already cloned) ───────────────────────────────────
if [ ! -d "medvision-eval" ]; then
    echo "[1/5] Cloning repo..."
    git clone https://github.com/tanmaynaik11/medqaeval.git medvision-eval
else
    echo "[1/5] Repo already present — pulling latest..."
    cd medvision-eval && git pull && cd ..
fi

cd medvision-eval

# ── 2. Install dependencies into system Python ────────────────────────────────
echo "[2/5] Installing dependencies..."
pip install --upgrade pip --quiet

# Install CUDA torch FIRST with the correct index URL.
# PyPI torch is CPU-only — it silently overwrites RunPod's pre-installed CUDA
# torch if listed in requirements.txt, causing the model to run on CPU.
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --quiet

pip install -r requirements-gpu.txt \
    --ignore-installed blinker charset-normalizer \
    --quiet

# FlashAttention-2: needs --no-build-isolation (setup.py imports torch).
# Only meaningful on A100/H100 — skip on RTX 4090.
# Uncomment if using A100 or H100:
# pip install flash-attn --no-build-isolation --quiet

echo "  Dependencies installed."

# ── 3. Set environment variables ─────────────────────────────────────────────
echo "[3/5] Setting environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  Created .env — edit it now to add HF_TOKEN and WANDB_API_KEY"
fi
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# ── 4. Create required directories ───────────────────────────────────────────
echo "[4/5] Creating directories..."
mkdir -p logs artifacts/checkpoints/stage1 artifacts/checkpoints/stage2 artifacts/model_cache

# ── 5. Verify GPU and CUDA ────────────────────────────────────────────────────
echo "[5/5] Verifying GPU..."
python -W ignore -c "
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
echo "     cat > .env << 'EOF'"
echo "     HF_TOKEN=hf_your_token"
echo "     WANDB_API_KEY=your_key"
echo "     EOF"
echo ""
echo "  2. Smoke tests (~5 min each):"
echo "     python -W ignore src/training/train_stage1.py --smoke"
echo "     python -W ignore src/training/train_stage2.py --smoke"
echo ""
echo "  3. Full training:"
echo "     nohup python -W ignore src/training/train_stage1.py > logs/stage1_stdout.log 2>&1 &"
echo "     nohup python -W ignore src/training/train_stage2.py > logs/stage2_stdout.log 2>&1 &"
echo ""
echo "  Monitor: tail -f logs/stage1_sft.log"
echo "============================================================"
