# MedVision

A multimodal medical LLM built by fine-tuning Qwen2-7B-Instruct with a domain-specific vision encoder, evaluated against Qwen2-VL-7B-Instruct on standard medical benchmarks.

---

## Architecture

MedVision follows a LLaVA-style architecture with three components:

```
BiomedCLIP (frozen)  →  VisionProjection (trained)  →  ┐
                                                         ├→  Qwen2-7B + LoRA  →  output
Tokenizer            →  Text Embeddings              →  ┘
```

### 1. Vision Encoder — BiomedCLIP ViT-B/16

Microsoft's [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) pretrained on 15 million PubMed Central figure-caption pairs. Unlike standard CLIP (trained on general web images), BiomedCLIP understands medical visual concepts: cell morphology, tissue structure, imaging artifacts, and anatomical landmarks.

- Input: `[B, 3, 224, 224]`
- Output: 196 patch embeddings (16×16 patches, CLS token stripped)
- Hidden dim: 768
- Frozen during all training stages

### 2. Vision Projection MLP

A 2-layer MLP with GELU activation that bridges BiomedCLIP's embedding space (768-dim) and Qwen2-7B's token embedding space (3584-dim). This is the only component trained from scratch.

```
[B, 196, 768]  →  Linear(768→3584)  →  GELU  →  Linear(3584→3584)  →  [B, 196, 3584]
```

The non-linear mapping is necessary because BiomedCLIP's geometry (contrastive loss on medical images) differs structurally from the LLM's embedding space (next-token prediction on text).

### 3. Language Model — Qwen2-7B-Instruct + QLoRA

Qwen2-7B-Instruct quantized to 4-bit NF4 via bitsandbytes, reducing VRAM from ~14GB to ~4GB. LoRA adapters (r=16, α=32) are added to `q_proj`, `k_proj`, `v_proj`, `o_proj` — only these adapter weights are trained.

### Image-Text Merging

The prompt contains a special `<image>` token as a placeholder. During the forward pass:

1. Encode image into 196 patch embeddings via BiomedCLIP
2. Project patches to LLM embedding space via the MLP
3. Find the `<image>` token position in `input_ids`
4. Replace the single `<image>` token with 196 patch embeddings
5. Forward the merged sequence through the LLM

```
Before merge:  [Q, u, e, <image>, t, e, x, t]    length = S
After merge:   [Q, u, e, p0..p195, t, e, x, t]   length = S - 1 + 196
```

---

## Training

Training is split into two stages. Both use supervised fine-tuning (SFT) with label masking: only answer tokens contribute to the loss, not prompt tokens.

### Stage 1 — Medical Text SFT

Teaches the LLM medical reasoning on text-only data before introducing visual input.

- **Data:** MedMCQA (30K stratified samples across 21 subjects) + MedQA-USMLE (9K)
- **Trains:** LoRA adapters only (projection frozen, no images)
- **Label format:** `"A. Hypertension. Because the kidneys regulate..."` — full answer + clinical explanation
- **Effective batch size:** 32 (batch=1 × grad_accum=32)
- **Steps:** ~10,500 (early stopped at val loss plateau ~1.22)
- **Config:** `configs/training/stage1_sft.yaml`

### Stage 2 — Multimodal SFT

Teaches the projection MLP to bridge vision and language, then jointly fine-tunes it with LoRA on image+text data.

- **Data:** PathVQA train split (~15K image+QA pairs)
- **Trains:** Projection MLP + LoRA adapters (initialized from Stage 1 best checkpoint)
- **Label format:** `"yes"` / `"no"` for closed questions; free-form answer for open questions
- **Epochs:** 2
- **Config:** `configs/training/stage2_sft.yaml`

### Key Training Decisions

| Decision | Reason |
|---|---|
| `batch_size=1, grad_accum=32` | 24GB RTX 4090 OOM at batch=4 with 7B model + activations |
| `max_length=256` | Reduces activation memory; medical QA answers are short |
| Gradient checkpointing | Recomputes activations on-demand instead of storing all of them |
| No `torch.amp.autocast` | Causes CUDA lazy init hang on some RunPod instances; bitsandbytes handles bf16 internally |
| No Accelerate | `accelerator.accumulate()` deadlocks with bitsandbytes `device_map="auto"` |
| `dataloader_workers=0` | Docker/RunPod containers have limited `/dev/shm`; multiprocessing deadlocks |
| `save_embedding_layers=False` | Prevents checkpoint bloat: LoRA adapters are ~80MB, not ~8GB |

---

## Checkpoints

Trained checkpoints are available on HuggingFace: [tanmaynaik11/medvision](https://huggingface.co/tanmaynaik11/medvision)

```
stage1/best/
  lora_adapters/   (~39 MB)  — Stage 1 LoRA weights
  projection.pt    (~18 MB)  — projection MLP state dict

stage2/best/
  lora_adapters/   (~39 MB)  — Stage 2 LoRA weights (multimodal)
  projection.pt    (~18 MB)  — updated projection weights
```

---

## Datasets

| Dataset | Task | Size | Used For |
|---|---|---|---|
| [PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) | Pathology image VQA | 32K train / 6K test | Stage 2 training + evaluation |
| [MedMCQA](https://huggingface.co/datasets/medmcqa) | Medical MCQ (21 subjects) | 182K train / 4K val | Stage 1 training + evaluation |
| [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | USMLE-style clinical MCQ | 10K train / 1.3K test | Stage 1 training + evaluation |

**Data isolation:** PathVQA and MedQA-USMLE test splits are never seen during training. MedMCQA's public test split has no released labels, so the validation split is used for evaluation (standard practice).

---

## Evaluation

Compares MedVision against Qwen2-VL-7B-Instruct as the multimodal baseline.

### Benchmarks

**PathVQA** — multimodal (image + question → answer)
- Yes/No questions (~80% of test set): binary classification
- Open questions (~20%): free-form answer, evaluated on partial exact match

**MedMCQA** — text-only, 4-option MCQ across 21 medical subjects

**MedQA-USMLE** — text-only, 4-option USMLE-style clinical MCQ

### Answer Extraction

Models output free text. Answers are extracted via regex before scoring:

```python
# MCQ: handles "A", "A.", "A. Hypertension", "The answer is A", "Answer: B"
extract_option_letter(text) → "A" | "B" | "C" | "D" | None

# PathVQA yes/no
extract_yes_no(text) → "yes" | "no" | None
```

Unparseable outputs are counted as `skipped` (not wrong), and accuracy is computed over answered samples only.

### Results (100-sample smoke test)

| Benchmark | MedVision | Qwen2-VL-7B |
|---|---|---|
| PathVQA (overall) | 59% | 62% |

Full evaluation results pending.

### Running Evaluation

```bash
# Both models, full test sets
python -W ignore scripts/evaluate.py --model both

# Smoke test (100 samples per benchmark)
python -W ignore scripts/evaluate.py --model both --max_samples 100

# Single model
python -W ignore scripts/evaluate.py --model medvision
python -W ignore scripts/evaluate.py --model qwen2vl
```

Results saved to `artifacts/eval_results.json`.

---

## Project Structure

```
medvision-eval/
├── configs/
│   └── training/
│       ├── stage1_sft.yaml       # Stage 1 hyperparameters
│       └── stage2_sft.yaml       # Stage 2 hyperparameters
├── scripts/
│   ├── evaluate.py               # Evaluation entry point
│   ├── train_stage1.py           # Stage 1 training entry point
│   ├── train_stage2.py           # Stage 2 training entry point
│   └── setup_runpod.sh           # RunPod environment setup
├── src/
│   ├── data/
│   │   ├── ingestion.py          # HuggingFace dataset loaders
│   │   ├── preprocessing.py      # Prompt builders, dataset mappers
│   │   ├── dataset.py            # PyTorch Dataset wrappers
│   │   └── collator.py           # Batch collation + label masking
│   ├── models/
│   │   ├── multimodal.py         # MedVisionModel (full architecture)
│   │   ├── vision_encoder.py     # BiomedCLIP + CLIP encoders
│   │   └── projection.py        # Vision→language projection MLP
│   ├── training/
│   │   ├── trainer.py            # SFTTrainer (direct PyTorch loop)
│   │   ├── train_stage1.py       # Stage 1 script
│   │   └── train_stage2.py       # Stage 2 script
│   ├── evaluation/
│   │   ├── evaluator.py          # MedVisionEvaluator, Qwen2VLEvaluator
│   │   └── metrics.py            # Answer extraction + accuracy computation
│   └── utils/
│       ├── env.py                # .env loader + HF/W&B login
│       ├── logging.py            # Logging setup
│       └── reproducibility.py   # Seed setting
├── RUNPOD_ISSUES.md              # 19 RunPod training issues and fixes
├── requirements.txt
└── requirements-gpu.txt
```

---

## Setup

### RunPod

```bash
git clone https://github.com/tanmaynaik11/medqaeval.git medvision-eval
cd medvision-eval

# Install torch with correct CUDA index (do NOT use PyPI torch — CPU-only)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt --ignore-installed blinker charset-normalizer
pip install -r requirements-gpu.txt

# Set credentials
echo "HF_TOKEN=your_token" >> .env
echo "WANDB_API_KEY=your_key" >> .env
```

### Training

```bash
# Stage 1 smoke test (fast pipeline check)
python -W ignore src/training/train_stage1.py --smoke

# Stage 1 full
python -W ignore src/training/train_stage1.py

# Stage 2 full
python -W ignore src/training/train_stage2.py
```

---

## Known Issues

See [RUNPOD_ISSUES.md](RUNPOD_ISSUES.md) for a full log of 19 environment issues encountered during training on RunPod (torch CUDA version mismatches, device placement bugs, label masking bug, OOM, etc.) and their fixes.

---

## Planned Improvements

- **Larger vision encoder** — ViT-L/14 at 336×336 (576 patches) for finer-grained pathology features
- **Dynamic resolution tiling** — split large images into crops, concatenate tokens (Qwen2-VL's approach)
- **Perceiver resampler** — compress 196+ patches into fixed M tokens via cross-attention
- **Unfreeze top vision layers** — allow BiomedCLIP to adapt visual representations to the QA task
- **Expanded Stage 2 data** — PatchCamelyon, PathMMU, TCGA tiles for broader pathology coverage
