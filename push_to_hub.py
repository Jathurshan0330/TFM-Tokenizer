"""
push_to_hub.py
--------------
Uploads all TFM-Token weights to a single Hugging Face Hub repo:
  - Pretrained TFM-Tokenizer (2x2x8)
  - MTP-pretrained TFM-Encoder (64x4)
  - Finetuned TFM-Encoder checkpoints (3 datasets × 5 seeds)

Usage:
    python push_to_hub.py --hf_org <your-hf-username-or-org>

    # If your finetuned checkpoints are in a non-default location:
    python push_to_hub.py --hf_org <your-org> \
        --results_root ./results/tfm_token_multiple_dataset
"""

import argparse
import json
import os
import shutil
import tempfile

from huggingface_hub import HfApi, create_repo


# ─── Pretrained weight paths ─────────────────────────────────────────────────
TOKENIZER_PTH = (
    "pretrained_weigths/multiple_dataset_settings/Pretrained_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth"
    # "pretrained_weights/multiple_dataset_settings/"
    # "Pretrained_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth"
)
ENCODER_PTH = (
    # "pretrained_weights/multiple_dataset_settings/"
    # "MTP_Pretrained_tfm_encoder_64x4/tfm_encoder_mtp_last.pth"
    "pretrained_weigths/multiple_dataset_settings/MTP_Pretrained_tfm_encoder_64x4/tfm_encoder_mtp_last.pth"
)

# ─── Finetuned checkpoint configuration ──────────────────────────────────────
DATASETS = ["TUEV", "TUAB"] #, "CHBMIT"]
SEEDS = [1, 2, 3, 4, 5]
CODE_BOOK_SIZE = 8192
EMB_SIZE = 64

# Dataset metadata
DATASET_INFO = {
    "TUEV": {
        "num_classes": 6,
        "classification_task": "multi_class",
        "description": "Temple University EEG Event Detection (6-class)",
        "eval_metrics": ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
    },
    "TUAB": {
        "num_classes": 1,
        "classification_task": "binary",
        "description": "Temple University Abnormal EEG Detection (binary)",
        "eval_metrics": ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc"],
    },
    "CHBMIT": {
        "num_classes": 1,
        "classification_task": "binary",
        "description": "CHB-MIT Seizure Detection (binary)",
        "eval_metrics": ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc"],
    },
}


def get_finetuned_path(results_root, dataset, seed):
    """
    Construct the path to a finetuned best_model.pth.

    *** UPDATE THIS FUNCTION to match your actual directory layout. ***

    Common patterns:
      {results_root}/Downstream_Finetuned_{dataset}_{CBS}_{EMB}_seed_{seed}/best_model.pth
      {results_root}/Downstream_Model_FINETUNING_{dataset}_{CBS}_{EMB}/seed_{seed}/best_model.pth
    """
    # --- Pattern A: seed is part of the folder name ---
    path_a = os.path.join(
        results_root,
        f"Downstream_Finetuned_{dataset}_{CODE_BOOK_SIZE}_{EMB_SIZE}_seed_{seed}",
        "best_model.pth",
    )
    if os.path.exists(path_a):
        return path_a

    # --- Pattern B: nested seed subfolder ---
    path_b = os.path.join(
        results_root,
        f"Downstream_Model_FINETUNING_{dataset}_{CODE_BOOK_SIZE}_{EMB_SIZE}",
        f"seed_{seed}",
        "best_model.pth",
    )
    if os.path.exists(path_b):
        return path_b

    # --- Pattern C: random_seed in folder name (matches the training script) ---
    path_c = os.path.join(
        results_root,
        f"Downstream_Model_FINETUNING_{dataset}_{CODE_BOOK_SIZE}_{EMB_SIZE}_random_seed_{seed}",
        "best_model.pth",
    )
    if os.path.exists(path_c):
        return path_c
    
    # --- Pattern D: finetuned_seed in folder name (matches the finetuned models by authors) ---
    path_d = os.path.join(
        results_root,
        f"Finetuned_{dataset}/seed_{seed}/",
        "best_model.pth",
    )
    if os.path.exists(path_d):
        return path_d

    return None  # Not found


# ─── Config ──────────────────────────────────────────────────────────────────

CONFIG = {
    "model_name": "TFM-Token",
    "paper": "Tokenizing Single-Channel EEG with Time-Frequency Motif Learning (ICLR 2026)",
    "paper_url": "https://openreview.net/forum?id=2sPmWHZ8Ir",
    "tokenizer": {
        "architecture": "TFM_VQVAE2_deep",
        "variant": "2x2x8",
        "in_channels": 1,
        "n_freq": 100,
        "n_freq_patch": 5,
        "emb_size": EMB_SIZE,
        "code_book_size": CODE_BOOK_SIZE,
        "trans_freq_encoder_depth": 2,
        "trans_temporal_encoder_depth": 2,
        "trans_decoder_depth": 8,
        "beta": 1.0,
        "resampling_rate": 200,
    },
    "encoder": {
        "architecture": "TFM_TOKEN_Classifier",
        "variant": "64x4",
        "emb_size": EMB_SIZE,
        "code_book_size": CODE_BOOK_SIZE,
        "num_heads": 8,
        "depth": 4,
        "max_seq_len": 2048,
    },
    "pretraining_datasets": ["TUAB", "TUEV", "CHBMIT"],
    "finetuned_datasets": {ds: DATASET_INFO[ds] for ds in DATASETS},
    "seeds": SEEDS,
}


# ─── README template ─────────────────────────────────────────────────────────

README_TEMPLATE = """\
---
license: mit
tags:
  - eeg
  - tokenizer
  - time-frequency
  - vq-vae
  - transformer
  - single-channel-eeg
  - iclr2026
library_name: pytorch
---

# TFM-Token — Multi-Dataset Pretrained & Finetuned Weights

Official pretrained and finetuned weights for
[**Tokenizing Single-Channel EEG with Time-Frequency Motif Learning**](https://openreview.net/forum?id=2sPmWHZ8Ir) (ICLR 2026).

## Repo contents
pretrained/
tfm_tokenizer_last.pth       # TFM-Tokenizer (VQ-VAE, 2x2x8)
tfm_encoder_mtp_last.pth     # TFM-Encoder pretrained via Masked Token Prediction
finetuned/
TUEV/seed_{{1..5}}/best_model.pth   # 6-class EEG event detection
TUAB/seed_{{1..5}}/best_model.pth   # Binary abnormal EEG detection
CHBMIT/seed_{{1..5}}/best_model.pth # Binary seizure detection
models/
tfm_token.py                         # Model definitions

---

## Quick start

### 1. Load the pretrained TFM-Tokenizer

```python
import torch
from huggingface_hub import hf_hub_download
from models.tfm_token import get_tfm_tokenizer_2x2x8
from utils.utils import get_stft_torch

ckpt = hf_hub_download(repo_id="{repo_id}", filename="pretrained/tfm_tokenizer_last.pth")
tokenizer = get_tfm_tokenizer_2x2x8(code_book_size=8192, emb_size=64)
tokenizer.load_state_dict(torch.load(ckpt, map_location="cpu"))
tokenizer.eval()
```

### 2. Load the MTP-pretrained TFM-Encoder (for finetuning on a new task)

```python
from models.tfm_token import get_tfm_token_classifier_64x4

ckpt = hf_hub_download(repo_id="{repo_id}", filename="pretrained/tfm_encoder_mtp_last.pth")
model = get_tfm_token_classifier_64x4(n_classes=YOUR_NUM_CLASSES, code_book_size=8192, emb_size=64)

checkpoint = torch.load(ckpt, map_location="cpu")
filtered = {{k: v for k, v in checkpoint.items() if "classification_head" not in k}}
model.load_state_dict(filtered, strict=False)
# classification_head is randomly initialized — finetune on your data
```

### 3. Load a finetuned checkpoint (for direct inference)

```python
# Example: TUEV dataset, seed 1
ckpt = hf_hub_download(repo_id="{repo_id}", filename="finetuned/TUEV/seed_1/best_model.pth")
model = get_tfm_token_classifier_64x4(n_classes=6, code_book_size=8192, emb_size=64)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()
```

Dataset-specific `n_classes`:
- **TUEV**: `n_classes=6` (multi-class)
- **TUAB**: `n_classes=1` (binary, use sigmoid)
- **CHBMIT**: `n_classes=1` (binary, use sigmoid)

### 4. Full inference pipeline

```python
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from models.tfm_token import get_tfm_tokenizer_2x2x8, get_tfm_token_classifier_64x4
from utils.utils import get_stft_torch

# Load tokenizer
tok_ckpt = hf_hub_download(repo_id="{repo_id}", filename="pretrained/tfm_tokenizer_last.pth")
tokenizer = get_tfm_tokenizer_2x2x8(code_book_size=8192, emb_size=64)
tokenizer.load_state_dict(torch.load(tok_ckpt, map_location="cpu"))
tokenizer.eval()

# Load finetuned encoder (e.g. TUEV seed 1)
enc_ckpt = hf_hub_download(repo_id="{repo_id}", filename="finetuned/TUEV/seed_1/best_model.pth")
encoder = get_tfm_token_classifier_64x4(n_classes=6, code_book_size=8192, emb_size=64)
encoder.load_state_dict(torch.load(enc_ckpt, map_location="cpu"))
encoder.eval()

# Inference on raw EEG: x shape (B, C, T) at 200 Hz
x_temporal = x
B, C, T = x_temporal.shape
x_stft = get_stft_torch(x_temporal, resampling_rate=200)
x_stft = rearrange(x_stft, 'B C F T -> (B C) F T')
x_temporal_flat = rearrange(x_temporal, 'B C T -> (B C) T')

with torch.no_grad():
    _, x_tokens, _ = tokenizer.tokenize(x_stft, x_temporal_flat)
    x_tokens = rearrange(x_tokens, '(B C) T -> B C T', C=C)
    preds = encoder(x_tokens, num_ch=C)
```

---

## Architecture

### TFM-Tokenizer (TFM_VQVAE2_deep, 2x2x8)

| Parameter | Value |
|---|---|
| Freq encoder depth | 2 |
| Temporal encoder depth | 2 |
| Decoder depth | 8 |
| Embedding dim | 64 |
| Codebook size | 8,192 |
| Input sampling rate | 200 Hz |

### TFM-Encoder (TFM_TOKEN_Classifier, 64x4)

| Parameter | Value |
|---|---|
| Embedding dim | 64 |
| Transformer depth | 4 |
| Attention heads | 8 |
| Max sequence length | 2,048 |
| Attention type | Linear Attention |

---

## Pretraining

Multi-dataset setting using TUAB, TUEV, and CHB-MIT.
The TFM-Tokenizer learns a VQ codebook of 8,192 time-frequency motifs.
The TFM-Encoder is then pretrained via Masked Token Prediction (MTP).

## Citation

```bibtex
@inproceedings{{pradeepkumar2026tokenizing,
  title={{Tokenizing Single-Channel {{EEG}} with Time-Frequency Motif Learning}},
  author={{Jathurshan Pradeepkumar and Xihao Piao and Zheng Chen and Jimeng Sun}},
  booktitle={{The Fourteenth International Conference on Learning Representations}},
  year={{2026}},
  url={{https://openreview.net/forum?id=2sPmWHZ8Ir}}
}}
```
"""


def main():
    parser = argparse.ArgumentParser(
        description="Upload TFM-Token weights to Hugging Face Hub"
    )
    parser.add_argument(
        "--hf_org", type=str, required=True,
        help="HF username or organization (e.g. 'jathurshan')",
    )
    parser.add_argument(
        "--repo_name", type=str, default="TFM-Tokenizer",
        help="Repository name (default: TFM-Tokenizer",
    )
    parser.add_argument(
        "--tokenizer_pth", type=str, default=TOKENIZER_PTH,
        help="Path to tfm_tokenizer_last.pth",
    )
    parser.add_argument(
        "--encoder_pth", type=str, default=ENCODER_PTH,
        help="Path to tfm_encoder_mtp_last.pth",
    )
    parser.add_argument(
        "--results_root", type=str,
        default="./results/tfm_token_multiple_dataset",
        help="Root folder containing finetuned experiment results",
    )
    parser.add_argument(
        "--include_source", action="store_true",
        help="Also upload models/tfm_token.py for self-contained usage",
    )
    args = parser.parse_args()

    repo_id = f"{args.hf_org}/{args.repo_name}"
    api = HfApi()

    # Create repo
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"\nRepo: https://huggingface.co/{repo_id}")

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── 1. Pretrained weights ─────────────────────────────────────────
        pretrained_dir = os.path.join(tmpdir, "pretrained")
        os.makedirs(pretrained_dir)

        assert os.path.exists(args.tokenizer_pth), \
            f"Tokenizer weights not found: {args.tokenizer_pth}"
        shutil.copy2(args.tokenizer_pth,
                     os.path.join(pretrained_dir, "tfm_tokenizer_last.pth"))
        print(f"  [OK] Tokenizer: {args.tokenizer_pth}")

        assert os.path.exists(args.encoder_pth), \
            f"Encoder weights not found: {args.encoder_pth}"
        shutil.copy2(args.encoder_pth,
                     os.path.join(pretrained_dir, "tfm_encoder_mtp_last.pth"))
        print(f"  [OK] MTP Encoder: {args.encoder_pth}")

        # ── 2. Finetuned checkpoints ──────────────────────────────────────
        finetuned_dir = os.path.join(tmpdir, "finetuned")
        found_count = 0
        missing = []

        for dataset in DATASETS:
            for seed in SEEDS:
                src = get_finetuned_path(args.results_root, dataset, seed)
                dst_dir = os.path.join(finetuned_dir, dataset, f"seed_{seed}")
                os.makedirs(dst_dir, exist_ok=True)

                if src and os.path.exists(src):
                    shutil.copy2(src, os.path.join(dst_dir, "best_model.pth"))
                    found_count += 1
                    print(f"  [OK] {dataset}/seed_{seed}: {src}")

                    # Also copy test results CSV if available
                    results_csv = os.path.join(
                        os.path.dirname(src),
                        f"test_results_random_seed_{seed}_best.csv"
                    )
                    if os.path.exists(results_csv):
                        shutil.copy2(results_csv,
                                     os.path.join(dst_dir, "test_results.csv"))
                        print(f"       + test_results.csv")
                else:
                    missing.append(f"{dataset}/seed_{seed}")
                    print(f"  [MISSING] {dataset}/seed_{seed} — "
                          f"update get_finetuned_path() for your directory layout")

        print(f"\n  Found {found_count}/{len(DATASETS)*len(SEEDS)} finetuned checkpoints")
        if missing:
            print(f"  Missing: {', '.join(missing)}")
            print(f"  -> Edit get_finetuned_path() in this script to match your paths")
            resp = input("  Continue uploading with available files? [y/N] ")
            if resp.lower() != "y":
                print("Aborted.")
                return

        # ── 3. Model source code (optional) ───────────────────────────────
        if args.include_source:
            models_dir = os.path.join(tmpdir, "models")
            os.makedirs(models_dir)
            src_path = "models/tfm_token.py"
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(models_dir, "tfm_token.py"))
                print(f"  [OK] Source: {src_path}")
            else:
                print(f"  [WARN] {src_path} not found, skipping")

        # ── 4. config.json ────────────────────────────────────────────────
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(CONFIG, f, indent=2)

        # ── 5. README.md ─────────────────────────────────────────────────
        readme_path = os.path.join(tmpdir, "README.md")
        with open(readme_path, "w") as f:
            f.write(README_TEMPLATE.format(repo_id=repo_id))

        # ── 6. Upload everything ──────────────────────────────────────────
        print(f"\nUploading to {repo_id}...")
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            repo_type="model",
        )

    print(f"\nDone! View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
    
    
# python push_to_hub.py --hf_org Jathurshan --results_root /home/jp65/Biosignals_Research/tfm_token_code_for_release/TFM-Tokenizer/pretrained_weigths/multiple_dataset_settings