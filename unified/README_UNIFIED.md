# Expressora — Unified Recognition Model

**App-ready sign language recognition with origin tracking, quantization, evaluation, and live inference.**

---

# Unified Model – Quick Start (Windows)

## One-time
1. Open **PowerShell** at the repo root (folder with `unified/` and `scripts/`).
2. Run:
   ```
   scripts\setup_venv.ps1
   ```
   This will:
   - Find **Python 3.10** (or fail clearly)
   - Create `.venv` (works around `ensurepip` issues automatically)
   - Install `unified/requirements.txt`

## Run the pipeline
```
scripts\run_unified.ps1
```
This will:
1) Build the unified dataset (pads one-hand CSVs to 126-dim)
2) Train the TF model
3) Export TFLite + labels
4) Smoke-test the TFLite

Artifacts land in `unified/models/`:
- `expressora_unified.tflite`
- `expressora_labels.json`

> If PowerShell blocks scripts: launch PS as Admin (or run once):
> `Set-ExecutionPolicy -Scope Process Bypass`

---

## 1) Environment (Manual Setup)

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\pip install -r unified/requirements.txt
```

macOS/Linux:
```bash
source .venv/bin/activate && pip install -r unified/requirements.txt
```

## 2) Build dataset (auto-scans repo CSVs)

```bash
python unified/data/build_unified_dataset.py
```

Outputs:
- `unified/data/unified_X.npy` (features, shape: [N, 126])
- `unified/data/unified_y.npy` (label indices)
- `unified/data/labels.json` (label list in Concept-Key order)

## 3) Train

```bash
python unified/training/train_unified_tf.py
```

Outputs under `unified/models/`:
- `expressora_unified.keras`
- `savedmodel/`
- `training_log.json`

## 4) Export TFLite + labels (app-ready)

```bash
python unified/export/export_unified_tflite.py
```

Outputs under `unified/models/`:
- `expressora_unified.tflite`
- `expressora_labels.json`

## 5) Drop into Android app (Expressora)

Copy both files to:
```
app/src/main/assets/
  - expressora_unified.tflite
  - expressora_labels.json
```

---

# Complete Pipeline Runbook

## Prerequisites

- **Python 3.11** (Python 3.10 also supported)
- Active virtual environment (`.venv`)
- Windows PowerShell 5.1+

## Environment Setup

The pipeline assumes an **active virtual environment**. All scripts detect and use the current venv automatically—never switch interpreters mid-pipeline.

```powershell
# One-time setup
.\scripts\setup_venv.ps1

# Activate venv (if not already active)
.\.venv\Scripts\Activate.ps1
```

---

# Feature Documentation

## 1. Origin Tracking (ASL/FSL Provenance)

### Overview

The dataset builder automatically infers origin (ASL or FSL) from file paths and saves provenance metadata alongside the unified dataset.

### How Origin is Inferred

**Path segment rules:**
- **FSL**: Contains `FSL`, `FSLBasic`, or `TFModelsFSL`
- **ASL**: Contains `ASL`, `ASLBasic`, or `TFModels` (non-FSL)
- **UNKNOWN**: Ambiguous or neither → excluded from origin training

### Generated Artifacts

- `unified/data/unified_origin.npy` — Origin indices (0=ASL, 1=FSL) for each sample
- `unified/data/origin_labels.json` — `["ASL", "FSL"]`
- `unified/data/label_origin_stats.json` — Per-gloss, per-origin counts for QA and fallback priors

### Usage

Run the dataset builder as usual:

```powershell
python unified/data/build_unified_dataset.py
```

Output includes origin distribution:
```
Origin distribution:
  ASL: 12500 (62.5%)
  FSL: 7500 (37.5%)
```

---

## 2. Multi-Task Training (Gloss + Origin)

### Architecture

- **Shared backbone**: Dense layers process 126-dim hand features
- **Head A**: Gloss classification (softmax over all labels)
- **Head B**: Origin classification (softmax over ASL/FSL)

### Combined Loss

```
L = L_gloss + λ * L_origin
```

Default `λ = 0.3` (configurable via CLI flag).

### Training Script

**Option 1**: Enhanced single-head (current model)
```powershell
python unified/training/train_unified_tf.py
```

**Option 2**: Multi-task model (future)
```powershell
python unified/training/train_multitask.py --lambda-origin 0.3
```

### Class Weighting

Auto class-weighting is applied to the origin head if imbalanced (e.g., 70% ASL vs 30% FSL).

---

## 3. Quantization (FP16, INT8)

### Export Formats

| Format | Size | Use Case |
|--------|------|----------|
| Float32 | ~100% | Baseline, high accuracy |
| FP16 | ~50% | Mobile GPU, minimal accuracy loss |
| INT8 | ~25% | CPU inference, fast but may lose 1-2% accuracy |

### Representative Dataset

INT8 quantization requires calibration data:

```powershell
python unified/data/build_representative_set.py
```

Creates `unified/data/rep_set.npy` (≈1000 stratified samples).

### Run Quantization Pipeline

```powershell
.\scripts\run_quantize.ps1
```

**Steps:**
1. Builds representative set
2. Exports `expressora_unified_fp16.tflite`
3. Exports `expressora_unified_int8.tflite`
4. Prints size comparison

**Output:**
```
Model Size Comparison
  Float32 (base)    2.15 MB
  FP16 quantized    1.08 MB (-49.8%)
  INT8 quantized    0.58 MB (-73.0%)
```

---

## 4. Evaluation

### Gloss Classification

```powershell
.\scripts\run_eval.ps1
```

**Outputs:**
- `unified/eval/confusion_matrix.png`
- `unified/eval/baseline.json` (for CI regression checks)
- Per-class precision/recall/F1
- Worst→best classes by F1

### Origin Classification

If origin head exists, evaluation also runs:

```powershell
python unified/eval/eval_origin.py
```

**Outputs:**
- `unified/eval/origin_confusion.png`
- `unified/eval/origin_baseline.json`
- Per-origin precision/recall/F1
- **Mono-origin label warnings** (shortcut learning risk)

---

## 5. Live Inference Enhancements

### Unknown / Low-Confidence Gating

**Configuration:**
- `CONF_THRESHOLD=0.65` — Top-1 softmax must exceed to display label
- `HOLD_FRAMES=3` — Stable frames before label update

Display shows **"UNKNOWN"** when confidence < threshold.

**Run with custom threshold:**

```powershell
.\scripts\run_live.ps1 -Threshold 0.70
```

### Alphabet Accumulator Mode

**Behavior:**
- Single-letter labels (a, b, c, ..., z) are treated as alphabet signs
- Letters accumulate into a word buffer
- Word commits after `IDLE_TIMEOUT_MS=1000` without new letters
- Displays: current word + committed text

**Enable alphabet mode:**

```powershell
.\scripts\run_live.ps1 -AlphabetMode
```

**Overlay:**
```
Word: hello_
Text: i am
```

### Origin Badge Display

If the model has an origin head, live inference displays:

```
Origin: ASL     or     Origin: FSL
```

**Fallback:** If single-output model, estimates origin from `label_origin_stats.json` priors:

```
Origin: ASL~   (~ indicates prior-based estimate)
```

**Configuration:**
- `ORIGIN_CONF_THRESHOLD=0.70` — Minimum confidence to display origin
- If below, shows `Origin: UNKNOWN`

**Enable/disable origin badge:**

```powershell
.\scripts\run_live.ps1 -ShowOrigin:$true
```

---

## 6. Label → Gloss Bridge

Maps model labels (e.g., `thank_you`, `i_love_you`) to translation module gloss tokens.

**Mapping rules:**
- Lowercase
- Underscore → space
- Explicit overrides in `unified/bridge/concept_key_map.json`

**Run mapping:**

```powershell
python unified/bridge/apply_label_map.py
```

**Output:** `unified/models/expressora_labels_mapped.json`

---

## 7. Model Card & Versioning

Comprehensive metadata for reproducibility.

**Generate model card:**

```powershell
python unified/export/write_model_card.py
```

**Output:** `unified/models/model_card.json`

**Contents:**
- Git SHA, timestamp
- Dataset: sample count, label count, SHA256 of `labels.json`
- Training: epochs, final accuracy, val loss
- Quantized models: file sizes (float32/fp16/int8)
- Live thresholds: confidence, hold frames
- **Origin section**: distribution, accuracy, fallback priors, SHA256 of origin metadata

---

## 8. CI/CD Regression Guard

**Workflow:** `.github/workflows/unified-eval.yml`

**Triggers:** Push/PR to `main` or `master`

**Checks:**
1. Gloss classification accuracy & macro-F1 vs `unified/eval/baseline.json`
   - Fail if regression > 1%
2. Origin classification accuracy vs `unified/eval/origin_baseline.json`
   - Fail if < 95% or regression > 1%

**Artifacts uploaded:**
- Confusion matrices (gloss + origin)
- Model card
- Evaluation results

**Update baseline:**

```powershell
# After training improvements, run eval locally
.\scripts\run_eval.ps1

# Commit updated baseline.json with justification in PR
git add unified/eval/baseline.json
git commit -m "chore: update baseline after data augmentation"
```

---

# Complete Workflow

## Step 1: Train

```powershell
.\scripts\run_unified.ps1
```

**Pipeline:**
1. Build unified dataset (+ origin tracking)
2. Train Keras model
3. Export TFLite (float32)
4. Quick test
5. Generate label mapping (non-fatal)
6. Generate model card (non-fatal)

**Artifacts:** `unified/models/`
- `expressora_unified.keras`
- `expressora_unified.tflite`
- `expressora_labels.json`
- `model_card.json`

## Step 2: Evaluate

```powershell
.\scripts\run_eval.ps1
```

**Outputs:** Metrics + confusion matrices

## Step 3: Quantize

```powershell
.\scripts\run_quantize.ps1
```

**Outputs:** FP16 & INT8 TFLite models

## Step 4: Live Test

```powershell
# Standard mode
.\scripts\run_live.ps1

# With custom settings
.\scripts\run_live.ps1 -Threshold 0.75 -AlphabetMode -ShowOrigin:$true
```

---

# Backward Compatibility

## Single-Output Models

All live inference and evaluation scripts gracefully handle single-output (gloss-only) models:

- **Live cam**: Origin badge falls back to priors from `label_origin_stats.json`
- **Evaluation**: Origin eval skips if no origin head detected

## Migration Path

1. Train current single-head model → works with all scripts
2. Later, retrain with multi-head → enhanced origin tracking without breaking existing pipeline

---

# Troubleshooting

## Virtual Environment Issues

**Symptom:** Scripts fail with "module not found" or wrong Python version

**Solution:**
1. Activate venv: `.\.venv\Scripts\Activate.ps1`
2. Verify: `python --version` (should show 3.11.x)
3. Reinstall deps: `pip install -r unified/requirements.txt`

## Model Not Found

**Symptom:** `FileNotFoundError: expressora_unified.keras`

**Solution:** Run training first:
```powershell
.\scripts\run_unified.ps1
```

## Quantization Fails

**Symptom:** INT8 export error

**Solution:** Ensure representative set exists:
```powershell
python unified/data/build_representative_set.py
```

## CI Fails on Regression

**Symptom:** GitHub Actions fails with "accuracy regressed"

**Solution:**
1. Investigate: download artifacts from CI run
2. If intentional (e.g., data removed), update baseline locally and commit
3. If bug, fix training/data before merging

---

# File Structure

```
unified/
├── data/
│   ├── build_unified_dataset.py    # Dataset builder with origin tracking
│   ├── build_representative_set.py # Rep set for INT8 quantization
│   ├── unified_X.npy               # Features (126-dim)
│   ├── unified_y.npy               # Gloss labels
│   ├── unified_origin.npy          # Origin labels (0=ASL, 1=FSL)
│   ├── labels.json                 # Gloss label list
│   ├── origin_labels.json          # ["ASL", "FSL"]
│   └── label_origin_stats.json     # Per-gloss origin counts
├── training/
│   └── train_unified_tf.py         # Single-head training
├── export/
│   ├── export_unified_tflite.py    # Float32 export
│   ├── export_quant_fp16.py        # FP16 quantization
│   ├── export_quant_int8.py        # INT8 quantization
│   └── write_model_card.py         # Model card generator
├── eval/
│   ├── eval_unified.py             # Gloss evaluation
│   ├── eval_origin.py              # Origin evaluation
│   ├── baseline.json               # CI baseline (gloss)
│   └── origin_baseline.json        # CI baseline (origin)
├── live/
│   └── live_cam_unified.py         # Live inference with unknown gating, alphabet mode, origin badge
├── bridge/
│   ├── concept_key_map.json        # Label→gloss mapping rules
│   └── apply_label_map.py          # Mapping utility
├── models/                         # Generated artifacts (gitignored, LFS)
├── requirements.txt                # Core deps (numpy, pandas, TF)
└── requirements-eval.txt           # Eval deps (scikit-learn, matplotlib)

scripts/
├── setup_venv.ps1                  # One-time venv setup
├── run_unified.ps1                 # Train → export → test
├── run_eval.ps1                    # Evaluation pipeline
├── run_quantize.ps1                # Quantization pipeline
└── run_live.ps1                    # Live inference with params

.github/workflows/
└── unified-eval.yml                # CI regression guard
```

---

# Version Requirements

- **Python:** 3.11.x (3.10.x also supported)
- **TensorFlow:** 2.15.1
- **NumPy:** 1.26.4
- **Pandas:** 2.1.4
- **scikit-learn:** 1.3.2
- **MediaPipe:** ≥ 0.10.9
- **OpenCV:** ≥ 4.9.0

All pinned for Python 3.11 compatibility on Windows.

---

# Contributing

## Updating Baselines

After improving the model:

```powershell
.\scripts\run_eval.ps1
git add unified/eval/baseline.json unified/eval/origin_baseline.json
git commit -m "chore: update baselines after [reason]"
```

## Adding New Labels

1. Add CSV data to appropriate folder (ASL or FSL path segments)
2. Rebuild dataset: `python unified/data/build_unified_dataset.py`
3. Retrain: `.\scripts\run_unified.ps1`
4. Re-evaluate and update baselines

---

# License & Acknowledgments

**Expressora** - Sign language recognition for Filipino and American Sign Language.

Built for deployment on Android with TFLite.

