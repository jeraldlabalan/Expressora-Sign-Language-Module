# Kaggle Training Readiness Checklist

## ✅ VERIFIED COMPONENTS

### 1. Dependencies (requirements_kaggle.txt)
- ✅ `numpy==1.26.4`
- ✅ `pandas==2.1.4`
- ✅ `scikit-learn==1.3.2` (covers sklearn imports)
- ✅ `tensorflow==2.15.1`
- ✅ `mediapipe`

**All imports covered:**
- `numpy`, `pandas`, `json`, `pathlib` (standard library)
- `tensorflow` (TF 2.15.1)
- `sklearn.utils.class_weight`, `sklearn.model_selection.train_test_split` (scikit-learn)
- `collections.Counter` (standard library)

### 2. File Paths
- ✅ All scripts use relative paths via `Path(__file__).resolve().parents[X]`
- ✅ No hardcoded Windows paths (`C:\`, backslashes, etc.)
- ✅ Paths are cross-platform compatible (Pathlib)

### 3. Required Directory Structure
The repository structure must be preserved:
```
Expressora-Sign-Language-Module/
├── unified/
│   ├── data/
│   │   └── build_unified_dataset.py
│   ├── training/
│   │   └── train_unified_tf.py
│   ├── export/
│   │   └── export_unified_tflite.py
│   └── models/ (created automatically)
├── tools/
│   └── monitor_training.py (REQUIRED for training)
├── AlphabetSignLanguages/ (REQUIRED - contains CSV data)
├── ASLBasicPhrasesSignLanguages/ (REQUIRED - contains CSV data)
├── FSLBasicPhrasesSignLanguages/ (REQUIRED - contains CSV data)
├── ASLFacialExpressionsLanguages/ (REQUIRED - contains CSV data)
├── FSLFacialExpressionsLanguages/ (REQUIRED - contains CSV data)
├── Tensorflow/ (REQUIRED - contains CSV data)
└── requirements_kaggle.txt
```

### 4. Script Dependencies
- ✅ `build_unified_dataset.py` → Standalone (no external deps)
- ✅ `train_unified_tf.py` → Imports `tools/monitor_training.py` (must be in zip)
- ✅ `export_unified_tflite.py` → Standalone (no external deps)

## ⚠️ CRITICAL REQUIREMENTS FOR KAGGLE

### 1. Data Files Must Be Included
The `build_unified_dataset.py` script searches for CSV files in:
- `AlphabetSignLanguages/**/*.csv`
- `ASLBasicPhrasesSignLanguages/**/*.csv`
- `FSLBasicPhrasesSignLanguages/**/*.csv`
- `ASLFacialExpressionsLanguages/**/*.csv`
- `FSLFacialExpressionsLanguages/**/*.csv`
- `Tensorflow/**/*.csv`

**ACTION REQUIRED:** Ensure ALL CSV files in these directories are included in the zip.

### 2. Tools Directory Must Be Included
The training script requires:
- `tools/monitor_training.py` (contains `PredictionDiversityCallback` and `PerClassAccuracyCallback`)

**ACTION REQUIRED:** Ensure `tools/` directory with `monitor_training.py` is in the zip.

### 3. Execution Order on Kaggle
```python
# Step 1: Build dataset (scans CSV files)
python unified/data/build_unified_dataset.py
# Outputs: unified/data/unified_X.npy, unified_y.npy, labels.json, etc.

# Step 2: Train model
python unified/training/train_unified_tf.py
# Outputs: unified/models/expressora_unified.keras, savedmodel/, etc.

# Step 3: Export TFLite (optional)
python unified/export/export_unified_tflite.py
# Outputs: unified/models/expressora_unified.tflite
```

## 🔍 PRE-ZIP VERIFICATION CHECKLIST

Before zipping, verify:

- [ ] All CSV files in data directories are included
- [ ] `tools/monitor_training.py` exists and is included
- [ ] `requirements_kaggle.txt` is in root directory
- [ ] No `.gitignore` excluding critical files
- [ ] No large model files (`.keras`, `.tflite`) need to be excluded (they'll be generated)
- [ ] No `__pycache__` or `.pyc` files (optional, but cleaner)

## 📝 KAGGLE NOTEBOOK SETUP

When creating a Kaggle notebook, use this setup:

```python
# Install dependencies
!pip install -r requirements_kaggle.txt

# Run pipeline
!python unified/data/build_unified_dataset.py
!python unified/training/train_unified_tf.py
!python unified/export/export_unified_tflite.py
```

## 🚨 POTENTIAL ISSUES & SOLUTIONS

### Issue 1: Missing CSV Files
**Symptom:** `build_unified_dataset.py` prints "No usable CSVs found"
**Solution:** Verify all data directories are included in zip

### Issue 2: Missing tools/monitor_training.py
**Symptom:** `ImportError: cannot import name 'PredictionDiversityCallback'`
**Solution:** Ensure `tools/` directory is in zip

### Issue 3: Path Resolution Issues
**Symptom:** `FileNotFoundError` when loading data
**Solution:** Run scripts from repository root, or adjust working directory

### Issue 4: Memory Issues
**Symptom:** Out of memory during training
**Solution:** Reduce batch size in `train_unified_tf.py` (line 166: `BATCH_SIZE = 64`)

## ✅ FINAL VERIFICATION

Run these commands locally before zipping:
```bash
# Test dataset building
python unified/data/build_unified_dataset.py

# Test training (with small subset if needed)
python unified/training/train_unified_tf.py

# Test export
python unified/export/export_unified_tflite.py
```

If all three run successfully, the repository is ready for Kaggle!

