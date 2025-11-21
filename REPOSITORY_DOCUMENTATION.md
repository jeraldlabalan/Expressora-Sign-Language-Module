# Expressora Sign Language Module - Complete Repository Documentation

**Last Updated:** Generated automatically  
**Repository:** Expressora-Sign-Language-Module  
**Purpose:** Comprehensive documentation of all directories, files, and components in the repository

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Directory Structure](#directory-structure)
3. [Core Components](#core-components)
4. [Data Collection](#data-collection)
5. [Training Pipeline](#training-pipeline)
6. [Model Artifacts](#model-artifacts)
7. [Tools and Utilities](#tools-and-utilities)
8. [Scripts](#scripts)
9. [Documentation Files](#documentation-files)
10. [Jupyter Notebooks](#jupyter-notebooks)

---

## Repository Overview

This repository contains a comprehensive sign language recognition system that supports both **American Sign Language (ASL)** and **Filipino Sign Language (FSL)**. The system uses MediaPipe for hand landmark extraction and TensorFlow/Keras for model training, with TensorFlow Lite (TFLite) models optimized for mobile deployment.

### Key Features:
- **Unified Model**: Single model supporting 197 sign language classes (ASL + FSL)
- **Hand Landmark Recognition**: Uses MediaPipe Hands for 3D landmark extraction
- **Facial Recognition Support**: Separate models for facial expressions and non-manual grammar
- **Multi-format Models**: Keras, TFLite (FP32, FP16, INT8) for different deployment scenarios
- **Live Inference**: Real-time webcam-based sign language recognition
- **Comprehensive Evaluation**: Metrics, confusion matrices, and baseline tracking

---

## Directory Structure

### Root Level Files

- **`ASL-Alphabets.ipynb`**: Jupyter notebook for ASL alphabet data collection
- **`ASL-face_recognition_basic_phrases.ipynb`**: Jupyter notebook for ASL facial recognition data collection
- **`ASL-hands_basic_phrases.ipynb`**: Jupyter notebook for ASL hand gesture data collection
- **`FSL-Alphabets.ipynb`**: Jupyter notebook for FSL alphabet data collection
- **`FSL-face_recognition_basic_phrases.ipynb`**: Jupyter notebook for FSL facial recognition data collection
- **`FSL-hands_basic_phrases.ipynb`**: Jupyter notebook for FSL hand gesture data collection
- **`training_output.log`**: Training log file (generated during training)

---

## Core Components

### 1. `unified/` - Main Unified Model System

The core of the repository, containing the unified sign language recognition pipeline.

#### `unified/data/` - Dataset Management

**Scripts:**
- **`build_unified_dataset.py`**: 
  - Scans all CSV files in the repository
  - Extracts MediaPipe hand landmarks (63-dim for one hand, 126-dim for two hands)
  - Pads one-hand data to 126 dimensions with zeros
  - Infers origin (ASL/FSL) from file paths
  - Normalizes labels (lowercase, underscore-separated)
  - Outputs: `unified_X.npy`, `unified_y.npy`, `unified_origin.npy`, `labels.json`, `origin_labels.json`, `label_origin_stats.json`

- **`build_representative_set.py`**: 
  - Creates a representative dataset for INT8 quantization
  - Generates stratified samples for calibration
  - Outputs: `rep_set.npy`

**Data Files:**
- **`unified_X.npy`**: Feature matrix (N samples × 126 features)
- **`unified_y.npy`**: Label indices (0 to num_classes-1)
- **`unified_origin.npy`**: Origin indices (0=ASL, 1=FSL)
- **`feature_mean.npy`**: Feature mean for normalization
- **`feature_std.npy`**: Feature standard deviation for normalization
- **`rep_set.npy`**: Representative dataset for quantization
- **`labels.json`**: List of all class labels in order
- **`origin_labels.json`**: `["ASL", "FSL"]`
- **`label_origin_stats.json`**: Per-label origin distribution statistics

#### `unified/training/` - Model Training

**Scripts:**
- **`train_unified_tf.py`**: 
  - Main training script with comprehensive fixes to prevent model collapse
  - Features:
    - Fixed class weights (balanced class weighting)
    - Stratified train/validation/test splits
    - Manual feature scaling (mean/std normalization)
    - Enhanced architecture (BatchNormalization, L2 regularization, Dropout)
    - Custom callbacks (PredictionDiversityCallback, PerClassAccuracyCallback)
    - Early stopping, learning rate reduction, model checkpointing
  - Architecture: Dense(512) → BN → Dropout(0.4) → Dense(256) → BN → Dropout(0.3) → Dense(128) → Dropout(0.2) → Dense(num_classes, softmax)
  - Outputs: `expressora_unified.keras`, `savedmodel/`, `training_log.json`, `class_weights.json`, `diversity_history.json`

- **`train_multitask.py`**: 
  - Multi-task training script (gloss + origin classification)
  - Currently available but not the primary training method

#### `unified/export/` - Model Export

**Scripts:**
- **`export_unified_tflite.py`**: 
  - Converts Keras SavedModel to TFLite formats
  - Generates: FP32 (baseline), FP16 (50% size), INT8 (25% size)
  - Creates model signature JSON
  - Outputs: `expressora_unified.tflite`, `expressora_unified_fp16.tflite`, `expressora_unified_int8.tflite`, `model_signature.json`

- **`export_quant_fp16.py`**: FP16 quantization export (standalone)

- **`export_quant_int8.py`**: INT8 quantization export (standalone)

- **`write_model_card.py`**: 
  - Generates comprehensive model metadata
  - Includes: Git SHA, dataset info, training metrics, quantization sizes, thresholds
  - Outputs: `model_card.json`

#### `unified/eval/` - Model Evaluation

**Scripts:**
- **`eval_unified.py`**: 
  - Evaluates gloss classification model
  - Calculates: accuracy, precision, recall, F1-score (per-class and macro-averaged)
  - Generates confusion matrix visualization
  - Creates baseline JSON for CI regression testing
  - Outputs: `confusion_matrix.png`, `baseline.json`

- **`eval_origin.py`**: 
  - Evaluates origin classification (ASL/FSL) if multi-task model exists
  - Outputs: `origin_confusion.png`, `origin_baseline.json`

**Output Files:**
- **`baseline.json`**: CI regression baseline metrics
- **`confusion_matrix.png`**: Visual confusion matrix

#### `unified/inference/` - Model Inference

**Scripts:**
- **`quick_test_tflite.py`**: 
  - Quick smoke test for TFLite models
  - Tests with random input
  - Prints top-1 prediction

#### `unified/live/` - Live Inference

**Scripts:**
- **`live_cam_unified.py`**: 
  - Real-time webcam inference using MediaPipe Hands
  - Features:
    - Hand landmark extraction (handles 1 or 2 hands)
    - Feature scaling (mean/std normalization)
    - TFLite model inference
    - Prediction smoothing (logit averaging over 5 frames)
    - Debouncing (requires 3 consistent frames)
    - Confidence thresholding (default 0.65)
    - Alphabet mode (accumulates letters into words)
    - Origin badge display (ASL/FSL indicator)
  - Environment variables: `CONF_THRESHOLD`, `HOLD_FRAMES`, `ALPHABET_MODE`, `SHOW_ORIGIN`, `ORIGIN_CONF_THRESHOLD`

#### `unified/bridge/` - Label Mapping

**Scripts:**
- **`apply_label_map.py`**: 
  - Maps model labels to translation module gloss tokens
  - Converts underscores to spaces, applies overrides
  - Outputs: `expressora_labels_mapped.json`

**Configuration:**
- **`concept_key_map.json`**: Label-to-gloss mapping rules

#### `unified/models/` - Generated Model Artifacts

**Model Files:**
- **`expressora_unified.keras`**: Keras model file
- **`expressora_unified.tflite`**: FP32 TFLite model (baseline)
- **`expressora_unified_fp16.tflite`**: FP16 quantized TFLite (~50% size)
- **`expressora_unified_int8.tflite`**: INT8 quantized TFLite (~25% size)
- **`savedmodel/`**: TensorFlow SavedModel directory
  - `saved_model.pb`: Model graph
  - `variables/`: Model weights
  - `fingerprint.pb`: Model fingerprint

**Metadata Files:**
- **`expressora_labels.json`**: Class labels list
- **`expressora_labels_mapped.json`**: Mapped labels for translation
- **`model_card.json`**: Comprehensive model metadata
- **`model_signature.json`**: Model input/output signatures
- **`training_log.json`**: Training history (loss, accuracy per epoch)
- **`class_weights.json`**: Class weights used during training
- **`diversity_history.json`**: Prediction diversity metrics during training

#### `unified/` - Configuration Files

- **`requirements.txt`**: Core dependencies (numpy, pandas, scikit-learn, tensorflow)
- **`requirements-eval.txt`**: Evaluation dependencies (matplotlib, scikit-learn)
- **`requirements-live.txt`**: Live inference dependencies (opencv-python, mediapipe)
- **`README_UNIFIED.md`**: Comprehensive unified model documentation

---

## Data Collection

### 2. `AlphabetSignLanguages/` - Alphabet Data

Contains alphabet sign language data for both ASL and FSL.

#### `AlphabetSignLanguages/ASL_/`
- **`Alphabetical_hand_sign_data.csv`**: Combined ASL alphabet data
- **`hand_sign_data_A.csv` through `hand_sign_data_Z.csv`**: Individual letter data files (26 files)

#### `AlphabetSignLanguages/FSL/`
- **`Alphabetical_hand_sign_data.csv`**: Combined FSL alphabet data
- **`hand_sign_data_A.csv` through `hand_sign_data_Z.csv`**: Individual letter data files (26 files)

### 3. `ASLBasicPhrasesSignLanguages/` - ASL Basic Phrases

Contains ASL basic phrase data organized by category.

**Categories:**
- **`Actions/`**: play, read, run, sleep, walk, write
  - `Actions_classes.npy`: Class labels
  - `combined_basic_phrases_dataset.csv`: Combined dataset
- **`Colors/`**: black, blue, green, orange, purple, red, white, yellow
- **`CommonVerbs/`**: bad, come, drink, eat, go, good, help, iloveyou, like, need, no, stop, want, yes
- **`Emotions/`**: angry, exicited, happy, sad, scared, tired
- **`Greetings_/`**: goodbye, hello, hey, nice, please, sorry, thankyou, whatsup
- **`Numbers/`**: one through ten
- **`ObjectThings/`**: book, car, food, house, phone, thing, water
- **`People/`**: family, father, friend, he, me, mother, she, they, we, you
- **`Places/`**: farm, home, school, store, work
- **`Questions/`**: how, what, when, where, who, why
- **`Time/`**: afternoon, morning, night, today, tommorow, yesterday

Each category contains:
- Individual CSV files for each phrase
- `*_classes.npy`: NumPy array of class labels
- `combined_basic_phrases_dataset.csv`: Combined dataset for the category

### 4. `FSLBasicPhrasesSignLanguages/` - FSL Basic Phrases

Contains FSL basic phrase data organized by category (Filipino translations).

**Categories:**
- **`Actions/`**: basahin, lakad, laro, magsulat, tulog, tumakbo
- **`Colors/`**: asul, berde, dilaw, itim, kahel, lila, pula, puti
- **`CommonVerbs/`**: gusto, halika, hindi, inumin, kailangan, katulad, kumain, mabuti, mahal kita, masama, oo, pumunta, tigil, tulong
- **`Emotions/`**: galit, malungkot, masaya, nasasabik, pagod, takot
- **`Greetings/`**: anong balita, hello, hoy, nays, paalam, pakiusap, patawad, salamat
- **`Numbers/`**: isa, dalawa, tatlo, apat, lima, anim, pito, walo, siyam, sampu
- **`ObjectThings/`**: bahay, gamit, kotse, libro, pagkain, telopono, tubig
- **`People/`**: ako, ama, ikaw, ina, kaibigan, pamilya, sila, siya(babae), siya(lalake), tayo
- **`Places/`**: bukid, paaralan, tahanan, tindahan, trabaho
- **`Questions/`**: ano, bakit, kailan, paano, saan, sino
- **`Time/`**: bukas, gabi, hapon, kahapon, ngayon, umaga

Each category contains:
- Individual CSV files for each phrase
- `FSL*_classes.npy`: NumPy array of class labels
- `fsl_combined_basic_phrases_dataset.csv`: Combined dataset for the category

### 5. `ASLFacialExpressionsLanguages/` - ASL Facial Recognition Data

Contains ASL facial expression and non-manual grammar data.

**Categories:**
- **`BackChannelFeedback/`**: confused, listening, nod, smile
- **`Expressions/`**: angry, confused, disappointed, fear, funny, happy, sad, shame, shock, shy, smile, surprise
- **`EyeGaze/`**: down_focus, object_left, object_right, partner, switch_role
- **`MouthMorphemes/`**: 
  - **`AAH/`**: direction_left, direction_right, distance_hold, far, short, very_far
  - **`CHA/`**: big, brief, huge, tall, wide
  - **`CS/`**: close, effort, quick, strained, sustained, very_close
  - **`MM/`**: loose, normal, quick, repeated, tight
  - **`OO/`**: fast, slow, small, sustained, thin, tiny
- **`NonManualGrammar/Questions/`**: condition, how, negative, no, statement, what, where, who, why, yes

### 6. `FSLFacialExpressionsLanguages/` - FSL Facial Recognition Data

Contains FSL facial expression data.

**Categories:**
- **`Expressions/`**: dismayado, galit, kahihiyan, malungkot, masaya, nabigla, nahihiya, nakakatawa, nalilito, ngiti, sorpresa, takot

---

## Training Pipeline

### 7. `Tensorflow/` - TensorFlow Models and Notebooks

Contains TensorFlow-specific training notebooks and pre-trained models.

#### Notebooks:
- **`ASL-TF-Alphabets.ipynb`**: ASL alphabet training with TensorFlow
- **`ASL-TF-hands_basic_phrases.ipynb`**: ASL basic phrases training
- **`FSL-TF-Alphabets.ipynb`**: FSL alphabet training with TensorFlow
- **`FSL-TF-hands_basic_phrases.ipynb`**: FSL basic phrases training

#### Model Directories:

**`Tensorflow/TFModels/`** - ASL TensorFlow Models:
- **`ASL_Alphabet_TF_Model_SavedModel/`**: ASL alphabet model
- **`ActionModels/`**: ASL actions model
- **`ColorsModels/`**: ASL colors model
- **`CommonVerbsModels/`**: ASL common verbs model
- **`EmotionsModels/`**: ASL emotions model
- **`GreetingsModels/`**: ASL greetings model
- **`NumbersModels/`**: ASL numbers model
- **`ObjectThingsModels/`**: ASL object/things model
- **`PeopleModels/`**: ASL people model
- **`PlacesModels/`**: ASL places model
- **`QuestionsModels/`**: ASL questions model
- **`TimeModels/`**: ASL time model

Each model directory contains:
- `*.keras`: Keras model file
- `*.tflite`: TFLite model file
- `*_SavedModel/`: TensorFlow SavedModel directory

**`Tensorflow/TFModelsFSL/`** - FSL TensorFlow Models:
- Similar structure to `TFModels/` but for FSL
- **`FSL_Alphabet_TF_Model_SavedModel/`**: FSL alphabet model
- **`FSLActionsModels/`**, **`FSLColorsModels/`**, etc.: FSL category models

**`Tensorflow/ASL-FacialwithHands-Recognition-TFModels/`** - ASL Facial + Hand Models:
- **`BackChannelFeedBackModels/`**: Back channel feedback model
- **`ExpressionModels/`**: Facial expressions model
- **`EyeGazeModels/`**: Eye gaze model
- **`MouthMorphemes/`**: 
  - **`AAH_Models/`**, **`CHA_Models/`**, **`CS_Models/`**, **`MM_Models/`**, **`OO_Models/`**: Mouth morpheme models
- **`NonManualGrammarModels/`**: Non-manual grammar model

Each model directory contains:
- `*_model_tf.keras`: Keras model
- `*_model_tf.tflite`: TFLite model
- `*_model_tf_labels.npy`: Class labels
- `*_model_tf_scaler.pkl`: Feature scaler (pickle)

**`Tensorflow/FSL-FacialwithHands-Recognition-TFModels/`** - FSL Facial + Hand Models:
- **`ExpressionModels/`**: FSL facial expressions model
- Similar structure to ASL facial models

---

## Model Artifacts

### 8. `Models/` - Pre-trained Pickle Models

Contains pre-trained models in pickle format (legacy format, before unified model).

**ASL Models:**
- **`ASL-Alphabets_Model.pkl`**: ASL alphabet classifier
- **`ASLBasicPhrasesModels/`**: 
  - `Actions_model.pkl`, `Colors_model.pkl`, `Common_Verbs_model.pkl`, `Emotions_model.pkl`, `Greetings_model.pkl`, `Numbers_model.pkl`, `ObjectThings_model.pkl`, `People_model.pkl`, `Places_model.pkl`, `Questions_model.pkl`, `Time_model.pkl`

**FSL Models:**
- **`FSL-Alphabets_Model.pkl`**: FSL alphabet classifier
- **`FSLBasicPhrasesModels/`**: 
  - Same structure as ASL models but for FSL

---

## Tools and Utilities

### 9. `tools/` - Utility Scripts and Reports

**Diagnostic Scripts:**
- **`diagnose_models.py`**: 
  - Tests model responsiveness with random inputs
  - Verifies input tensor updates
  - Calculates output entropy
  - Detects static outputs (model collapse)
  - Outputs: `diagnostic_results.json`, `DIAGNOSTIC_REPORT.md`

- **`test_input_sensitivity.py`**: 
  - Measures output variance across different inputs
  - Assesses model sensitivity to input changes
  - Outputs: `sensitivity_results.json`

- **`test_real_data.py`**: 
  - Tests model on actual dataset samples
  - Provides realistic performance assessment
  - Outputs: Performance metrics

- **`test_unified_models.py`**: 
  - Tests all unified TFLite models (FP32, FP16, INT8)
  - Compares performance across quantization formats
  - Outputs: `unified_models_performance.json`, `UNIFIED_MODELS_PERFORMANCE_REPORT.md`

- **`compare_keras_tflite.py`**: 
  - Compares Keras and TFLite model outputs
  - Validates TFLite conversion accuracy
  - Outputs: `keras_tflite_comparison.json`

**Analysis Scripts:**
- **`analyze_class_distribution.py`**: 
  - Analyzes class distribution in dataset
  - Identifies class imbalance
  - Outputs: `class_distribution.json`

- **`monitor_training.py`**: 
  - Custom Keras callbacks for training monitoring
  - `PredictionDiversityCallback`: Monitors prediction diversity to detect collapse
  - `PerClassAccuracyCallback`: Tracks per-class accuracy during training
  - Used by `train_unified_tf.py`

- **`monitor_training_progress.py`**: 
  - Alternative training progress monitoring script

**Utility Scripts:**
- **`check_python.py`**: 
  - Verifies Python version and dependencies
  - Used by PowerShell scripts

**Documentation Files:**
- **`TRAINING_FIXES_REQUIRED.md`**: Documents critical bugs and fixes needed
- **`TRAINING_IMPLEMENTATION_SUMMARY.md`**: Summary of implemented fixes
- **`FINAL_TRAINING_SUMMARY.md`**: Final training success report
- **`TRAINING_SUCCESS_REPORT.md`**: Training success metrics
- **`NEW_MODEL_PERFORMANCE_ANALYSIS.md`**: Performance analysis of new models
- **`UNIFIED_MODELS_PERFORMANCE_REPORT.md`**: Unified models performance report
- **`DIAGNOSTIC_REPORT.md`**: Model diagnostic results
- **`ENCODING_ERROR_EXPLANATION.md`**: Explanation of encoding errors

**Data Files:**
- **`class_distribution.json`**: Class distribution statistics
- **`diagnostic_results.json`**: Model diagnostic results
- **`sensitivity_results.json`**: Input sensitivity test results
- **`unified_models_performance.json`**: Unified models performance metrics
- **`keras_tflite_comparison.json`**: Keras vs TFLite comparison results

---

## Scripts

### 10. `scripts/` - PowerShell Automation Scripts

**Setup:**
- **`setup_venv.ps1`**: 
  - One-time virtual environment setup
  - Finds Python 3.10/3.11
  - Creates `.venv` directory
  - Installs `unified/requirements.txt`
  - Handles Windows-specific issues

**Pipeline Scripts:**
- **`run_unified.ps1`**: 
  - Complete unified pipeline execution
  - Steps:
    1. Checks Python version
    2. Builds unified dataset
    3. Trains model
    4. Exports TFLite models
    5. Quick tests TFLite
    6. Generates label mapping (non-fatal)
    7. Generates model card (non-fatal)
  - Outputs artifacts to `unified/models/`

- **`run_eval.ps1`**: 
  - Runs model evaluation pipeline
  - Executes `eval_unified.py`
  - Generates confusion matrix and baseline

- **`run_quantize.ps1`**: 
  - Quantization pipeline
  - Builds representative set
  - Exports FP16 and INT8 models
  - Prints size comparison

- **`run_live.ps1`**: 
  - Launches live webcam inference
  - Supports parameters:
    - `-Threshold`: Confidence threshold
    - `-AlphabetMode`: Enable alphabet accumulation
    - `-ShowOrigin`: Show origin badge

---

## Documentation Files

### 11. `gemini_review/` - Gemini Pro Review Package

Self-contained package for external review of the training pipeline.

**Structure:**
- **`README.md`**: Main documentation for Gemini Pro
- **`REVIEW_INSTRUCTIONS.md`**: Instructions for the review
- **`code/`**: 
  - `train_unified_tf.py`: Main training script
  - `build_unified_dataset.py`: Dataset builder
  - `export_unified_tflite.py`: TFLite export
  - `monitor_training.py`: Training callbacks
  - `build_representative_set.py`: Quantization dataset
  - `eval_unified.py`: Evaluation script
  - `test_real_data.py`: Real data testing
  - `test_unified_models.py`: Model testing
  - `requirements.txt`: Dependencies
- **`docs/`**: 
  - `TRAINING_FIXES_REQUIRED.md`: Issues and fixes
  - `TRAINING_IMPLEMENTATION_SUMMARY.md`: Implementation summary

---

## Jupyter Notebooks

### Data Collection Notebooks

**Alphabet Collection:**
- **`ASL-Alphabets.ipynb`**: 
  - ASL alphabet data collection
  - Uses MediaPipe Hands
  - Single hand detection
  - Saves to `AlphabetSignLanguages/ASL_/`

- **`FSL-Alphabets.ipynb`**: 
  - FSL alphabet data collection
  - Similar structure to ASL
  - Saves to `AlphabetSignLanguages/FSL/`

**Hand Gesture Collection:**
- **`ASL-hands_basic_phrases.ipynb`**: 
  - ASL basic phrases hand gesture collection
  - Saves to `ASLBasicPhrasesSignLanguages/`

- **`FSL-hands_basic_phrases.ipynb`**: 
  - FSL basic phrases hand gesture collection
  - Saves to `FSLBasicPhrasesSignLanguages/`

**Facial Recognition Collection:**
- **`ASL-face_recognition_basic_phrases.ipynb`**: 
  - ASL facial recognition data collection
  - Combines hand landmarks (80%) and face landmarks (20%)
  - Uses MediaPipe Hands and Face Mesh
  - Saves to `ASLFacialExpressionsLanguages/`

- **`FSL-face_recognition_basic_phrases.ipynb`**: 
  - FSL facial recognition data collection
  - Similar structure to ASL
  - Saves to `FSLFacialExpressionsLanguages/`

---

## Data Format

### CSV Format

All CSV files contain MediaPipe hand landmark data with the following structure:

**One Hand (63 features):**
- Columns: `L1_x0`, `L1_y0`, `L1_z0`, `L1_x1`, `L1_y1`, `L1_z1`, ..., `L1_x20`, `L1_y20`, `L1_z20`, `label`
- 21 landmarks × 3 coordinates (x, y, z) = 63 features

**Two Hands (126 features):**
- Columns: `L1_x0`, ..., `L1_z20`, `L2_x0`, ..., `L2_z20`, `label`
- 2 hands × 21 landmarks × 3 coordinates = 126 features

**Facial Recognition (additional features):**
- Hand landmarks (80% weight) + selected face landmarks (20% weight)
- Face landmarks use selected indices from MediaPipe Face Mesh

### NumPy Format

- **`.npy` files**: NumPy arrays
  - Feature arrays: `[N, 126]` or `[N, 63]`
  - Label arrays: `[N]` (integer indices)
  - Class arrays: `[num_classes]` (string labels)

### JSON Format

- **`labels.json`**: List of strings `["label1", "label2", ...]`
- **`origin_labels.json`**: `["ASL", "FSL"]`
- **`label_origin_stats.json`**: Dictionary mapping labels to origin counts
- **`model_card.json`**: Comprehensive model metadata
- **`training_log.json`**: Training history with epoch-by-epoch metrics

---

## Model Architecture

### Unified Model

**Input:** 126-dimensional feature vector (hand landmarks)
- One hand: 63 features + 63 zeros (padded)
- Two hands: 63 features (left) + 63 features (right)

**Architecture:**
```
Input(126) 
→ Dense(512, ReLU, L2=1e-4) 
→ BatchNormalization 
→ Dropout(0.4) 
→ Dense(256, ReLU, L2=1e-4) 
→ BatchNormalization 
→ Dropout(0.3) 
→ Dense(128, ReLU, L2=1e-4) 
→ Dropout(0.2) 
→ Dense(197, Softmax)
```

**Output:** 197-dimensional probability vector (one per class)

**Training Configuration:**
- Optimizer: Adam (learning_rate=3e-4)
- Loss: SparseCategoricalCrossentropy
- Class weights: Balanced (sklearn)
- Validation split: 10% (stratified)
- Test split: 10% (stratified)
- Early stopping: Patience=15, monitor='val_loss'
- Learning rate reduction: Patience=7, factor=0.5
- Batch size: 32
- Epochs: 100 (with early stopping)

---

## Dependencies

### Core Dependencies (`unified/requirements.txt`):
- `numpy==1.26.4`
- `pandas==2.1.4`
- `scikit-learn==1.3.2`
- `ml-dtypes==0.3.2`
- `tensorflow==2.15.1`

### Evaluation Dependencies (`unified/requirements-eval.txt`):
- `matplotlib==3.8.3` (for plotting)

### Live Inference Dependencies (`unified/requirements-live.txt`):
- `opencv-python>=4.9.0`
- `mediapipe>=0.10.9`

### Python Version:
- **Primary:** Python 3.11
- **Secondary:** Python 3.10

---

## Workflow

### Complete Training Workflow:

1. **Data Collection** (Jupyter Notebooks):
   - Collect hand landmarks using MediaPipe
   - Save to CSV files in appropriate directories

2. **Dataset Building**:
   ```powershell
   python unified/data/build_unified_dataset.py
   ```
   - Scans all CSV files
   - Creates unified dataset
   - Generates feature scaling parameters

3. **Training**:
   ```powershell
   python unified/training/train_unified_tf.py
   ```
   - Trains unified model
   - Saves Keras model and training history

4. **Export**:
   ```powershell
   python unified/export/export_unified_tflite.py
   ```
   - Converts to TFLite formats
   - Generates FP32, FP16, INT8 models

5. **Evaluation**:
   ```powershell
   python unified/eval/eval_unified.py
   ```
   - Calculates metrics
   - Generates confusion matrix

6. **Live Testing**:
   ```powershell
   python unified/live/live_cam_unified.py
   ```
   - Real-time webcam inference

### Quick Start (PowerShell):

```powershell
# One-time setup
.\scripts\setup_venv.ps1

# Complete pipeline
.\scripts\run_unified.ps1

# Evaluation
.\scripts\run_eval.ps1

# Quantization
.\scripts\run_quantize.ps1

# Live inference
.\scripts\run_live.ps1
```

---

## Key Features

### 1. Unified Model
- Single model for 197 ASL + FSL classes
- Handles both one-hand and two-hand signs
- Zero-padding for one-hand inputs

### 2. Origin Tracking
- Automatically infers ASL/FSL origin from file paths
- Tracks origin distribution per label
- Supports multi-task training (future)

### 3. Feature Scaling
- Manual mean/std normalization
- Saved scaling parameters for inference
- Required for Android app integration

### 4. Model Collapse Prevention
- Stratified data splits
- Balanced class weights
- Prediction diversity monitoring
- Per-class accuracy tracking

### 5. Quantization Support
- FP32 (baseline, ~2.15 MB)
- FP16 (mobile GPU, ~1.08 MB, -50%)
- INT8 (CPU inference, ~0.58 MB, -73%)

### 6. Live Inference Features
- Prediction smoothing
- Debouncing
- Confidence thresholding
- Alphabet accumulation mode
- Origin badge display

---

## File Count Summary

### Directories:
- **Root level:** 12 main directories
- **unified/:** 8 subdirectories
- **Data directories:** 6 main data directories
- **Tensorflow/:** 3 model directories
- **Models/:** 2 model directories

### Files:
- **Python scripts:** ~30+ scripts
- **Jupyter notebooks:** 6 notebooks
- **PowerShell scripts:** 5 scripts
- **CSV data files:** 100+ files
- **Model files:** 50+ model files
- **Documentation:** 15+ markdown files
- **JSON configs:** 10+ JSON files

---

## Notes

1. **Virtual Environment**: All scripts assume a virtual environment (`.venv`) is active or will be created.

2. **Windows Focus**: Scripts are primarily designed for Windows PowerShell, though Python code is cross-platform.

3. **Git LFS**: Large model files may be stored in Git LFS.

4. **Feature Scaling**: The Android app must apply feature scaling: `(features - mean) / std` using `feature_mean.npy` and `feature_std.npy`.

5. **One-Hand vs Two-Hand**: The model expects 126-dimensional input. One-hand inputs are zero-padded to 126 dimensions.

6. **Facial Recognition**: Facial recognition models exist but are not integrated into the unified pipeline yet.

7. **Model Formats**: 
   - Legacy: Pickle models (`.pkl`) in `Models/`
   - Current: Keras (`.keras`) and TFLite (`.tflite`) in `unified/models/` and `Tensorflow/`

---

## Contact and Support

For questions or issues, refer to:
- `unified/README_UNIFIED.md`: Comprehensive unified model documentation
- `tools/` documentation files: Training fixes and implementation summaries
- `gemini_review/README.md`: Training pipeline review documentation

---

**End of Documentation**

