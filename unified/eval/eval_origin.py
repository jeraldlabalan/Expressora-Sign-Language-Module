"""
Evaluate origin classification head of the unified model.
Reports origin accuracy, per-origin metrics, and warns about mono-origin labels.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "unified" / "data"
MODELS_DIR = ROOT / "unified" / "models"
EVAL_DIR = ROOT / "unified" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load test data with origin labels"""
    X = np.load(DATA_DIR / "unified_X.npy")
    y = np.load(DATA_DIR / "unified_y.npy")
    origin = np.load(DATA_DIR / "unified_origin.npy")
    
    with open(DATA_DIR / "labels.json", "r", encoding="utf-8") as f:
        gloss_labels = json.load(f)
    
    with open(DATA_DIR / "origin_labels.json", "r", encoding="utf-8") as f:
        origin_labels = json.load(f)
    
    # Use last 10% as test set (matching training split)
    n = len(y)
    split = int(n * 0.9)
    X_test = X[split:]
    y_test = y[split:]
    origin_test = origin[split:]
    
    print(f"Loaded test set: {len(origin_test)} samples, {len(origin_labels)} origin classes")
    return X_test, y_test, origin_test, gloss_labels, origin_labels

def load_model():
    """Load Keras model (may have single or multi-output)"""
    model_path = MODELS_DIR / "expressora_unified.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run training first: .\\scripts\\run_unified.ps1"
        )
    
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

def check_mono_origin_labels(y_test, origin_test, gloss_labels):
    """Check for labels that only appear in one origin (shortcut learning risk)"""
    label_origins = defaultdict(set)
    
    for gloss_idx, origin_idx in zip(y_test, origin_test):
        label_origins[int(gloss_idx)].add(int(origin_idx))
    
    mono_origin_labels = []
    for gloss_idx, origins in label_origins.items():
        if len(origins) == 1:
            mono_origin_labels.append((gloss_labels[gloss_idx], list(origins)[0]))
    
    return mono_origin_labels

def plot_confusion_matrix(cm, labels, output_path):
    """Generate and save confusion matrix visualization"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - Origin Classification')
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Origin')
    plt.xlabel('Predicted Origin')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")

def main():
    print("="*60)
    print("Evaluating Origin Classification")
    print("="*60)
    print()
    
    # Load data and model
    X_test, y_test, origin_test, gloss_labels, origin_labels = load_data()
    model = load_model()
    
    # Check if model has multi-output (origin head)
    print("\nChecking model outputs...")
    test_pred = model.predict(X_test[:1], verbose=0)
    
    if isinstance(test_pred, list) and len(test_pred) >= 2:
        print(f"✓ Model has {len(test_pred)} outputs - using output #1 for origin")
        use_multi_output = True
    else:
        print("⚠ Model has single output - origin head not found")
        print("  This evaluation requires a multi-task model with origin head.")
        print("  Skipping origin evaluation.")
        return
    
    # Predict
    print("\nRunning inference on test set...")
    predictions = model.predict(X_test, verbose=0)
    
    # Extract origin predictions (second output)
    origin_pred_probs = predictions[1]
    origin_pred = np.argmax(origin_pred_probs, axis=1)
    
    # Overall accuracy
    accuracy = accuracy_score(origin_test, origin_pred)
    print(f"\n{'='*60}")
    print(f"Origin Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    # Per-origin metrics
    print("\nComputing per-origin metrics...")
    precision, recall, f1, support = precision_recall_fscore_support(
        origin_test, origin_pred, average=None, zero_division=0
    )
    
    # Macro-averaged metrics
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print(f"\nMacro-averaged metrics:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-score:  {macro_f1:.4f}")
    
    print(f"\n{'='*60}")
    print("Per-Origin Metrics:")
    print(f"{'='*60}")
    for i, origin_name in enumerate(origin_labels):
        print(f"  {origin_name:10s} - Prec: {precision[i]:.4f}, "
              f"Rec: {recall[i]:.4f}, F1: {f1[i]:.4f}, "
              f"Support: {int(support[i])}")
    
    # Check for mono-origin labels
    print(f"\n{'='*60}")
    print("Checking for Mono-Origin Labels (Shortcut Learning Risk):")
    print(f"{'='*60}")
    mono_origin = check_mono_origin_labels(y_test, origin_test, gloss_labels)
    
    if mono_origin:
        print(f"\n⚠ Found {len(mono_origin)} labels with samples from only one origin:")
        for label, origin_idx in mono_origin[:20]:
            origin_name = origin_labels[origin_idx]
            print(f"  - {label:20s} (only {origin_name})")
        if len(mono_origin) > 20:
            print(f"  ... and {len(mono_origin) - 20} more")
        print("\nWARNING: Model may learn to predict origin from gloss alone (shortcut).")
        print("Consider collecting more data from both origins for these labels.")
    else:
        print("✓ All labels have samples from multiple origins")
    
    # Provenance summary
    print(f"\n{'='*60}")
    print("Provenance Summary:")
    print(f"{'='*60}")
    for i, origin_name in enumerate(origin_labels):
        count = int((origin_test == i).sum())
        pct = 100.0 * count / len(origin_test)
        print(f"  {origin_name}: {count} samples ({pct:.1f}%)")
    
    imbalance_ratio = max(support) / min(support) if min(support) > 0 else float('inf')
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 2.0:
        print("⚠ Significant class imbalance detected")
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(origin_test, origin_pred)
    plot_confusion_matrix(cm, origin_labels, EVAL_DIR / "origin_confusion.png")
    
    # Save baseline metrics for CI
    baseline = {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'per_origin': [
            {
                'origin': origin_labels[i],
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(len(origin_labels))
        ],
        'mono_origin_labels_count': len(mono_origin),
        'imbalance_ratio': float(imbalance_ratio)
    }
    
    baseline_path = EVAL_DIR / "origin_baseline.json"
    with open(baseline_path, 'w', encoding='utf-8') as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved baseline metrics to {baseline_path}")
    print(f"\n{'='*60}")
    print("Origin Evaluation Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}\n")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

