"""
Evaluate the unified gloss classification model.
Generates metrics, confusion matrix, and baseline for CI regression testing.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
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
    """Load test data and labels"""
    X = np.load(DATA_DIR / "unified_X.npy")
    y = np.load(DATA_DIR / "unified_y.npy")
    
    with open(DATA_DIR / "labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    
    # Use last 10% as test set (matching training split)
    n = len(y)
    split = int(n * 0.9)
    X_test = X[split:]
    y_test = y[split:]
    
    print(f"Loaded test set: {len(y_test)} samples, {len(labels)} classes")
    return X_test, y_test, labels

def load_model():
    """Load Keras model"""
    model_path = MODELS_DIR / "expressora_unified.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run training first: .\\scripts\\run_unified.ps1"
        )
    
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

def plot_confusion_matrix(cm, labels, output_path):
    """Generate and save confusion matrix visualization"""
    # For large number of classes, make figure larger
    n_classes = len(labels)
    figsize = (max(12, n_classes * 0.3), max(10, n_classes * 0.3))
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - Gloss Classification')
    plt.colorbar()
    
    # Show labels (but sparse if too many classes)
    if n_classes <= 50:
        tick_marks = np.arange(n_classes)
        plt.xticks(tick_marks, labels, rotation=90, fontsize=6)
        plt.yticks(tick_marks, labels, fontsize=6)
    else:
        # Show every Nth label
        step = max(1, n_classes // 50)
        tick_marks = np.arange(0, n_classes, step)
        plt.xticks(tick_marks, [labels[i] for i in tick_marks], rotation=90, fontsize=6)
        plt.yticks(tick_marks, [labels[i] for i in tick_marks], fontsize=6)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")

def main():
    print("="*60)
    print("Evaluating Unified Gloss Classification Model")
    print("="*60)
    print()
    
    # Load data and model
    X_test, y_test, labels = load_data()
    model = load_model()
    
    # Predict
    print("\nRunning inference on test set...")
    y_pred_probs = model.predict(X_test, verbose=0)
    
    # Handle multi-output models (take first output as gloss predictions)
    if isinstance(y_pred_probs, list):
        y_pred_probs = y_pred_probs[0]
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    # Per-class metrics
    print("\nComputing per-class metrics...")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    # Macro-averaged metrics
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print(f"\nMacro-averaged metrics:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-score:  {macro_f1:.4f}")
    
    # Sort classes by F1 score
    class_scores = []
    for i, label in enumerate(labels):
        class_scores.append({
            'label': label,
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        })
    
    class_scores.sort(key=lambda x: x['f1'])
    
    # Show worst 10 classes
    print(f"\n{'='*60}")
    print("Worst 10 Classes (by F1-score):")
    print(f"{'='*60}")
    for item in class_scores[:10]:
        print(f"  {item['label']:20s} - F1: {item['f1']:.4f}, "
              f"Prec: {item['precision']:.4f}, Rec: {item['recall']:.4f}, "
              f"Support: {item['support']}")
    
    # Show best 10 classes
    print(f"\n{'='*60}")
    print("Best 10 Classes (by F1-score):")
    print(f"{'='*60}")
    for item in class_scores[-10:][::-1]:
        print(f"  {item['label']:20s} - F1: {item['f1']:.4f}, "
              f"Prec: {item['precision']:.4f}, Rec: {item['recall']:.4f}, "
              f"Support: {item['support']}")
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, EVAL_DIR / "confusion_matrix.png")
    
    # Save baseline metrics for CI
    baseline = {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'per_class': class_scores
    }
    
    baseline_path = EVAL_DIR / "baseline.json"
    with open(baseline_path, 'w', encoding='utf-8') as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved baseline metrics to {baseline_path}")
    print(f"\n{'='*60}")
    print("Evaluation complete!")
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

