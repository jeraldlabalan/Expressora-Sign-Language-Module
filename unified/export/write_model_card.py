"""
Generate comprehensive model card with training metadata, metrics, and provenance.
"""
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "unified" / "models"
DATA_DIR = ROOT / "unified" / "data"
EVAL_DIR = ROOT / "unified" / "eval"

def get_git_sha():
    """Get current git commit SHA"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"

def compute_sha256(file_path):
    """Compute SHA256 hash of a file"""
    if not file_path.exists():
        return None
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_file_size_mb(file_path):
    """Get file size in MB"""
    if not file_path.exists():
        return None
    return file_path.stat().st_size / (1024 * 1024)

def load_json_safe(file_path):
    """Load JSON file, return None if not found"""
    if not file_path.exists():
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def main():
    print("="*60)
    print("Generating Model Card")
    print("="*60)
    print()
    
    # Basic metadata
    card = {
        "version": "1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "git_sha": get_git_sha()
    }
    
    # Dataset information
    print("Gathering dataset information...")
    dataset_info = {}
    
    X_path = DATA_DIR / "unified_X.npy"
    y_path = DATA_DIR / "unified_y.npy"
    labels_path = DATA_DIR / "labels.json"
    
    if X_path.exists() and y_path.exists():
        X = np.load(X_path)
        y = np.load(y_path)
        dataset_info["num_samples"] = int(len(y))
        dataset_info["feature_dim"] = int(X.shape[1])
    
    if labels_path.exists():
        labels = load_json_safe(labels_path)
        if labels:
            dataset_info["num_classes"] = len(labels)
            dataset_info["labels_sha256"] = compute_sha256(labels_path)
    
    card["dataset"] = dataset_info
    
    # Training information
    print("Gathering training information...")
    training_info = {}
    
    training_log_path = MODELS_DIR / "training_log.json"
    if training_log_path.exists():
        log = load_json_safe(training_log_path)
        if log:
            training_info["final_loss"] = log.get("loss", [])[-1] if log.get("loss") else None
            training_info["final_accuracy"] = log.get("accuracy", [])[-1] if log.get("accuracy") else None
            training_info["final_val_loss"] = log.get("val_loss", [])[-1] if log.get("val_loss") else None
            training_info["final_val_accuracy"] = log.get("val_accuracy", [])[-1] if log.get("val_accuracy") else None
            training_info["num_epochs"] = len(log.get("loss", []))
    
    card["training"] = training_info
    
    # Quantized models
    print("Gathering quantized model sizes...")
    quantized_models = {}
    
    model_files = [
        ("float32", "expressora_unified.tflite"),
        ("fp16", "expressora_unified_fp16.tflite"),
        ("int8", "expressora_unified_int8.tflite")
    ]
    
    for name, filename in model_files:
        path = MODELS_DIR / filename
        size = get_file_size_mb(path)
        if size is not None:
            quantized_models[name] = {
                "filename": filename,
                "size_mb": round(size, 2)
            }
    
    card["quantized_models"] = quantized_models
    
    # Live inference thresholds
    print("Recording live inference configuration...")
    card["live_thresholds"] = {
        "conf_threshold": 0.65,
        "hold_frames": 3,
        "idle_timeout_ms": 1000
    }
    
    # Origin tracking information
    print("Gathering origin tracking information...")
    origin_info = {}
    
    origin_labels_path = DATA_DIR / "origin_labels.json"
    origin_data_path = DATA_DIR / "unified_origin.npy"
    label_origin_stats_path = DATA_DIR / "label_origin_stats.json"
    
    if origin_labels_path.exists() and origin_data_path.exists():
        origin_info["enabled"] = True
        origin_info["origin_labels_sha256"] = compute_sha256(origin_labels_path)
        origin_info["label_origin_stats_sha256"] = compute_sha256(label_origin_stats_path)
        
        # Load origin distribution
        origin_data = np.load(origin_data_path)
        origin_labels = load_json_safe(origin_labels_path)
        
        if origin_labels:
            distribution = {}
            for i, label in enumerate(origin_labels):
                count = int((origin_data == i).sum())
                pct = 100.0 * count / len(origin_data)
                distribution[label] = {
                    "count": count,
                    "percentage": round(pct, 2)
                }
            origin_info["distribution"] = distribution
        
        # Load origin evaluation metrics if available
        origin_baseline_path = EVAL_DIR / "origin_baseline.json"
        if origin_baseline_path.exists():
            origin_baseline = load_json_safe(origin_baseline_path)
            if origin_baseline:
                origin_info["accuracy"] = origin_baseline.get("accuracy")
                origin_info["macro_f1"] = origin_baseline.get("macro_f1")
        
        # Check for fallback priors
        if label_origin_stats_path.exists():
            origin_info["fallback_priors_available"] = True
    else:
        origin_info["enabled"] = False
    
    card["origin_tracking"] = origin_info
    
    # Evaluation metrics
    print("Gathering evaluation metrics...")
    baseline_path = EVAL_DIR / "baseline.json"
    if baseline_path.exists():
        baseline = load_json_safe(baseline_path)
        if baseline:
            card["evaluation"] = {
                "accuracy": baseline.get("accuracy"),
                "macro_precision": baseline.get("macro_precision"),
                "macro_recall": baseline.get("macro_recall"),
                "macro_f1": baseline.get("macro_f1")
            }
    
    # Save model card
    output_path = MODELS_DIR / "model_card.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Model card saved to: {output_path}")
    print("\nModel Card Summary:")
    print(f"  Git SHA: {card['git_sha']}")
    print(f"  Dataset: {dataset_info.get('num_samples', 'N/A')} samples, "
          f"{dataset_info.get('num_classes', 'N/A')} classes")
    if training_info:
        print(f"  Training: {training_info.get('num_epochs', 'N/A')} epochs, "
              f"val_acc={training_info.get('final_val_accuracy', 0):.4f}")
    if quantized_models:
        print(f"  Quantized models: {len(quantized_models)} variants")
    if origin_info.get("enabled"):
        print(f"  Origin tracking: ENABLED")
        if "distribution" in origin_info:
            for label, stats in origin_info["distribution"].items():
                print(f"    {label}: {stats['count']} ({stats['percentage']:.1f}%)")
    
    print("\n" + "="*60)
    print("Model Card Generation Complete!")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail - model card generation is non-critical
        print("\nWarning: Model card generation failed, but continuing...")
        exit(0)  # Non-fatal exit

