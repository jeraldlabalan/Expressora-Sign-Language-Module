"""
Build a representative dataset for TFLite INT8 quantization.
Extracts ~1000 stratified samples across all labels.
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "unified" / "data"

# Configuration
TARGET_SAMPLES = 1000
MIN_PER_CLASS = 1

def main():
    # Load full dataset
    X = np.load(DATA_DIR / "unified_X.npy")
    y = np.load(DATA_DIR / "unified_y.npy")
    
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    num_classes = int(y.max()) + 1
    print(f"Number of classes: {num_classes}")
    
    # Stratified sampling
    rng = np.random.default_rng(42)
    selected_indices = []
    
    # Calculate samples per class
    samples_per_class = max(MIN_PER_CLASS, TARGET_SAMPLES // num_classes)
    
    for class_idx in range(num_classes):
        class_mask = (y == class_idx)
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) == 0:
            continue
        
        # Sample with replacement if needed
        n_to_sample = min(samples_per_class, len(class_indices))
        sampled = rng.choice(class_indices, size=n_to_sample, replace=False)
        selected_indices.extend(sampled.tolist())
    
    # Trim to target size if we oversampled
    if len(selected_indices) > TARGET_SAMPLES:
        selected_indices = rng.choice(
            selected_indices, size=TARGET_SAMPLES, replace=False
        ).tolist()
    
    # Extract representative samples
    rep_X = X[selected_indices]
    
    # Save
    out_path = DATA_DIR / "rep_set.npy"
    np.save(out_path, rep_X)
    
    print(f"\nCreated representative set: {rep_X.shape}")
    print(f"Saved to: {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.2f} KB")

if __name__ == "__main__":
    main()

