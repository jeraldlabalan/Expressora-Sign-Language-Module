"""Analyze class distribution in training data"""
import numpy as np
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
data_dir = root / "unified" / "data"
models_dir = root / "unified" / "models"

# Load data
X = np.load(data_dir / "unified_X.npy")
y = np.load(data_dir / "unified_y.npy")

# Load labels
with open(data_dir / "labels.json", 'r', encoding='utf-8') as f:
    labels = json.load(f)

print("="*70)
print("CLASS DISTRIBUTION ANALYSIS")
print("="*70)
print(f"\nTotal samples: {len(y)}")
print(f"Total classes: {len(labels)}")

# Calculate class distribution
unique, counts = np.unique(y, return_counts=True)
class_dist = dict(zip(unique, counts))

# Sort by count
sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop 10 most frequent classes:")
for class_idx, count in sorted_classes[:10]:
    pct = 100.0 * count / len(y)
    label = labels[class_idx] if class_idx < len(labels) else f"Class_{class_idx}"
    print(f"  Class {class_idx:3d} ({label:20s}): {count:6d} samples ({pct:5.2f}%)")

print(f"\nBottom 10 least frequent classes:")
for class_idx, count in sorted_classes[-10:]:
    pct = 100.0 * count / len(y)
    label = labels[class_idx] if class_idx < len(labels) else f"Class_{class_idx}"
    print(f"  Class {class_idx:3d} ({label:20s}): {count:6d} samples ({pct:5.2f}%)")

# Check class 41 specifically
if 41 in class_dist:
    count_41 = class_dist[41]
    pct_41 = 100.0 * count_41 / len(y)
    label_41 = labels[41] if 41 < len(labels) else "Unknown"
    print(f"\nClass 41 ({label_41}): {count_41} samples ({pct_41:.2f}%)")

# Calculate imbalance ratio
max_count = max(counts)
min_count = min(counts)
imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}x")
print(f"Max class count: {max_count}")
print(f"Min class count: {min_count}")
print(f"Mean class count: {np.mean(counts):.2f}")
print(f"Median class count: {np.median(counts):.2f}")

# Classes with very few samples
rare_classes = [idx for idx, count in class_dist.items() if count < 10]
print(f"\nClasses with <10 samples: {len(rare_classes)}")

# Save distribution
dist_file = root / "tools" / "class_distribution.json"
with open(dist_file, 'w') as f:
    json.dump({
        'total_samples': int(len(y)),
        'total_classes': len(labels),
        'class_distribution': {int(k): int(v) for k, v in class_dist.items()},
        'imbalance_ratio': float(imbalance_ratio),
        'max_count': int(max_count),
        'min_count': int(min_count),
        'rare_classes_count': len(rare_classes)
    }, f, indent=2)

print(f"\nDistribution saved to: {dist_file}")

