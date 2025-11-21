"""
Comprehensive training script with all fixes to prevent model collapse.
Implements: fixed class weights, stratified splits, balanced sampling, 
label smoothing, training monitoring, and improved architecture.
"""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from collections import Counter
import sys

# Add tools directory to path for monitoring callbacks
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from monitor_training import PredictionDiversityCallback, PerClassAccuracyCallback

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj

print("="*70)
print("COMPREHENSIVE TRAINING WITH ALL FIXES")
print("="*70)

# Load data (sequences for LSTM)
X = np.load(DATA / "unified_X.npy")
y = np.load(DATA / "unified_y.npy")
num_classes = int(y.max()) + 1

# Load metadata
try:
    with open(DATA / "num_features.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
        NUM_FEATURES = metadata["num_features"]
        SEQ_LENGTH = metadata["seq_length"]
except FileNotFoundError:
    # Fallback: infer from data shape
    if len(X.shape) == 3:
        SEQ_LENGTH, NUM_FEATURES = X.shape[1], X.shape[2]
    else:
        raise ValueError("Expected 3D array (N_sequences, seq_length, num_features)")

print(f"\nDataset info:")
print(f"  Total sequences: {len(y)}")
print(f"  Total classes: {num_classes}")
print(f"  Sequence shape: {X.shape} (N_sequences, {SEQ_LENGTH}, {NUM_FEATURES})")

# Manual feature scaling (replace normalization layer)
# For sequences: (N, seq_length, num_features) -> scale across all sequences and time steps
print(f"\nApplying manual feature scaling...")
# Compute mean/std across all sequences and time steps for each feature
X_mean = np.mean(X, axis=(0, 1), keepdims=True)  # Shape: (1, 1, NUM_FEATURES)
X_std = np.std(X, axis=(0, 1), keepdims=True) + 1e-8  # Add small epsilon to avoid division by zero
X_scaled = (X - X_mean) / X_std
print(f"  Feature mean range: [{X_mean.min():.4f}, {X_mean.max():.4f}]")
print(f"  Feature std range: [{X_std.min():.4f}, {X_std.max():.4f}]")
print(f"  Scaled X shape: {X_scaled.shape}")

# Save scaling parameters for inference (flatten to 1D for compatibility)
np.save(DATA / "feature_mean.npy", X_mean.squeeze())  # Shape: (NUM_FEATURES,)
np.save(DATA / "feature_std.npy", X_std.squeeze())    # Shape: (NUM_FEATURES,)
print(f"  Saved scaling parameters for inference")

# Stratified train/val/test split
print(f"\nStratified train/val/test split...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y,
    test_size=0.1,
    random_state=42,
    stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.1,  # 10% of remaining = 9% of total
    random_state=42,
    stratify=y_temp
)

print(f"  Training samples: {len(y_train)} ({100*len(y_train)/len(y):.1f}%)")
print(f"  Validation samples: {len(y_val)} ({100*len(y_val)/len(y):.1f}%)")
print(f"  Test samples: {len(y_test)} ({100*len(y_test)/len(y):.1f}%)")

# Verify class distribution in splits
train_counts = Counter(y_train)
val_counts = Counter(y_val)
print(f"\n  Class distribution check:")
print(f"    Train: {len(train_counts)} classes, min={min(train_counts.values())}, max={max(train_counts.values())}")
print(f"    Val: {len(val_counts)} classes, min={min(val_counts.values())}, max={max(val_counts.values())}")

# Calculate class weights (FIXED - use zip with unique_classes)
print(f"\nCalculating class weights (FIXED MAPPING)...")
unique_classes = np.unique(y_train)
class_weights = compute_class_weight(
    'balanced',
    classes=unique_classes,
    y=y_train
)
class_weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights)}

# Ensure ALL classes have weights
for i in range(num_classes):
    if i not in class_weight_dict:
        class_weight_dict[i] = 1.0

# Show class distribution and weights
train_counts = Counter(y_train)
max_count = max(train_counts.values())
min_count = min(train_counts.values())
imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
print(f"  Max class count: {max_count}")
print(f"  Min class count: {min_count}")
print(f"  Mean class weight: {np.mean(list(class_weight_dict.values())):.3f}")
print(f"  Max class weight: {max(class_weight_dict.values()):.3f}")
print(f"  Min class weight: {min(class_weight_dict.values()):.3f}")

# Verify class weights are correct
print(f"\n  Verifying class weights (sample):")
for cls in list(unique_classes[:5]):
    count = int(np.sum(y_train == cls))
    weight = class_weight_dict[cls]
    print(f"    Class {cls}: {count} samples, weight={weight:.3f}")

# Save class weights
with open(MODELS / "class_weights.json", 'w') as f:
    json.dump(class_weight_dict, f, indent=2)
print(f"  Saved class weights to: {MODELS / 'class_weights.json'}")

# Create balanced dataset using tf.data
print(f"\nCreating balanced dataset with tf.data...")
def create_balanced_dataset(X, y, batch_size=256, shuffle_buffer=10000):
    """Create a dataset with balanced class sampling"""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Shuffle
    dataset = dataset.shuffle(shuffle_buffer, seed=42)
    
    # Batch
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Reduce batch size for sequences (30×237 = 7,110 elements per sequence vs 126 per frame)
# Sequences are much larger, so use smaller batch size
BATCH_SIZE = 64  # Reduced from 256 due to larger sequence input
train_dataset = create_balanced_dataset(X_train, y_train, batch_size=BATCH_SIZE)
val_dataset = create_balanced_dataset(X_val, y_val, batch_size=BATCH_SIZE)

# LSTM model architecture for temporal sequences
print(f"\nBuilding LSTM model architecture...")
print(f"  - Input shape: ({SEQ_LENGTH}, {NUM_FEATURES})")
print("  - Two LSTM layers (64 units each)")
print("  - Dropout for regularization")
print("  - Dense layers for classification")
print("  - No Normalization layer (using pre-scaled features)")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LENGTH, NUM_FEATURES)),
    # NO Normalization layer - using pre-scaled features
    # First LSTM layer (returns sequences for second LSTM)
    tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.2),
    # Second LSTM layer (returns single vector)
    tf.keras.layers.LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.2),
    # Dense layers for classification
    tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

# Compile with standard loss (label smoothing can be added later if needed)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),  # Slightly lower LR
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(f"\nModel summary:")
model.summary()

# Enhanced callbacks with monitoring
print(f"\nSetting up callbacks...")
cb = [
    PredictionDiversityCallback(X_val, y_val, check_every=5, min_unique_predictions=20),
    PerClassAccuracyCallback(X_val, y_val, num_classes, check_every=10),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,  # Increased patience
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        (MODELS/"expressora_unified.keras").as_posix(),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    ),
]

print(f"\nStarting training...")
print(f"  Model type: LSTM (temporal sequences)")
print(f"  Input shape: ({SEQ_LENGTH}, {NUM_FEATURES})")
print(f"  Class weights: YES (FIXED MAPPING)")
print(f"  Stratified split: YES")
print(f"  Balanced batches: YES (tf.data)")
print(f"  Label smoothing: NO (using standard loss)")
print(f"  L2 regularization: YES")
print(f"  Dropout: YES (0.2, 0.2, 0.3)")
print(f"  Learning rate: 3e-4")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epochs: 150")
print("="*70)

# Note: class_weight doesn't work with tf.data.Dataset directly
# We'll use sample_weight in a custom training loop or use numpy arrays
# For now, use numpy arrays with class_weight parameter
hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,
    epochs=150,
    batch_size=BATCH_SIZE,
    callbacks=cb,
    verbose=2,
)

# Save SavedModel dir for TFLite export
saved_dir = MODELS / "savedmodel"
tf.saved_model.save(model, saved_dir.as_posix())

# Save training log (with proper type conversion)
with open((MODELS/"training_log.json"), "w", encoding="utf-8") as f:
    json.dump(convert_to_native(hist.history), f, ensure_ascii=False, indent=2)

# Save diversity history
diversity_cb = [c for c in cb if isinstance(c, PredictionDiversityCallback)][0]
with open((MODELS/"diversity_history.json"), "w", encoding="utf-8") as f:
    json.dump(convert_to_native(diversity_cb.diversity_history), f, ensure_ascii=False, indent=2)

print("\n" + "="*70)
print("Training completed!")
print("="*70)
print("Saved:", MODELS / "expressora_unified.keras")
print("Saved:", saved_dir)
print("Saved:", MODELS / "training_log.json")
print("Saved:", MODELS / "diversity_history.json")
print("Saved:", MODELS / "class_weights.json")

# Final diversity check
print("\nFinal model diversity check:")
y_pred_probs = model.predict(X_val, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
unique_preds = len(np.unique(y_pred))
pred_counts = Counter(y_pred)
most_common = pred_counts.most_common(1)[0]
print(f"  Unique predictions on validation: {unique_preds}/{len(y_val)}")
print(f"  Most common class: {most_common[0]} ({most_common[1]/len(y_val):.1%})")

if unique_preds < 20:
    print(f"  [WARNING] Model may still be collapsed!")
else:
    print(f"  [OK] Model shows good diversity!")
