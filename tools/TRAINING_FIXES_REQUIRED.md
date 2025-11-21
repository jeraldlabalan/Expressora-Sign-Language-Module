# Critical Fixes Required Before Training

## 🚨 CRITICAL BUG #1: Class Weight Dictionary Mapping

**Location:** `unified/training/train_unified_tf.py` line 43

**Problem:**
```python
class_weight_dict = {i: float(weight) for i, weight in enumerate(class_weights)}
```

This is **WRONG**. `compute_class_weight` returns weights in the order of `np.unique(y_train)`, NOT in class index order. If some classes are missing from training data, the mapping will be incorrect.

**Example of the bug:**
- If `np.unique(y_train)` returns `[0, 1, 3, 5, 7]` (missing classes 2, 4, 6)
- `class_weights` will have weights for classes: 0, 1, 3, 5, 7
- But the dict will map: `{0: weight_for_class_0, 1: weight_for_class_1, 2: weight_for_class_3, ...}`
- Class 3's weight gets assigned to index 2, which is wrong!

**Correct Fix:**
```python
# Calculate class weights to handle imbalance
print(f"\nCalculating class weights...")
unique_classes = np.unique(y_train)
class_weights = compute_class_weight(
    'balanced',
    classes=unique_classes,
    y=y_train
)
class_weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights)}

# Ensure ALL classes have weights (even if not in training set)
# This shouldn't happen, but safety check
for i in range(num_classes):
    if i not in class_weight_dict:
        class_weight_dict[i] = 1.0  # Default weight for missing classes
```

## ⚠️ ISSUE #2: Non-Stratified Train/Val Split

**Location:** `unified/training/train_unified_tf.py` lines 27-30

**Problem:**
The current split doesn't preserve class distribution, which could cause validation set to have different class distribution than training set.

**Better Approach:**
```python
from sklearn.model_selection import train_test_split

# Stratified split to preserve class distribution
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42,
    stratify=y  # Preserve class distribution
)
```

## ✅ VERIFIED: Representative Dataset Format

**Location:** `unified/export/export_unified_tflite.py` line 44

**Status:** CORRECT - The format `yield [X[i:i+1]]` is correct for TFLite. It expects a generator that yields lists/tuples of numpy arrays.

## 🔍 ADDITIONAL IMPROVEMENTS

### 1. Add Validation for Class Weights
After calculating class weights, verify they make sense:
```python
# Verify class weights
print(f"\nClass weight verification:")
sample_classes = list(unique_classes[:5])
for cls in sample_classes:
    count = np.sum(y_train == cls)
    weight = class_weight_dict[cls]
    print(f"  Class {cls}: {count} samples, weight={weight:.3f}")
```

### 2. Add Per-Class Metrics Monitoring
Consider adding a callback to monitor per-class performance:
```python
from sklearn.metrics import classification_report

class PerClassMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Every 5 epochs
            y_val_pred = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
            # Check if model is collapsing to one class
            unique_preds = len(np.unique(y_val_pred))
            print(f"\n  Epoch {epoch}: {unique_preds} unique predictions in validation set")
```

### 3. Save Class Weight Information
Save the class weights used for reference:
```python
# Save class weights for reference
with open(MODELS / "class_weights.json", 'w') as f:
    json.dump(class_weight_dict, f, indent=2)
print(f"Saved class weights to: {MODELS / 'class_weights.json'}")
```

## 📋 COMPLETE CORRECTED TRAINING SCRIPT

Here's the corrected section (lines 26-57):

```python
# Stratified train/val split to preserve class distribution
from sklearn.model_selection import train_test_split

print(f"\nTrain/Val split (stratified)...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42,
    stratify=y  # Preserve class distribution
)

print(f"  Training samples: {len(y_train)}")
print(f"  Validation samples: {len(y_val)}")

# Calculate class weights to handle imbalance
print(f"\nCalculating class weights...")
unique_classes = np.unique(y_train)
class_weights = compute_class_weight(
    'balanced',
    classes=unique_classes,
    y=y_train
)
class_weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights)}

# Ensure ALL classes have weights (safety check)
for i in range(num_classes):
    if i not in class_weight_dict:
        class_weight_dict[i] = 1.0  # Default weight for missing classes

# Show class distribution
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

# Verify a few class weights
print(f"\n  Sample class weights:")
for cls in list(unique_classes[:5]):
    count = int(np.sum(y_train == cls))
    weight = class_weight_dict[cls]
    print(f"    Class {cls}: {count} samples, weight={weight:.3f}")

# Save class weights for reference
with open(MODELS / "class_weights.json", 'w') as f:
    json.dump(class_weight_dict, f, indent=2)
print(f"  Saved class weights to: {MODELS / 'class_weights.json'}")
```

## 🎯 SUMMARY

**MUST FIX:**
1. ✅ Class weight dictionary mapping (CRITICAL BUG)
2. ✅ Use stratified train/val split (IMPORTANT)

**RECOMMENDED:**
3. Add class weight verification output
4. Save class weights to file
5. Add per-class metrics monitoring (optional)

**VERIFIED CORRECT:**
- Representative dataset format in export script
- Model architecture
- Callbacks configuration

## ⚡ QUICK FIX CHECKLIST

Before training, ensure:
- [ ] Class weight dict uses `zip(unique_classes, class_weights)` not `enumerate(class_weights)`
- [ ] Train/val split is stratified
- [ ] All classes have weights in the dict
- [ ] Class weights are saved for reference
- [ ] Import `train_test_split` from sklearn

