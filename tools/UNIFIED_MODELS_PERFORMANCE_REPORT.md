# Unified Models Performance Report

## Executive Summary

All unified models in the `unified/models/` folder exhibit **severe performance issues**. They are heavily biased toward a single class (Class 41: "gabi" - night/evening), predicting it 78-88% of the time across all model formats.

## Test Results

### Models Tested
1. `expressora_unified.keras` (Keras format)
2. `expressora_unified.tflite` (Float32 TFLite)
3. `expressora_unified_fp16.tflite` (Float16 quantized)
4. `expressora_unified_int8.tflite` (INT8 quantized)

### Performance Metrics

| Model | Unique Predictions | Most Common Class | Frequency | Output Variance |
|-------|-------------------|------------------|-----------|-----------------|
| Keras | 3/100 | Class 41 ("gabi") | 83% | 0.001475 |
| TFLite (FP32) | 3/100 | Class 41 ("gabi") | 79% | 0.001776 |
| TFLite (FP16) | 3/100 | Class 41 ("gabi") | 78% | 0.001815 |
| TFLite (INT8) | 3/100 | Class 41 ("gabi") | 88% | 60.752996* |

*Note: INT8 model's high variance may be due to quantization artifacts, but the prediction bias is consistent.

### Key Findings

1. **Extreme Class Bias**: All models predict only 3 classes out of 197 total classes:
   - Class 41 ("gabi") - 78-88% of predictions
   - Class 125 ("puti" - white) - 5-12% of predictions  
   - Class 174 ("v") - 5-7% of predictions

2. **Low Output Variance**: Output variance is extremely low (0.001-0.002), indicating the model produces nearly identical outputs regardless of input.

3. **Consistent Across Formats**: The problem exists in all model formats (Keras, FP32, FP16, INT8), confirming this is a **training issue**, not a conversion/quantization issue.

4. **Model Architecture**: 
   - Input: 126 dimensions (hand landmarks)
   - Output: 197 classes
   - Architecture: 3-layer Dense network (256 → 128 → 197)

## Root Cause Analysis

The models **did train** (training logs show ~90% accuracy), but they learned a **degenerate solution** that heavily favors one class. This is a classic case of:

1. **Class Imbalance**: Class 41 may be overrepresented in the training data
2. **Model Collapse**: The model converged to a local minimum that predicts the majority class
3. **Insufficient Regularization**: The model lacks dropout or other regularization to prevent collapse
4. **Normalization Layer**: The `Normalization` layer may be causing the model to ignore input variations

## Impact on Application

**The models are NOT usable in their current state** because:
- They will predict "gabi" (night/evening) for almost any input
- Only 1.5% of possible classes (3/197) are ever predicted
- The model does not respond meaningfully to different hand gestures

## Recommendations

### Immediate Actions

1. **Analyze Training Data**:
   ```python
   # Check class distribution
   import numpy as np
   y = np.load('unified/data/unified_y.npy')
   unique, counts = np.unique(y, return_counts=True)
   # Check if class 41 is overrepresented
   ```

2. **Retrain with Class Balancing**:
   - Use `class_weight` parameter in `model.fit()`
   - Implement stratified sampling
   - Use balanced train/val splits

3. **Improve Model Architecture**:
   - Add dropout layers (0.3-0.5)
   - Increase model capacity
   - Add batch normalization
   - Consider deeper networks

4. **Monitor Training**:
   - Track per-class accuracy during training
   - Monitor prediction distribution
   - Alert if model collapses to single class
   - Use early stopping based on balanced metrics

### Code Changes Needed

**In `unified/training/train_unified_tf.py`**:

```python
# Add class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Add dropout
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Normalization(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),  # ADD THIS
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),  # ADD THIS
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

# Use class weights in training
hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,  # ADD THIS
    epochs=50,
    batch_size=256,
    callbacks=cb,
    verbose=2,
)
```

## Conclusion

The unified models have a **fundamental training problem** - they learned to predict only 3 classes out of 197, with class 41 ("gabi") dominating 78-88% of predictions. This is **not an inference or conversion issue** - the TensorFlow Lite interpreter is working correctly. The models need to be retrained with proper class balancing and regularization.

## Test Scripts Created

- `tools/test_unified_models.py` - Comprehensive performance testing
- `tools/diagnose_models.py` - General model diagnostics
- `tools/test_input_sensitivity.py` - Input sensitivity analysis
- `tools/compare_keras_tflite.py` - Keras vs TFLite comparison

All test results are saved in `tools/unified_models_performance.json`.

