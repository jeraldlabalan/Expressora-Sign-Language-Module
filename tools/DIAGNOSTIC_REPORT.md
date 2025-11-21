# Model Diagnostic Report

## Executive Summary

**Problem Identified**: The unified model (`expressora_unified.tflite`) is producing static outputs, always predicting class 41 ("gabi" - night/evening) regardless of input changes. However, this issue is **NOT** caused by the TensorFlow Lite interpreter or quantization - the problem exists in the original Keras model itself.

## Key Findings

### 1. Unified Model is Static
- **Test Results**: Out of 20 random inputs, the model produces only 1-3 unique predictions
- **Primary Prediction**: Class 41 ("gabi" - night/evening) is predicted ~84-95% of the time
- **Output Variance**: Extremely low (0.000000 to 0.001364)
- **Both Keras and TFLite**: The issue exists in both formats, confirming it's not a conversion problem

### 2. Individual Models Work Correctly
- **ASL Alphabet Model**: ✅ Responsive (5 unique predictions out of 20)
- **FSL Alphabet Model**: ✅ Responsive (6 unique predictions out of 20)
- These models correctly respond to different inputs

### 3. Model Training Was Successful
- Training logs show the model reached ~90% accuracy
- Validation accuracy reached ~90% as well
- The model DID learn, but learned a biased solution

## Root Cause Analysis

The model appears to have **collapsed to a degenerate solution** during training. Possible causes:

1. **Class Imbalance**: Class 41 ("gabi" - night/evening) may be overrepresented in the training data
2. **Normalization Layer Issue**: The `Normalization` layer may be causing the model to ignore input variations
3. **Model Architecture**: The simple 3-layer architecture may be insufficient for the complexity of 197 classes
4. **Training Dynamics**: The model may have converged to a local minimum that favors one class

## Evidence

### Diagnostic Test Results

**Unified Model (expressora_unified.tflite)**:
```
Unique predictions: 1 out of 20
Output variance: 0.000000
All outputs identical: True
All predictions identical: True
Always predicts class 41
```

**Keras vs TFLite Comparison**:
```
Keras model: 3 unique predictions out of 20
TFLite model: 3 unique predictions out of 20
Average output difference: 0.000000 (perfect match)
```

This confirms:
- The TFLite conversion is correct
- The problem exists in the original trained model
- Quantization is not the issue

### Individual Models (Working Correctly)

**ASL Alphabet Model**:
```
Unique predictions: 5 out of 20
Output variance: 0.019851
Responsive: ✅ YES
```

**FSL Alphabet Model**:
```
Unique predictions: 6 out of 20
Output variance: 0.021462
Responsive: ✅ YES
```

## Recommendations

### Immediate Fixes

1. **Check Training Data Distribution**
   - Analyze class distribution in `unified/data/unified_X.npy` and `unified_y.npy`
   - Verify if class 41 is overrepresented
   - Use class weights or balanced sampling if needed

2. **Retrain with Better Configuration**
   - Add class weights to handle imbalance
   - Increase model capacity (more layers/neurons)
   - Use different normalization approach
   - Try different optimizers or learning rates
   - Add regularization to prevent collapse

3. **Verify Data Preprocessing**
   - Check if the normalization layer is working correctly
   - Ensure input features are properly scaled
   - Verify that the 126-dim feature extraction is correct

### Long-term Improvements

1. **Model Architecture**
   - Consider deeper networks (4-5 layers)
   - Add dropout for regularization
   - Consider batch normalization instead of layer normalization

2. **Training Strategy**
   - Use stratified train/val split
   - Implement class balancing
   - Monitor per-class metrics during training
   - Use early stopping based on per-class F1 scores

3. **Evaluation**
   - Add per-class accuracy reporting
   - Monitor prediction distribution during training
   - Alert if model collapses to single class

## Conclusion

The models **did learn** (training accuracy ~90%), but the unified model learned a degenerate solution that heavily favors one class. This is a **training/data issue**, not an inference or conversion issue. The TensorFlow Lite interpreter is working correctly - it's faithfully reproducing the biased behavior of the trained model.

## Files Created

1. `tools/diagnose_models.py` - Main diagnostic script
2. `tools/test_input_sensitivity.py` - Input sensitivity testing
3. `tools/compare_keras_tflite.py` - Keras vs TFLite comparison
4. `tools/diagnostic_results.json` - Detailed test results
5. `tools/sensitivity_results.json` - Sensitivity test results
6. `tools/keras_tflite_comparison.json` - Comparison results

## Next Steps

1. Analyze training data class distribution
2. Retrain model with class balancing
3. Monitor training to prevent class collapse
4. Re-export TFLite model after successful retraining

