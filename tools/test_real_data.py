"""
Test model on actual dataset samples (not random inputs).
This provides a more realistic assessment of model performance.
"""
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "unified" / "data"
MODELS = ROOT / "unified" / "models"

def test_model_on_real_data(model_path, num_samples=1000):
    """Test model on actual dataset samples"""
    print(f"\n{'='*70}")
    print(f"Testing {model_path.name} on REAL DATA")
    print(f"{'='*70}")
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
    
    # Load data
    X = np.load(DATA / "unified_X.npy")
    y = np.load(DATA / "unified_y.npy")
    
    with open(DATA / "labels.json", 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    # Use test set (last 10% of data)
    n = len(y)
    split = int(n * 0.9)
    X_test = X[split:]
    y_test = y[split:]
    
    # Sample subset for testing
    if len(X_test) > num_samples:
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
    
    print(f"\nTest set: {len(y_test)} samples")
    
    # Load model
    if model_path.suffix == '.keras':
        model = tf.keras.models.load_model(str(model_path))
        is_tflite = False
    else:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        is_tflite = True
    
    # Apply same scaling as training
    X_mean = np.load(DATA / "feature_mean.npy") if (DATA / "feature_mean.npy").exists() else np.mean(X, axis=0, keepdims=True)
    X_std = np.load(DATA / "feature_std.npy") if (DATA / "feature_std.npy").exists() else np.std(X, axis=0, keepdims=True) + 1e-8
    X_test_scaled = (X_test - X_mean) / X_std
    
    # Predict
    predictions = []
    confidences = []
    
    print(f"\nRunning inference...")
    for i in range(len(X_test)):
        x = X_test_scaled[i:i+1]
        
        if is_tflite:
            interpreter.set_tensor(input_details['index'], x.astype(input_details['dtype']))
            interpreter.invoke()
            output = interpreter.get_tensor(output_details['index'])[0]
        else:
            output = model.predict(x, verbose=0)[0]
        
        pred = int(np.argmax(output))
        conf = float(output[pred])
        predictions.append(pred)
        confidences.append(conf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    unique_preds = len(set(predictions))
    pred_counts = {}
    for p in predictions:
        pred_counts[p] = pred_counts.get(p, 0) + 1
    
    most_common_pred = max(pred_counts, key=pred_counts.get)
    most_common_freq = pred_counts[most_common_pred] / len(predictions)
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Unique predictions: {unique_preds}/{len(predictions)}")
    print(f"  Most common prediction: Class {most_common_pred} ({most_common_freq:.1%})")
    print(f"  Average confidence: {np.mean(confidences):.4f}")
    
    # Per-class metrics
    print(f"\nPer-class performance (top 10 by frequency):")
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    
    class_scores = []
    for i in range(len(labels)):
        if str(i) in report:
            class_scores.append({
                'class': i,
                'label': labels[i],
                'precision': report[str(i)]['precision'],
                'recall': report[str(i)]['recall'],
                'f1': report[str(i)]['f1-score'],
                'support': report[str(i)]['support']
            })
    
    class_scores.sort(key=lambda x: x['support'], reverse=True)
    for item in class_scores[:10]:
        print(f"    {item['label']:20s} - Prec: {item['precision']:.3f}, Rec: {item['recall']:.3f}, F1: {item['f1']:.3f}, Support: {item['support']}")
    
    return {
        'model_path': str(model_path),
        'accuracy': float(accuracy),
        'unique_predictions': unique_preds,
        'most_common_prediction': int(most_common_pred),
        'most_common_frequency': float(most_common_freq),
        'avg_confidence': float(np.mean(confidences)),
        'class_scores': class_scores
    }

if __name__ == "__main__":
    model_path = MODELS / "expressora_unified.tflite"
    if not model_path.exists():
        model_path = MODELS / "expressora_unified.keras"
    
    if model_path.exists():
        result = test_model_on_real_data(model_path, num_samples=1000)
        if result:
            print(f"\n{'='*70}")
            print("SUMMARY")
            print(f"{'='*70}")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Unique predictions: {result['unique_predictions']}")
            print(f"Most common: Class {result['most_common_prediction']} ({result['most_common_frequency']:.1%})")
    else:
        print(f"Model not found. Please train first.")

