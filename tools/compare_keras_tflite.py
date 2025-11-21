"""
Compare Keras model vs TFLite model outputs to identify quantization issues.
"""
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
import sys

def test_model(model, model_type, num_tests=10, input_dim=126):
    """Test a model (Keras or TFLite) and return results"""
    print(f"\n{'='*60}")
    print(f"Testing {model_type} model")
    print(f"{'='*60}")
    
    predictions = []
    outputs_list = []
    
    for i in range(num_tests):
        x = np.random.rand(1, input_dim).astype(np.float32)
        
        if model_type == "Keras":
            output = model.predict(x, verbose=0)[0]
        else:  # TFLite
            interpreter = model
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]
            
            interpreter.set_tensor(input_details['index'], x)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details['index'])[0]
        
        predictions.append(int(np.argmax(output)))
        outputs_list.append(output.copy())
    
    unique_preds = len(set(predictions))
    outputs_array = np.array(outputs_list)
    variance = np.mean(np.var(outputs_array, axis=0))
    
    print(f"Unique predictions: {unique_preds}/{num_tests}")
    print(f"Output variance: {variance:.6f}")
    print(f"Predictions: {predictions[:10]}...")
    
    return {
        'unique_predictions': unique_preds,
        'variance': variance,
        'predictions': predictions
    }


def main():
    root = Path(__file__).resolve().parents[1]
    
    # Load Keras model
    keras_path = root / "unified" / "models" / "expressora_unified.keras"
    if not keras_path.exists():
        print(f"❌ Keras model not found: {keras_path}")
        return 1
    
    print("Loading Keras model...")
    keras_model = tf.keras.models.load_model(str(keras_path))
    
    # Load TFLite model
    tflite_path = root / "unified" / "models" / "expressora_unified.tflite"
    if not tflite_path.exists():
        print(f"❌ TFLite model not found: {tflite_path}")
        return 1
    
    print("Loading TFLite model...")
    tflite_interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    tflite_interpreter.allocate_tensors()
    
    # Get input dimension
    input_details = tflite_interpreter.get_input_details()[0]
    input_dim = input_details['shape'][1] if len(input_details['shape']) > 1 else input_details['shape'][0]
    
    # Test both models with same inputs
    print(f"\n{'='*60}")
    print("COMPARISON TEST")
    print(f"{'='*60}")
    print("Testing both models with the same random inputs...")
    
    keras_results = test_model(keras_model, "Keras", num_tests=20, input_dim=input_dim)
    tflite_results = test_model(tflite_interpreter, "TFLite", num_tests=20, input_dim=input_dim)
    
    # Compare outputs on same inputs
    print(f"\n{'='*60}")
    print("DIRECT OUTPUT COMPARISON")
    print(f"{'='*60}")
    
    differences = []
    for i in range(10):
        x = np.random.rand(1, input_dim).astype(np.float32)
        
        # Keras output
        keras_out = keras_model.predict(x, verbose=0)[0]
        keras_pred = int(np.argmax(keras_out))
        
        # TFLite output
        input_details = tflite_interpreter.get_input_details()[0]
        output_details = tflite_interpreter.get_output_details()[0]
        tflite_interpreter.set_tensor(input_details['index'], x)
        tflite_interpreter.invoke()
        tflite_out = tflite_interpreter.get_tensor(output_details['index'])[0]
        tflite_pred = int(np.argmax(tflite_out))
        
        # Compare
        max_diff = np.max(np.abs(keras_out - tflite_out))
        pred_match = keras_pred == tflite_pred
        
        differences.append({
            'input_idx': i,
            'max_output_diff': float(max_diff),
            'predictions_match': pred_match,
            'keras_pred': int(keras_pred),
            'tflite_pred': int(tflite_pred)
        })
        
        print(f"Input {i+1}: max_diff={max_diff:.6f}, pred_match={pred_match}, "
              f"keras={keras_pred}, tflite={tflite_pred}")
    
    avg_diff = np.mean([d['max_output_diff'] for d in differences])
    pred_matches = sum(1 for d in differences if d['predictions_match'])
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Keras model:")
    print(f"  Unique predictions: {keras_results['unique_predictions']}/20")
    print(f"  Output variance: {keras_results['variance']:.6f}")
    print(f"\nTFLite model:")
    print(f"  Unique predictions: {tflite_results['unique_predictions']}/20")
    print(f"  Output variance: {tflite_results['variance']:.6f}")
    print(f"\nDirect comparison:")
    print(f"  Average output difference: {avg_diff:.6f}")
    print(f"  Prediction matches: {pred_matches}/10")
    
    if keras_results['unique_predictions'] > tflite_results['unique_predictions']:
        print(f"\n⚠️  WARNING: Keras model is more responsive than TFLite!")
        print(f"   This suggests quantization may have degraded the model.")
    
    if avg_diff > 0.1:
        print(f"\n⚠️  WARNING: Large output differences between Keras and TFLite!")
        print(f"   Quantization may have introduced significant errors.")
    
    # Save results
    results = {
        'keras_results': keras_results,
        'tflite_results': tflite_results,
        'direct_comparison': differences,
        'summary': {
            'avg_output_diff': float(avg_diff),
            'prediction_matches': pred_matches,
            'keras_unique_preds': keras_results['unique_predictions'],
            'tflite_unique_preds': tflite_results['unique_predictions']
        }
    }
    
    results_file = root / "tools" / "keras_tflite_comparison.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

