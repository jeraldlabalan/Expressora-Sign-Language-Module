"""
Test all unified models to check their performance and responsiveness.
"""
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
import sys

def test_model_performance(model_path, num_tests=50):
    """Test a model's performance with various inputs"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_path.name}")
    print(f"{'='*70}")
    
    if not model_path.exists():
        return {'error': 'Model not found'}
    
    try:
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
        
        # Get input dimension
        if is_tflite:
            input_dim = input_details['shape'][1] if len(input_details['shape']) > 1 else input_details['shape'][0]
            output_dim = output_details['shape'][-1]
            input_dtype = input_details.get('dtype')
            # Get quantization parameters if needed
            quant_params = input_details.get('quantization_parameters', {})
            if quant_params and len(quant_params.get('scales', [])) > 0:
                input_scale = float(quant_params['scales'][0])
                input_zero_point = int(quant_params['zero_points'][0]) if len(quant_params.get('zero_points', [])) > 0 else 0
            else:
                input_scale = 1.0
                input_zero_point = 0
        else:
            input_dim = model.input_shape[1]
            output_dim = model.output_shape[-1]
            input_dtype = np.float32
            input_scale = 1.0
            input_zero_point = 0
        
        print(f"Input dimension: {input_dim}")
        print(f"Output dimension: {output_dim}")
        if is_tflite:
            print(f"Input dtype: {input_dtype}")
        
        # Test with random inputs
        predictions = []
        confidences = []
        outputs_list = []
        
        for i in range(num_tests):
            # Handle INT8 models differently
            if is_tflite and input_dtype == np.int8:
                # For INT8 models, we need to quantize the input
                x_float = np.random.rand(1, input_dim).astype(np.float32)
                x = np.round(x_float / input_scale + input_zero_point).astype(np.int8)
                x = np.clip(x, -128, 127)  # INT8 range
            else:
                x = np.random.rand(1, input_dim).astype(np.float32)
            
            if is_tflite:
                interpreter.set_tensor(input_details['index'], x)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details['index'])[0]
            else:
                output = model.predict(x, verbose=0)[0]
            
            pred = int(np.argmax(output))
            conf = float(output[pred])
            
            predictions.append(pred)
            confidences.append(conf)
            outputs_list.append(output.copy())
        
        # Statistics
        unique_preds = len(set(predictions))
        pred_counts = {}
        for p in predictions:
            pred_counts[p] = pred_counts.get(p, 0) + 1
        
        most_common_pred = max(pred_counts, key=pred_counts.get)
        most_common_freq = pred_counts[most_common_pred] / num_tests
        
        # Output variance
        outputs_array = np.array(outputs_list)
        output_variance = np.mean(np.var(outputs_array, axis=0))
        
        # Entropy
        avg_output = np.mean(outputs_array, axis=0)
        avg_probs = avg_output / (avg_output.sum() + 1e-10)
        entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-10))
        max_entropy = np.log(len(avg_probs))
        normalized_entropy = entropy / max_entropy
        
        # Test with extreme inputs
        if is_tflite and input_dtype == np.int8:
            extreme_inputs = [
                np.zeros((1, input_dim), dtype=np.int8),
                np.full((1, input_dim), 127, dtype=np.int8),
                np.full((1, input_dim), -128, dtype=np.int8),
            ]
        else:
            extreme_inputs = [
                np.zeros((1, input_dim), dtype=np.float32),
                np.ones((1, input_dim), dtype=np.float32),
                np.full((1, input_dim), 0.5, dtype=np.float32),
            ]
        
        extreme_preds = []
        for x in extreme_inputs:
            if is_tflite:
                interpreter.set_tensor(input_details['index'], x)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details['index'])[0]
            else:
                output = model.predict(x, verbose=0)[0]
            extreme_preds.append(int(np.argmax(output)))
        
        extreme_differ = len(set(extreme_preds)) > 1
        
        # Assessment
        is_responsive = (
            unique_preds > 1 and
            output_variance > 1e-6 and
            extreme_differ
        )
        
        # Print results
        print(f"\nResults:")
        print(f"  Unique predictions: {unique_preds}/{num_tests}")
        print(f"  Most common prediction: Class {most_common_pred} ({most_common_freq:.1%})")
        print(f"  Output variance: {output_variance:.6f}")
        print(f"  Normalized entropy: {normalized_entropy:.4f} (1.0=uniform, 0.0=deterministic)")
        print(f"  Average confidence: {np.mean(confidences):.4f}")
        print(f"  Extreme inputs produce different outputs: {extreme_differ}")
        
        if is_responsive:
            print(f"\n  [OK] Model IS responsive")
        else:
            print(f"\n  [FAIL] Model appears STATIC")
            if unique_preds == 1:
                print(f"     → Always predicts class {most_common_pred}")
            if output_variance < 1e-6:
                print(f"     → Output variance is extremely low")
            if not extreme_differ:
                print(f"     → Extreme inputs produce same output")
        
        # Top predictions
        print(f"\n  Top 5 most predicted classes:")
        sorted_preds = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for pred, count in sorted_preds:
            print(f"     Class {pred}: {count}/{num_tests} ({count/num_tests:.1%})")
        
        return {
            'model_path': str(model_path),
            'model_type': 'TFLite' if is_tflite else 'Keras',
            'input_dim': int(input_dim),
            'output_dim': int(output_dim),
            'num_tests': num_tests,
            'unique_predictions': unique_preds,
            'most_common_prediction': int(most_common_pred),
            'most_common_frequency': float(most_common_freq),
            'output_variance': float(output_variance),
            'normalized_entropy': float(normalized_entropy),
            'avg_confidence': float(np.mean(confidences)),
            'extreme_inputs_differ': extreme_differ,
            'responsive': is_responsive,
            'prediction_distribution': {int(k): int(v) for k, v in pred_counts.items()}
        }
        
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        return {'error': str(e)}


def main():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "unified" / "models"
    
    # Find all model files
    model_files = [
        models_dir / "expressora_unified.keras",
        models_dir / "expressora_unified.tflite",
        models_dir / "expressora_unified_fp16.tflite",
        models_dir / "expressora_unified_int8.tflite",
    ]
    
    print("="*70)
    print("UNIFIED MODELS PERFORMANCE TEST")
    print("="*70)
    
    results = []
    
    for model_path in model_files:
        if model_path.exists():
            result = test_model_performance(model_path, num_tests=100)
            results.append(result)
        else:
            print(f"\n[WARN] {model_path.name} not found, skipping...")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    responsive_count = sum(1 for r in results if r.get('responsive', False))
    total_count = len(results)
    
    print(f"\nResponsive models: {responsive_count}/{total_count}\n")
    
    for result in results:
        if 'error' in result:
            model_name = Path(result.get('model_path', 'unknown')).name if 'model_path' in result else 'unknown'
            print(f"[ERROR] {model_name}: {result['error']}")
        else:
            status = "[OK]" if result.get('responsive', False) else "[FAIL]"
            print(f"{status} {Path(result['model_path']).name} ({result['model_type']})")
            print(f"     Unique predictions: {result['unique_predictions']}/100")
            print(f"     Most common: Class {result['most_common_prediction']} "
                  f"({result['most_common_frequency']:.1%})")
            print(f"     Output variance: {result['output_variance']:.6f}")
            print()
    
    # Load labels to show what class 41 is
    labels_path = models_dir / "expressora_labels.json"
    if labels_path.exists():
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        
        print("Class 41 (most commonly predicted):", labels[41] if len(labels) > 41 else "N/A")
        print()
    
    # Save results
    results_file = root / "tools" / "unified_models_performance.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    return 0 if responsive_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())

