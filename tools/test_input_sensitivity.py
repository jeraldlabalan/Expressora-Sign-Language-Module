"""
Test input sensitivity of TFLite models.
Measures how much outputs vary when inputs change.
"""
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
import sys

def calculate_output_statistics(outputs):
    """Calculate statistics about output variance"""
    outputs_array = np.array(outputs)
    
    # Variance across samples for each class
    class_variances = np.var(outputs_array, axis=0)
    
    # Mean and max variance
    mean_variance = np.mean(class_variances)
    max_variance = np.max(class_variances)
    
    # Prediction consistency
    predictions = [int(np.argmax(out)) for out in outputs]
    unique_predictions = len(set(predictions))
    most_common_pred = max(set(predictions), key=predictions.count)
    most_common_count = predictions.count(most_common_pred)
    
    # Output range for each class
    min_outputs = np.min(outputs_array, axis=0)
    max_outputs = np.max(outputs_array, axis=0)
    output_ranges = max_outputs - min_outputs
    
    return {
        'mean_variance': float(mean_variance),
        'max_variance': float(max_variance),
        'class_variances': [float(v) for v in class_variances],
        'unique_predictions': unique_predictions,
        'total_samples': len(outputs),
        'most_common_prediction': int(most_common_pred),
        'most_common_frequency': most_common_count,
        'prediction_consistency': most_common_count / len(outputs),
        'output_ranges': [float(r) for r in output_ranges],
        'mean_output_range': float(np.mean(output_ranges))
    }


def test_input_sensitivity(model_path, num_samples=50):
    """
    Test how sensitive the model is to input changes.
    
    Args:
        model_path: Path to .tflite model
        num_samples: Number of different inputs to test
        
    Returns:
        dict with sensitivity metrics
    """
    print(f"\n{'='*60}")
    print(f"Input Sensitivity Test: {model_path.name}")
    print(f"{'='*60}")
    
    if not model_path.exists():
        return {'error': f'Model not found: {model_path}'}
    
    try:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        input_shape = input_details['shape']
        input_dim = input_shape[1] if len(input_shape) > 1 else input_shape[0]
        
        print(f"Input dimension: {input_dim}")
        print(f"Output dimension: {output_details['shape'][-1]}")
        print(f"Testing with {num_samples} different inputs...")
        
        outputs = []
        inputs_used = []
        
        # Test 1: Random inputs
        print(f"\n[Test 1] Random inputs...")
        for i in range(num_samples):
            x = np.random.rand(1, input_dim).astype(input_details['dtype'])
            inputs_used.append(x.copy())
            
            interpreter.set_tensor(input_details['index'], x)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details['index'])
            if len(output.shape) > 1:
                output = output[0]
            outputs.append(output.copy())
        
        stats_random = calculate_output_statistics(outputs)
        print(f"  Unique predictions: {stats_random['unique_predictions']}/{num_samples}")
        print(f"  Mean variance: {stats_random['mean_variance']:.6f}")
        print(f"  Prediction consistency: {stats_random['prediction_consistency']:.2%}")
        
        # Test 2: Incrementally different inputs
        print(f"\n[Test 2] Incrementally different inputs...")
        base_input = np.random.rand(1, input_dim).astype(input_details['dtype'])
        outputs_incremental = []
        
        for i in range(num_samples):
            # Gradually change input
            noise = np.random.randn(1, input_dim).astype(input_details['dtype']) * (i / num_samples) * 0.1
            x = base_input + noise
            x = np.clip(x, 0, 1)  # Keep in valid range
            
            interpreter.set_tensor(input_details['index'], x)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details['index'])
            if len(output.shape) > 1:
                output = output[0]
            outputs_incremental.append(output.copy())
        
        stats_incremental = calculate_output_statistics(outputs_incremental)
        print(f"  Unique predictions: {stats_incremental['unique_predictions']}/{num_samples}")
        print(f"  Mean variance: {stats_incremental['mean_variance']:.6f}")
        
        # Test 3: Extreme inputs
        print(f"\n[Test 3] Extreme inputs...")
        extreme_inputs = [
            np.zeros((1, input_dim), dtype=input_details['dtype']),  # All zeros
            np.ones((1, input_dim), dtype=input_details['dtype']),   # All ones
            np.full((1, input_dim), 0.5, dtype=input_details['dtype']),  # All 0.5
        ]
        
        extreme_outputs = []
        for x in extreme_inputs:
            interpreter.set_tensor(input_details['index'], x)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details['index'])
            if len(output.shape) > 1:
                output = output[0]
            extreme_outputs.append(output.copy())
        
        # Check if extreme inputs produce different outputs
        extreme_differ = not all(
            np.allclose(extreme_outputs[0], out, atol=1e-4) 
            for out in extreme_outputs[1:]
        )
        
        print(f"  Extreme inputs produce different outputs: {extreme_differ}")
        for i, out in enumerate(extreme_outputs):
            pred = int(np.argmax(out))
            print(f"    Input {i}: pred={pred}, max_prob={out.max():.4f}")
        
        # Test 4: Single input, multiple invocations (should be identical)
        print(f"\n[Test 4] Same input, multiple invocations (consistency check)...")
        test_input = np.random.rand(1, input_dim).astype(input_details['dtype'])
        consistent_outputs = []
        
        for _ in range(10):
            interpreter.set_tensor(input_details['index'], test_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details['index'])
            if len(output.shape) > 1:
                output = output[0]
            consistent_outputs.append(output.copy())
        
        # Check if outputs are consistent
        all_consistent = all(
            np.allclose(consistent_outputs[0], out, atol=1e-6) 
            for out in consistent_outputs[1:]
        )
        print(f"  Outputs are consistent: {all_consistent}")
        if not all_consistent:
            print(f"  ⚠️  WARNING: Same input produces different outputs!")
        
        # Overall assessment
        is_sensitive = (
            stats_random['unique_predictions'] > 1 and
            stats_random['mean_variance'] > 1e-6 and
            extreme_differ
        )
        
        result = {
            'model_path': str(model_path),
            'input_dim': int(input_dim),
            'num_samples': num_samples,
            'random_inputs_stats': stats_random,
            'incremental_inputs_stats': stats_incremental,
            'extreme_inputs_differ': extreme_differ,
            'consistent_on_same_input': all_consistent,
            'is_sensitive': is_sensitive
        }
        
        print(f"\n[Conclusion]")
        if is_sensitive:
            print(f"  ✅ Model IS sensitive to input changes")
        else:
            print(f"  ❌ Model appears to be INSENSITIVE to input changes")
        
        return result
        
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return {'error': str(e)}


def main():
    """Run sensitivity tests on models"""
    root = Path(__file__).resolve().parents[1]
    
    models_to_test = [
        root / "unified" / "models" / "expressora_unified.tflite",
        root / "Tensorflow" / "TFModels" / "ASL_Alphabet_TF_Model_SavedModel" / "ASL_Alphabet_TF_Model.tflite",
        root / "Tensorflow" / "TFModelsFSL" / "FSL_Alphabet_TF_Model_SavedModel" / "FSL_Alphabet_TF_Model.tflite",
    ]
    
    results = []
    
    for model_path in models_to_test:
        if model_path.exists():
            result = test_input_sensitivity(model_path, num_samples=50)
            results.append(result)
        else:
            print(f"\n⚠️  Skipping {model_path.name} (not found)")
    
    # Summary
    print(f"\n{'='*60}")
    print("SENSITIVITY TEST SUMMARY")
    print(f"{'='*60}")
    
    sensitive_count = sum(1 for r in results if r.get('is_sensitive', False))
    total_count = len(results)
    
    print(f"\nSensitive models: {sensitive_count}/{total_count}")
    
    for result in results:
        if 'error' in result:
            print(f"  ❌ {Path(result['model_path']).name}: ERROR")
        elif result.get('is_sensitive', False):
            stats = result['random_inputs_stats']
            print(f"  ✅ {Path(result['model_path']).name}: SENSITIVE")
            print(f"     → {stats['unique_predictions']} unique predictions, "
                  f"variance={stats['mean_variance']:.6f}")
        else:
            stats = result.get('random_inputs_stats', {})
            print(f"  ❌ {Path(result['model_path']).name}: INSENSITIVE")
            if stats:
                print(f"     → {stats['unique_predictions']} unique predictions, "
                      f"variance={stats['mean_variance']:.6f}")
    
    # Save results
    results_file = root / "tools" / "sensitivity_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return 0 if sensitive_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())

