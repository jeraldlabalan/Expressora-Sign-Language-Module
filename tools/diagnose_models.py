"""
Diagnostic script to test if TFLite models respond to different inputs.
Checks if models are actually learning or if outputs are static.
"""
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
import sys

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

def test_model_responsiveness(model_path, num_tests=10, input_dim=126):
    """
    Test if a TFLite model produces different outputs for different inputs.
    
    Args:
        model_path: Path to .tflite model file
        num_tests: Number of different inputs to test
        input_dim: Expected input dimension
        
    Returns:
        dict with diagnostic results
    """
    print(f"\n{'='*60}")
    print(f"Testing model: {model_path.name}")
    print(f"{'='*60}")
    
    if not model_path.exists():
        return {
            'error': f'Model not found: {model_path}',
            'responsive': False
        }
    
    try:
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        print(f"Input shape: {input_details['shape']}")
        print(f"Input dtype: {input_details['dtype']}")
        print(f"Output shape: {output_details['shape']}")
        print(f"Output dtype: {output_details['dtype']}")
        
        # Get actual input dimension from model
        actual_input_dim = input_details['shape'][1] if len(input_details['shape']) > 1 else input_details['shape'][0]
        print(f"Actual input dimension: {actual_input_dim}")
        
        # Test 1: Generate multiple random inputs
        print(f"\n[Test 1] Generating {num_tests} random inputs...")
        test_inputs = []
        test_outputs = []
        test_predictions = []
        
        for i in range(num_tests):
            # Generate random input with proper shape
            if len(input_details['shape']) == 2:
                x = np.random.rand(1, actual_input_dim).astype(input_details['dtype'])
            else:
                x = np.random.rand(actual_input_dim).astype(input_details['dtype'])
            
            test_inputs.append(x.copy())
            
            # Set tensor and invoke
            interpreter.set_tensor(input_details['index'], x)
            interpreter.invoke()
            
            # Get output
            output = interpreter.get_tensor(output_details['index'])
            if len(output.shape) > 1:
                output = output[0]  # Remove batch dimension if present
            
            test_outputs.append(output.copy())
            test_predictions.append(int(np.argmax(output)))
            
            print(f"  Input {i+1}: max={x.max():.4f}, min={x.min():.4f}, mean={x.mean():.4f}")
            print(f"    Output: pred_class={test_predictions[-1]}, max_prob={output.max():.4f}")
        
        # Analyze results
        print(f"\n[Analysis]")
        
        # Check if predictions vary
        unique_predictions = len(set(test_predictions))
        print(f"  Unique predictions: {unique_predictions} out of {num_tests}")
        
        # Check output variance
        output_array = np.array(test_outputs)
        output_variance = np.var(output_array, axis=0)
        mean_variance = np.mean(output_variance)
        max_variance = np.max(output_variance)
        
        print(f"  Output variance (mean): {mean_variance:.6f}")
        print(f"  Output variance (max): {max_variance:.6f}")
        
        # Check if outputs are identical
        all_same = all(np.allclose(test_outputs[0], out, atol=1e-6) for out in test_outputs[1:])
        print(f"  All outputs identical: {all_same}")
        
        # Check if predictions are identical
        all_same_pred = len(set(test_predictions)) == 1
        print(f"  All predictions identical: {all_same_pred}")
        
        # Test 2: Verify input tensor is actually being set
        print(f"\n[Test 2] Verifying input tensor is updated...")
        
        # Create two very different inputs
        input1 = np.ones((1, actual_input_dim), dtype=input_details['dtype']) * 0.1
        input2 = np.ones((1, actual_input_dim), dtype=input_details['dtype']) * 0.9
        
        interpreter.set_tensor(input_details['index'], input1)
        interpreter.invoke()
        output1 = interpreter.get_tensor(output_details['index'])
        if len(output1.shape) > 1:
            output1 = output1[0]
        pred1 = int(np.argmax(output1))
        
        interpreter.set_tensor(input_details['index'], input2)
        interpreter.invoke()
        output2 = interpreter.get_tensor(output_details['index'])
        if len(output2.shape) > 1:
            output2 = output2[0]
        pred2 = int(np.argmax(output2))
        
        outputs_differ = not np.allclose(output1, output2, atol=1e-4)
        predictions_differ = pred1 != pred2
        
        print(f"  Input 1 (all 0.1): pred={pred1}, max_prob={output1.max():.4f}")
        print(f"  Input 2 (all 0.9): pred={pred2}, max_prob={output2.max():.4f}")
        print(f"  Outputs differ: {outputs_differ}")
        print(f"  Predictions differ: {predictions_differ}")
        
        # Test 3: Check output entropy (higher = more varied)
        print(f"\n[Test 3] Calculating output entropy...")
        avg_output = np.mean(output_array, axis=0)
        # Normalize to probabilities
        avg_probs = avg_output / (avg_output.sum() + 1e-10)
        entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-10))
        max_entropy = np.log(len(avg_probs))
        normalized_entropy = entropy / max_entropy
        
        print(f"  Average output entropy: {entropy:.4f}")
        print(f"  Max possible entropy: {max_entropy:.4f}")
        print(f"  Normalized entropy: {normalized_entropy:.4f} (1.0 = uniform, 0.0 = deterministic)")
        
        # Determine if model is responsive
        is_responsive = (
            unique_predictions > 1 and
            not all_same and
            outputs_differ and
            mean_variance > 1e-6
        )
        
        result = {
            'model_path': str(model_path),
            'input_dim': actual_input_dim,
            'output_dim': len(test_outputs[0]),
            'num_tests': num_tests,
            'unique_predictions': unique_predictions,
            'all_outputs_identical': all_same,
            'all_predictions_identical': all_same_pred,
            'output_variance_mean': float(mean_variance),
            'output_variance_max': float(max_variance),
            'outputs_differ_on_extreme_inputs': outputs_differ,
            'predictions_differ_on_extreme_inputs': predictions_differ,
            'normalized_entropy': float(normalized_entropy),
            'responsive': is_responsive,
            'test_predictions': [int(p) for p in test_predictions]
        }
        
        print(f"\n[Conclusion]")
        if is_responsive:
            print(f"  [OK] Model IS responsive to input changes")
        else:
            print(f"  [FAIL] Model appears to be STATIC or not learning")
            if all_same_pred:
                print(f"     → Always predicts the same class")
            if all_same:
                print(f"     → Outputs are identical regardless of input")
            if mean_variance < 1e-6:
                print(f"     → Output variance is extremely low")
        
        return result
        
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Error testing model: {e}")
        traceback.print_exc()
        return {
            'error': str(e),
            'responsive': False
        }


def main():
    """Test multiple models"""
    root = Path(__file__).resolve().parents[1]
    
    # Models to test
    models_to_test = [
        root / "unified" / "models" / "expressora_unified.tflite",
        root / "Tensorflow" / "TFModels" / "ASL_Alphabet_TF_Model_SavedModel" / "ASL_Alphabet_TF_Model.tflite",
        root / "Tensorflow" / "TFModelsFSL" / "FSL_Alphabet_TF_Model_SavedModel" / "FSL_Alphabet_TF_Model.tflite",
    ]
    
    results = []
    
    for model_path in models_to_test:
        if model_path.exists():
            result = test_model_responsiveness(model_path, num_tests=20)
            results.append(result)
        else:
            print(f"\n[WARN] Skipping {model_path.name} (not found)")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    responsive_count = sum(1 for r in results if r.get('responsive', False))
    total_count = len(results)
    
    print(f"\nResponsive models: {responsive_count}/{total_count}")
    
    for result in results:
        if 'error' in result:
            print(f"  [ERROR] {Path(result['model_path']).name}: ERROR - {result['error']}")
        elif result.get('responsive', False):
            print(f"  [OK] {Path(result['model_path']).name}: RESPONSIVE")
        else:
            print(f"  [FAIL] {Path(result['model_path']).name}: STATIC")
            if result.get('all_predictions_identical', False):
                print(f"     → Always predicts class {result['test_predictions'][0]}")
    
    # Save results
    results_file = root / "tools" / "diagnostic_results.json"
    with open(results_file, 'w') as f:
        json.dump(convert_to_native(results), f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return 0 if responsive_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())

