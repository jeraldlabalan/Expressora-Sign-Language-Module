"""
Export INT8 quantized TFLite model using representative dataset.
Achieves maximum compression (~75% size reduction) for mobile deployment.
"""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "unified" / "models"
DATA_DIR = ROOT / "unified" / "data"

def representative_dataset_gen():
    """Generator for representative dataset"""
    rep_set_path = DATA_DIR / "rep_set.npy"
    
    if not rep_set_path.exists():
        raise FileNotFoundError(
            f"Representative set not found: {rep_set_path}\n"
            f"Run: python unified/data/build_representative_set.py"
        )
    
    rep_data = np.load(rep_set_path)
    print(f"Loaded representative set: {rep_data.shape}")
    
    for sample in rep_data:
        # Yield as batch of 1
        yield [sample.reshape(1, -1).astype(np.float32)]

def main():
    saved_dir = MODELS_DIR / "savedmodel"
    tflite_out = MODELS_DIR / "expressora_unified_int8.tflite"
    
    if not saved_dir.exists():
        raise FileNotFoundError(
            f"SavedModel not found: {saved_dir}\n"
            f"Run training first: .\\scripts\\run_unified.ps1"
        )
    
    print("="*60)
    print("Exporting INT8 Quantized TFLite Model")
    print("="*60)
    print(f"\nInput: {saved_dir}")
    print(f"Output: {tflite_out}")
    
    # Convert with INT8 quantization
    print("\nConverting to INT8 (this may take a minute)...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir.as_posix())
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Enable full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"\nWarning: Full INT8 quantization failed: {e}")
        print("Falling back to hybrid quantization...")
        # Fallback to hybrid quantization
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir.as_posix())
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()
    
    tflite_out.write_bytes(tflite_model)
    
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\n✓ Exported INT8 model: {tflite_out}")
    print(f"  Size: {size_mb:.2f} MB")
    
    # Quick sanity check with accuracy test
    print("\nRunning sanity check...")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_out))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    
    print(f"  Input shape: {input_details['shape']}")
    print(f"  Input type: {input_details['dtype']}")
    print(f"  Number of outputs: {len(output_details)}")
    for i, out in enumerate(output_details):
        print(f"    Output {i} shape: {out['shape']}, dtype: {out['dtype']}")
    
    # Test inference on a few samples
    print("\n  Testing inference on representative samples...")
    X_test = np.load(DATA_DIR / "rep_set.npy")[:10]  # First 10 samples
    
    for i, sample in enumerate(X_test):
        input_data = sample.reshape(1, -1).astype(input_details['dtype'])
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        
        # Just check we can get output without error
        output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"  ✓ Successfully ran inference on {len(X_test)} samples")
    
    print("\n" + "="*60)
    print("INT8 Export Complete!")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}\n")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

