"""
Export FP16 quantized TFLite model.
Reduces model size by ~50% with minimal accuracy loss.
"""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "unified" / "models"
DATA_DIR = ROOT / "unified" / "data"

def main():
    saved_dir = MODELS_DIR / "savedmodel"
    tflite_out = MODELS_DIR / "expressora_unified_fp16.tflite"
    
    if not saved_dir.exists():
        raise FileNotFoundError(
            f"SavedModel not found: {saved_dir}\n"
            f"Run training first: .\\scripts\\run_unified.ps1"
        )
    
    print("="*60)
    print("Exporting FP16 Quantized TFLite Model")
    print("="*60)
    print(f"\nInput: {saved_dir}")
    print(f"Output: {tflite_out}")
    
    # Convert with FP16 quantization
    print("\nConverting to FP16...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir.as_posix())
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    tflite_out.write_bytes(tflite_model)
    
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\n✓ Exported FP16 model: {tflite_out}")
    print(f"  Size: {size_mb:.2f} MB")
    
    # Quick sanity check
    print("\nRunning sanity check...")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_out))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    
    print(f"  Input shape: {input_details['shape']}")
    print(f"  Number of outputs: {len(output_details)}")
    for i, out in enumerate(output_details):
        print(f"    Output {i} shape: {out['shape']}")
    
    # Test inference on a dummy sample
    dummy_input = np.random.randn(*input_details['shape']).astype(np.float32)
    interpreter.set_tensor(input_details['index'], dummy_input)
    interpreter.invoke()
    
    print("\n✓ Sanity check passed - model can run inference")
    print("\n" + "="*60)
    print("FP16 Export Complete!")
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

