import json
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

saved_dir = MODELS / "savedmodel"
tflite_out = MODELS / "expressora_unified.tflite"
labels_src = DATA / "labels.json"
labels_out = MODELS / "expressora_labels.json"
signature_out = MODELS / "model_signature.json"

print("="*70)
print("EXPORTING TFLITE MODELS")
print("="*70)

# 1. Float32 TFLite (no quantization)
print("\n[1/3] Converting to Float32 TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir.as_posix())
# Enable SELECT_TF_OPS for LSTM support
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Try TFLite first
    tf.lite.OpsSet.SELECT_TF_OPS      # Fallback to TF for LSTM ops
]
converter.allow_custom_ops = True
# Disable experimental lowering that causes TensorListReserve errors
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()
tflite_out.write_bytes(tflite_model)
print(f"  Saved: {tflite_out.name}")

# 2. FP16 quantized
print("\n[2/3] Converting to FP16 quantized TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir.as_posix())
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
# Enable SELECT_TF_OPS for LSTM support
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Try TFLite first
    tf.lite.OpsSet.SELECT_TF_OPS      # Fallback to TF for LSTM ops
]
converter.allow_custom_ops = True
# Disable experimental lowering that causes TensorListReserve errors
converter._experimental_lower_tensor_list_ops = False
tflite_fp16 = converter.convert()
tflite_fp16_out = MODELS / "expressora_unified_fp16.tflite"
tflite_fp16_out.write_bytes(tflite_fp16)
print(f"  Saved: {tflite_fp16_out.name}")

# 3. INT8 quantized (requires representative dataset)
print("\n[3/3] Converting to INT8 quantized TFLite...")
def representative_dataset():
    # Load sequences for quantization calibration (shape: N_sequences, seq_length, num_features)
    X = np.load(DATA / "unified_X.npy")
    # Verify shape is 3D (sequences)
    if len(X.shape) != 3:
        raise ValueError(f"Expected 3D array (N_sequences, seq_length, num_features), got shape {X.shape}")
    
    # Load normalization parameters (same as used in training)
    # The model expects normalized inputs, so representative dataset must also be normalized
    X_mean = np.load(DATA / "feature_mean.npy")  # Shape: (NUM_FEATURES,)
    X_std = np.load(DATA / "feature_std.npy")     # Shape: (NUM_FEATURES,)
    
    # Normalize the data (same normalization as training)
    X_normalized = (X - X_mean) / (X_std + 1e-8)
    
    # Use first 100 sequences for calibration
    for i in range(min(100, len(X_normalized))):
        # Yield sequence with batch dimension: [1, seq_length, num_features]
        yield [X_normalized[i:i+1].astype(np.float32)]  # Shape: [1, seq_length, num_features]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir.as_posix())
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# Enable SELECT_TF_OPS for LSTM support
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Try TFLite first
    tf.lite.OpsSet.SELECT_TF_OPS      # Fallback to TF for LSTM ops
]
converter.allow_custom_ops = True
# Disable experimental lowering that causes TensorListReserve errors
converter._experimental_lower_tensor_list_ops = False
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
try:
    tflite_int8 = converter.convert()
    tflite_int8_out = MODELS / "expressora_unified_int8.tflite"
    tflite_int8_out.write_bytes(tflite_int8)
    print(f"  Saved: {tflite_int8_out.name}")
except Exception as e:
    print(f"  Warning: INT8 quantization failed (LSTM quantization may be problematic): {e}")
    print(f"  Skipping INT8 export. Use FP16 or Float32 instead.")

print("\n" + "="*70)

# copy labels
labels = json.loads(labels_src.read_text(encoding="utf-8"))
labels_out.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

# Generate model signature
print("\nGenerating model signature...")
interpreter = tf.lite.Interpreter(model_path=str(tflite_out))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

signature = {
    "model_type": "float32",
    "inputs": [
        {
            "name": detail.get('name', f"input_{i}"),
            "index": int(detail['index']),
            "shape": [int(d) for d in detail['shape']],
            "dtype": str(detail['dtype'])
        }
        for i, detail in enumerate(input_details)
    ],
    "outputs": [
        {
            "name": detail.get('name', 'gloss_logits' if i == 0 else 'origin_logits'),
            "index": int(detail['index']),
            "shape": [int(d) for d in detail['shape']],
            "dtype": str(detail['dtype'])
        }
        for i, detail in enumerate(output_details)
    ]
}

with open(signature_out, 'w', encoding='utf-8') as f:
    json.dump(signature, f, indent=2, ensure_ascii=False)

print("Wrote:", tflite_out)
print("Wrote:", labels_out)
print("Wrote:", signature_out)
print(f"\nModel has {len(output_details)} output(s)")

