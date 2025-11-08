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

# TFLite conversion (dynamic range quantization)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir.as_posix())
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
tflite_out.write_bytes(tflite_model)

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

