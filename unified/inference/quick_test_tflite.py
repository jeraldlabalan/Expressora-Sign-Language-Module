import json
import numpy as np
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

interpreter = tf.lite.Interpreter(model_path=(MODELS/"expressora_unified.tflite").as_posix())
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

# random feature for smoke test (126-dim)
x = np.random.rand(1, inp["shape"][1]).astype(np.float32)
interpreter.set_tensor(inp["index"], x)
interpreter.invoke()
probs = interpreter.get_tensor(out["index"])[0]

labels = json.loads((MODELS/"expressora_labels.json").read_text(encoding="utf-8"))
pred = int(probs.argmax())
print("Top-1:", labels[pred], float(probs[pred]))

