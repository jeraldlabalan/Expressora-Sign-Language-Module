import json
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

X = np.load(DATA / "unified_X.npy")
y = np.load(DATA / "unified_y.npy")
num_classes = int(y.max()) + 1

# train/val split
n = len(y)
split = int(n * 0.9)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Normalization(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

# adapt normalization
model.layers[0].adapt(X_train)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

cb = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint((MODELS/"expressora_unified.keras").as_posix(), save_best_only=True),
]

hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=256,
    callbacks=cb,
    verbose=2,
)

# Save SavedModel dir for TFLite export
saved_dir = MODELS / "savedmodel"
tf.saved_model.save(model, saved_dir.as_posix())

with open((MODELS/"training_log.json"), "w", encoding="utf-8") as f:
    json.dump(hist.history, f, ensure_ascii=False, indent=2)

print("Saved:", MODELS / "expressora_unified.keras")
print("Saved:", saved_dir)

