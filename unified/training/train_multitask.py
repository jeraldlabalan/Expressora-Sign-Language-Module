"""
Multi-task training: Gloss classification + Origin classification
Trains a model with two heads sharing a common backbone.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "unified" / "data"
MODELS = ROOT / "unified" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train multi-task model')
    parser.add_argument('--lambda-origin', type=float, default=0.3,
                        help='Weight for origin loss (default: 0.3)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    return parser.parse_args()

def load_data():
    """Load unified dataset with origin labels"""
    X = np.load(DATA / "unified_X.npy")
    y_gloss = np.load(DATA / "unified_y.npy")
    y_origin = np.load(DATA / "unified_origin.npy")
    
    with open(DATA / "labels.json", 'r', encoding='utf-8') as f:
        gloss_labels = json.load(f)
    
    with open(DATA / "origin_labels.json", 'r', encoding='utf-8') as f:
        origin_labels = json.load(f)
    
    num_gloss_classes = len(gloss_labels)
    num_origin_classes = len(origin_labels)
    
    # Split data
    n = len(y_gloss)
    split = int(n * 0.9)
    
    X_train, X_val = X[:split], X[split:]
    y_gloss_train, y_gloss_val = y_gloss[:split], y_gloss[split:]
    y_origin_train, y_origin_val = y_origin[:split], y_origin[split:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Gloss classes: {num_gloss_classes}")
    print(f"Origin classes: {num_origin_classes}")
    
    return (X_train, y_gloss_train, y_origin_train,
            X_val, y_gloss_val, y_origin_val,
            num_gloss_classes, num_origin_classes)

def build_multitask_model(input_dim, num_gloss_classes, num_origin_classes):
    """
    Build multi-task model with shared backbone and two heads.
    
    Architecture:
        Input → Normalization → Dense(256) → Dense(128) → [Gloss Head, Origin Head]
    """
    # Input
    inputs = tf.keras.layers.Input(shape=(input_dim,), name='input')
    
    # Shared backbone
    x = tf.keras.layers.Normalization(name='normalization')(inputs)
    x = tf.keras.layers.Dense(256, activation='relu', name='backbone_dense1')(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='backbone_dense2')(x)
    
    # Gloss classification head
    gloss_output = tf.keras.layers.Dense(
        num_gloss_classes, activation='softmax', name='gloss_output'
    )(x)
    
    # Origin classification head
    origin_output = tf.keras.layers.Dense(
        num_origin_classes, activation='softmax', name='origin_output'
    )(x)
    
    # Create model with two outputs
    model = tf.keras.Model(inputs=inputs, outputs=[gloss_output, origin_output])
    
    return model

def main():
    args = parse_args()
    
    print("="*60)
    print("Multi-Task Training: Gloss + Origin Classification")
    print("="*60)
    print(f"Lambda (origin weight): {args.lambda_origin}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Load data
    (X_train, y_gloss_train, y_origin_train,
     X_val, y_gloss_val, y_origin_val,
     num_gloss_classes, num_origin_classes) = load_data()
    
    # Build model
    model = build_multitask_model(X_train.shape[1], num_gloss_classes, num_origin_classes)
    
    # Adapt normalization layer
    model.get_layer('normalization').adapt(X_train)
    
    # Check for class imbalance in origin
    origin_counts = np.bincount(y_origin_train)
    if len(origin_counts) > 1:
        imbalance_ratio = max(origin_counts) / min(origin_counts)
        if imbalance_ratio > 1.5:
            print(f"\nOrigin class imbalance detected: {imbalance_ratio:.2f}:1")
            print("Applying class weighting to origin head...")
            # Compute class weights
            total = len(y_origin_train)
            origin_class_weight = {
                i: total / (len(origin_counts) * count)
                for i, count in enumerate(origin_counts)
            }
        else:
            origin_class_weight = None
    else:
        origin_class_weight = None
    
    # Compile with weighted losses
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            'gloss_output': 'sparse_categorical_crossentropy',
            'origin_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'gloss_output': 1.0,
            'origin_output': args.lambda_origin
        },
        metrics={
            'gloss_output': ['accuracy'],
            'origin_output': ['accuracy']
        }
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Prepare training data
    train_data = (
        X_train,
        {'gloss_output': y_gloss_train, 'origin_output': y_origin_train}
    )
    val_data = (
        X_val,
        {'gloss_output': y_gloss_val, 'origin_output': y_origin_val}
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            (MODELS / "expressora_unified.keras").as_posix(),
            save_best_only=True
        )
    ]
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        train_data[0], train_data[1],
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2
    )
    
    # Save history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(MODELS / "training_log.json", 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Export SavedModel
    model.save(MODELS / "savedmodel", save_format='tf')
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Keras model: {MODELS / 'expressora_unified.keras'}")
    print(f"SavedModel: {MODELS / 'savedmodel'}")
    print(f"Training log: {MODELS / 'training_log.json'}")
    
    # Print final metrics
    print("\nFinal Metrics:")
    for metric, values in history.history.items():
        if metric.startswith('val_'):
            print(f"  {metric}: {values[-1]:.4f}")

if __name__ == "__main__":
    main()

