"""
Training monitoring utilities to detect model collapse early.
"""
import numpy as np
import tensorflow as tf
from collections import Counter

class PredictionDiversityCallback(tf.keras.callbacks.Callback):
    """Monitor prediction diversity during training to detect collapse"""
    
    def __init__(self, X_val, y_val, check_every=5, min_unique_predictions=10):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.check_every = check_every
        self.min_unique_predictions = min_unique_predictions
        self.diversity_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.check_every == 0:
            # Get predictions on validation set
            y_pred_probs = self.model.predict(self.X_val, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Calculate diversity metrics
            unique_preds = len(np.unique(y_pred))
            pred_counts = Counter(y_pred)
            most_common = pred_counts.most_common(1)[0]
            most_common_freq = most_common[1] / len(y_pred)
            
            # Calculate entropy
            avg_probs = np.mean(y_pred_probs, axis=0)
            avg_probs = avg_probs / (avg_probs.sum() + 1e-10)
            entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-10))
            max_entropy = np.log(len(avg_probs))
            normalized_entropy = entropy / max_entropy
            
            self.diversity_history.append({
                'epoch': epoch,
                'unique_predictions': unique_preds,
                'most_common_class': int(most_common[0]),
                'most_common_frequency': float(most_common_freq),
                'normalized_entropy': float(normalized_entropy)
            })
            
            print(f"\n  [Diversity Check] Epoch {epoch}:")
            print(f"    Unique predictions: {unique_preds}/{len(self.y_val)}")
            print(f"    Most common class: {most_common[0]} ({most_common_freq:.1%})")
            print(f"    Normalized entropy: {normalized_entropy:.4f}")
            
            # Warning if collapsing
            if unique_preds < self.min_unique_predictions:
                print(f"    [WARNING] Model may be collapsing! Only {unique_preds} unique predictions.")
            
            if most_common_freq > 0.5:
                print(f"    [WARNING] Single class dominates {most_common_freq:.1%} of predictions!")
            
            if normalized_entropy < 0.2:
                print(f"    [WARNING] Low entropy ({normalized_entropy:.4f}) indicates low diversity!")


class PerClassAccuracyCallback(tf.keras.callbacks.Callback):
    """Track per-class accuracy during training"""
    
    def __init__(self, X_val, y_val, num_classes, check_every=10):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.num_classes = num_classes
        self.check_every = check_every
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.check_every == 0:
            y_pred_probs = self.model.predict(self.X_val, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Calculate per-class accuracy
            per_class_correct = np.zeros(self.num_classes)
            per_class_total = np.zeros(self.num_classes)
            
            for true_label, pred_label in zip(self.y_val, y_pred):
                per_class_total[true_label] += 1
                if true_label == pred_label:
                    per_class_correct[true_label] += 1
            
            per_class_acc = per_class_correct / (per_class_total + 1e-10)
            
            # Find worst performing classes
            valid_classes = per_class_total > 0
            if np.any(valid_classes):
                worst_classes = np.argsort(per_class_acc[valid_classes])[:5]
                worst_accs = per_class_acc[valid_classes][worst_classes]
                
                print(f"\n  [Per-Class Accuracy] Epoch {epoch}:")
                print(f"    Worst 5 classes: {worst_classes} with accuracies: {worst_accs}")
                print(f"    Mean per-class accuracy: {np.mean(per_class_acc[valid_classes]):.4f}")

