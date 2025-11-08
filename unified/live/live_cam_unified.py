"""
Live webcam inference for Expressora Sign Language Recognition
Uses MediaPipe Hands → 126-dim features → TFLite classifier → on-screen overlay
"""
import json
import os
import time
from collections import deque
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf


# Constants
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "unified" / "models" / "expressora_unified.tflite"
LABELS_PATH = ROOT / "unified" / "models" / "expressora_labels.json"
LABEL_ORIGIN_STATS_PATH = ROOT / "unified" / "data" / "label_origin_stats.json"
ORIGIN_LABELS_PATH = ROOT / "unified" / "data" / "origin_labels.json"

ONE_HAND_DIM = 21 * 3  # 63 floats per hand
TWO_HAND_DIM = ONE_HAND_DIM * 2  # 126 floats total

SMOOTHING_WINDOW = 5  # frames for logit averaging
DEBOUNCE_THRESHOLD = 3  # consecutive frames needed to change label

# Confidence thresholds (can be overridden via environment variables)
CONF_THRESHOLD = float(os.environ.get('CONF_THRESHOLD', '0.65'))  # Top-1 must exceed this
HOLD_FRAMES = int(os.environ.get('HOLD_FRAMES', '3'))  # Stable frames before update

# Alphabet mode configuration
ALPHABET_MODE = os.environ.get('ALPHABET_MODE', 'false').lower() == 'true'
IDLE_TIMEOUT_MS = int(os.environ.get('IDLE_TIMEOUT_MS', '1000'))  # Commit word after idle

# Origin display configuration
SHOW_ORIGIN = os.environ.get('SHOW_ORIGIN', 'true').lower() == 'true'
ORIGIN_CONF_THRESHOLD = float(os.environ.get('ORIGIN_CONF_THRESHOLD', '0.70'))


class LiveInference:
    """Real-time sign language recognition from webcam"""
    
    def __init__(self):
        self.labels = self._load_labels()
        self.interpreter = self._load_model()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        
        # Smoothing & debouncing state
        self.logits_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.prediction_history = deque(maxlen=HOLD_FRAMES)
        self.current_label = ""
        self.current_confidence = 0.0
        self.conf_threshold = CONF_THRESHOLD
        self.hold_frames = HOLD_FRAMES
        
        # Alphabet accumulator state
        self.alphabet_mode = ALPHABET_MODE
        self.idle_timeout_ms = IDLE_TIMEOUT_MS
        self.current_word = ""
        self.committed_text = ""
        self.last_letter_time = 0
        self.last_letter = ""
        
        # Origin tracking state
        self.show_origin = SHOW_ORIGIN
        self.origin_conf_threshold = ORIGIN_CONF_THRESHOLD
        self.has_origin_head = self._check_origin_head()
        self.origin_labels = self._load_origin_labels()
        self.label_origin_priors = self._load_label_origin_priors()
        self.current_origin = ""
        self.current_origin_confidence = 0.0
        self.origin_logits_buffer = deque(maxlen=SMOOTHING_WINDOW) if self.has_origin_head else None
        
        # FPS tracking
        self.frame_times = deque(maxlen=30)
        
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def _load_labels(self):
        """Load label list from JSON, or synthesize if missing"""
        try:
            if LABELS_PATH.exists():
                with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                    if labels:
                        print(f"Loaded {len(labels)} labels from {LABELS_PATH}")
                        return labels
        except Exception as e:
            print(f"Warning: Could not load labels from {LABELS_PATH}: {e}")
        
        # Fallback: synthesize labels based on model output size
        print("Labels missing or empty - will synthesize after loading model")
        return None
    
    def _load_model(self):
        """Load TFLite interpreter"""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                f"Run: .\\scripts\\run_unified.ps1 to train and export the model"
            )
        
        print(f"Loading model from {MODEL_PATH}")
        interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        interpreter.allocate_tensors()
        
        # If labels weren't loaded, synthesize based on output shape
        if self.labels is None:
            output_shape = interpreter.get_output_details()[0]['shape']
            num_classes = output_shape[-1]
            self.labels = [f"CLASS_{i}" for i in range(num_classes)]
            print(f"Synthesized {num_classes} labels: CLASS_0 to CLASS_{num_classes-1}")
        
        return interpreter
    
    def _check_origin_head(self):
        """Check if model has multiple outputs (origin head)"""
        output_details = self.interpreter.get_output_details()
        return len(output_details) >= 2
    
    def _load_origin_labels(self):
        """Load origin labels (e.g., ["ASL", "FSL"])"""
        if not ORIGIN_LABELS_PATH.exists():
            return None
        try:
            with open(ORIGIN_LABELS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def _load_label_origin_priors(self):
        """Load label→origin statistics for fallback"""
        if not LABEL_ORIGIN_STATS_PATH.exists():
            return None
        try:
            with open(LABEL_ORIGIN_STATS_PATH, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                # Convert to priors: for each label, compute majority origin
                priors = {}
                for label, origin_counts in stats.items():
                    if origin_counts:
                        majority_origin = max(origin_counts.items(), key=lambda x: x[1])[0]
                        priors[label] = majority_origin
                return priors
        except Exception:
            return None
    
    def extract_features(self, results):
        """
        Extract 126-dim feature vector from MediaPipe Hands results
        
        Returns:
            np.ndarray of shape (126,) or None if no hands detected
        """
        if not results.multi_hand_landmarks:
            return None
        
        # Collect all detected hands
        hands_data = []
        for hand_landmarks in results.multi_hand_landmarks:
            hand_features = []
            for landmark in hand_landmarks.landmark:
                hand_features.extend([landmark.x, landmark.y, landmark.z])
            hands_data.append(hand_features)
        
        # Handle 1 or 2 hands
        if len(hands_data) == 1:
            # One hand: pad second hand with zeros
            features = hands_data[0] + [0.0] * ONE_HAND_DIM
        else:
            # Two hands: concatenate first two
            features = hands_data[0] + hands_data[1]
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, features):
        """
        Run inference and return smoothed probabilities
        
        Args:
            features: np.ndarray of shape (126,)
            
        Returns:
            tuple: (label, confidence, top5_list)
        """
        # Reshape for batch dimension
        input_data = features.reshape(1, -1)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details['index'], input_data)
        self.interpreter.invoke()
        
        # Get gloss logits (first output)
        logits = self.interpreter.get_tensor(self.output_details['index'])[0]
        
        # Get origin logits if multi-output model
        if self.has_origin_head and len(self.interpreter.get_output_details()) >= 2:
            origin_output_details = self.interpreter.get_output_details()[1]
            origin_logits = self.interpreter.get_tensor(origin_output_details['index'])[0]
            
            # Add to origin smoothing buffer
            self.origin_logits_buffer.append(origin_logits)
            
            # Smooth origin logits
            if len(self.origin_logits_buffer) > 0:
                smoothed_origin_logits = np.mean(self.origin_logits_buffer, axis=0)
                origin_idx = int(np.argmax(smoothed_origin_logits))
                origin_conf = float(smoothed_origin_logits[origin_idx])
                
                if origin_conf >= self.origin_conf_threshold and self.origin_labels:
                    self.current_origin = self.origin_labels[origin_idx]
                    self.current_origin_confidence = origin_conf
                else:
                    self.current_origin = "UNKNOWN"
                    self.current_origin_confidence = origin_conf
        
        # Add to smoothing buffer
        self.logits_buffer.append(logits)
        
        # Average logits over recent frames
        if len(self.logits_buffer) > 0:
            smoothed_logits = np.mean(self.logits_buffer, axis=0)
        else:
            smoothed_logits = logits
        
        # Get top-1 prediction
        top1_idx = int(np.argmax(smoothed_logits))
        confidence = float(smoothed_logits[top1_idx])
        
        # Apply confidence threshold
        if confidence < self.conf_threshold:
            predicted_label = "UNKNOWN"
        else:
            predicted_label = self.labels[top1_idx]
        
        # Debounce: only update displayed label if consistent
        self.prediction_history.append((top1_idx, confidence))
        if len(self.prediction_history) == self.hold_frames:
            # Check if all recent predictions are the same AND above threshold
            recent_indices = [idx for idx, conf in self.prediction_history]
            recent_confs = [conf for idx, conf in self.prediction_history]
            
            if all(idx == top1_idx for idx in recent_indices):
                avg_conf = np.mean(recent_confs)
                if avg_conf >= self.conf_threshold:
                    self.current_label = self.labels[top1_idx]
                    self.current_confidence = avg_conf
                else:
                    self.current_label = "UNKNOWN"
                    self.current_confidence = avg_conf
        
        # Get top-5
        top5_indices = np.argsort(smoothed_logits)[-5:][::-1]
        top5_list = [
            (self.labels[i], float(smoothed_logits[i]))
            for i in top5_indices
        ]
        
        return self.current_label, self.current_confidence, top5_list
    
    def is_alphabet_letter(self, label):
        """Check if label is a single alphabet letter"""
        return len(label) == 1 and label.isalpha()
    
    def estimate_origin_from_prior(self, label):
        """Estimate origin from label priors (fallback when no origin head)"""
        if not self.label_origin_priors or label == "UNKNOWN":
            return ""
        
        prior_origin = self.label_origin_priors.get(label)
        if prior_origin:
            return f"{prior_origin}~"  # ~ indicates prior-based estimate
        return ""
    
    def handle_alphabet_accumulation(self, label, current_time_ms):
        """
        Handle alphabet letter accumulation into words.
        
        Args:
            label: Current predicted label
            current_time_ms: Current timestamp in milliseconds
        """
        if not self.alphabet_mode or label == "UNKNOWN":
            return
        
        # Check for backspace label (if it exists in the dataset)
        if label.lower() in ["backspace", "delete", "back"]:
            if self.current_word:
                self.current_word = self.current_word[:-1]
                self.last_letter_time = current_time_ms
            return
        
        # Check if it's an alphabet letter
        if self.is_alphabet_letter(label):
            # Avoid duplicating the same letter rapidly
            if label != self.last_letter or (current_time_ms - self.last_letter_time) > 500:
                self.current_word += label
                self.last_letter = label
                self.last_letter_time = current_time_ms
        
        # Check for idle timeout to commit word
        if self.current_word and (current_time_ms - self.last_letter_time) > self.idle_timeout_ms:
            # Commit the word
            if self.committed_text:
                self.committed_text += " " + self.current_word
            else:
                self.committed_text = self.current_word
            self.current_word = ""
            self.last_letter = ""
    
    def draw_overlay(self, frame, num_hands, label, confidence, top5, fps):
        """Draw prediction overlay on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay background
        overlay = frame.copy()
        
        # Top section: Hand count and main prediction
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Hand count
        hand_text = f"Hands: {num_hands}"
        cv2.putText(frame, hand_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Main prediction (large text)
        if label:
            pred_text = f"{label} ({confidence*100:.1f}%)"
            cv2.putText(frame, pred_text, (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Origin badge (if enabled)
            if self.show_origin:
                origin_display = self.current_origin
                if not origin_display and label != "UNKNOWN":
                    # Fallback to prior-based estimate
                    origin_display = self.estimate_origin_from_prior(label)
                
                if origin_display:
                    origin_text = f"Origin: {origin_display}"
                    # Position to the right of main prediction or below
                    cv2.putText(frame, origin_text, (20, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        else:
            cv2.putText(frame, "Waiting for stable prediction...", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Top-5 list in bottom-left corner
        if top5:
            y_offset = h - 200
            cv2.rectangle(overlay, (10, y_offset - 30), (350, h - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, "Top-5:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for i, (lbl, conf) in enumerate(top5):
                text = f"{i+1}. {lbl} ({conf*100:.1f}%)"
                cv2.putText(frame, text, (20, y_offset + 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # FPS in top-right corner
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Alphabet mode: show current word and committed text
        if self.alphabet_mode:
            y_text = h - 100
            cv2.rectangle(overlay, (10, y_text - 30), (w - 10, h - 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Current word being typed
            if self.current_word:
                word_text = f"Word: {self.current_word}_"
                cv2.putText(frame, word_text, (20, y_text),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Committed text
            if self.committed_text:
                committed_text = f"Text: {self.committed_text}"
                cv2.putText(frame, committed_text, (20, y_text + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Instructions at bottom
        instr_text = "Press ESC to quit"
        cv2.putText(frame, instr_text, (w - 250, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        """Main loop: capture frames, process, display"""
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError(
                "Cannot open webcam (index 0).\n"
                "Check that your camera is connected and not in use by another application."
            )
        
        print("\n" + "="*50)
        print("Expressora Live Sign Language Recognition")
        print("="*50)
        print(f"Model: {MODEL_PATH.name}")
        print(f"Classes: {len(self.labels)}")
        print(f"Confidence Threshold: {self.conf_threshold:.2f}")
        print(f"Hold Frames: {self.hold_frames}")
        if self.alphabet_mode:
            print(f"Alphabet Mode: ENABLED (idle timeout: {self.idle_timeout_ms}ms)")
        if self.show_origin:
            if self.has_origin_head:
                print(f"Origin Display: ENABLED (multi-output model)")
            elif self.label_origin_priors:
                print(f"Origin Display: ENABLED (prior-based fallback)")
            else:
                print(f"Origin Display: DISABLED (no origin data available)")
        print("\nControls:")
        print("  ESC - Quit")
        print("\nStarting camera...")
        print("="*50 + "\n")
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Draw hand landmarks
                num_hands = 0
                if results.multi_hand_landmarks:
                    num_hands = len(results.multi_hand_landmarks)
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                
                # Extract features and predict
                label = self.current_label
                confidence = self.current_confidence
                top5 = []
                
                if num_hands > 0:
                    features = self.extract_features(results)
                    if features is not None:
                        label, confidence, top5 = self.predict(features)
                        
                        # Handle alphabet accumulation if enabled
                        current_time_ms = int(time.time() * 1000)
                        self.handle_alphabet_accumulation(label, current_time_ms)
                else:
                    # No hands: clear buffers
                    self.logits_buffer.clear()
                    self.prediction_history.clear()
                
                # Calculate FPS
                frame_time = time.time() - start_time
                self.frame_times.append(frame_time)
                fps = len(self.frame_times) / sum(self.frame_times)
                
                # Draw overlay
                self.draw_overlay(frame, num_hands, label, confidence, top5, fps)
                
                # Display
                cv2.imshow('Expressora Sign Language Recognition', frame)
                
                # Check for ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\nExiting...")
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("Camera released. Goodbye!")


def main():
    """Entry point"""
    try:
        inference = LiveInference()
        inference.run()
    except FileNotFoundError as e:
        print(f"\nError: {e}\n")
        return 1
    except RuntimeError as e:
        print(f"\nError: {e}\n")
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 0
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

