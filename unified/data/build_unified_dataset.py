import os, json, glob
from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
OUT_DIR = ROOT / "unified" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Heuristics: scan all CSVs that look like MediaPipe landmark dumps.
# Expect columns like L1_x0..L1_z20 (+ L2_* for second hand) + 'label'
CSV_GLOBS = [
    "AlphabetSignLanguages/**/*.csv",
    "ASLBasicPhrasesSignLanguages/**/*.csv",
    "FSLBasicPhrasesSignLanguages/**/*.csv",
    "ASLFacialExpressionsLanguages/**/*.csv",  # Face data with NMM
    "FSLFacialExpressionsLanguages/**/*.csv",  # Face data
    "Tensorflow/**/*.csv",
]

ONE_HAND_DIM = 21*3   # 63
TWO_HAND_DIM = ONE_HAND_DIM*2  # 126

# NMM (Non-Manual Markers) Face Landmark Indices for Sign Language Grammar
# Eyebrows (Grammar/Questions): 10 points
EYEBROW_INDICES = [46, 52, 53, 65, 70, 276, 282, 283, 295, 300]
# Lips (Mouthing/Adverbs): 27 points
LIP_INDICES = [0, 13, 14, 17, 37, 39, 40, 61, 80, 81, 82, 178, 181, 185, 191, 267, 269, 270, 291, 310, 311, 312, 318, 402, 405, 409, 415]
# Combined NMM face indices
FACE_INDICES = EYEBROW_INDICES + LIP_INDICES  # 37 points total
FACE_DIM = len(FACE_INDICES) * 3  # 111 features (37 points × 3 coords)

# Total feature dimension: hands + face
NUM_FEATURES = TWO_HAND_DIM + FACE_DIM  # 126 + 111 = 237

# Sequence length for temporal model
SEQ_LENGTH = 30

# Origin labels
ORIGIN_LABELS = ["ASL", "FSL"]
ORIGIN_ASL = 0
ORIGIN_FSL = 1
ORIGIN_UNKNOWN = -1

def infer_origin(path: Path) -> int:
    """
    Infer origin (ASL/FSL) from file path segments.
    
    Returns:
        ORIGIN_ASL (0), ORIGIN_FSL (1), or ORIGIN_UNKNOWN (-1)
    """
    path_str = path.as_posix().upper()
    
    # FSL indicators
    fsl_indicators = ["FSL", "FSLBASIC", "TFMODELSFSL"]
    # ASL indicators (non-FSL)
    asl_indicators = ["ASL", "ASLBASIC", "TFMODELS"]
    
    has_fsl = any(indicator in path_str for indicator in fsl_indicators)
    has_asl = any(indicator in path_str and "FSL" not in path_str.split(indicator)[0] 
                  for indicator in asl_indicators)
    
    # Disambiguate
    if has_fsl and not has_asl:
        return ORIGIN_FSL
    elif has_asl and not has_fsl:
        return ORIGIN_ASL
    else:
        # Ambiguous or neither - mark as unknown
        return ORIGIN_UNKNOWN

def load_csv(path: Path):
    try:
        df = pd.read_csv(path)
        if "label" not in df.columns:
            return None
        
        # Extract hand features - handle two formats:
        # Format 1: L1_x0..L1_z20, L2_x0..L2_z20 (two-hand with prefixes)
        # Format 2: 0,1,2,...,62 (single-hand with numeric column names)
        hand_cols = [c for c in df.columns if c.startswith(("L1_", "L2_"))]
        
        if not hand_cols:
            # Try format 2: numeric columns (single hand, 63 columns = 21 points × 3)
            numeric_cols = [c for c in df.columns if c != "label" and str(c).isdigit()]
            if len(numeric_cols) == 63:
                # This is single-hand format with numeric column names
                # Sort numeric columns to ensure correct order (0, 1, 2, ..., 62)
                numeric_cols = sorted(numeric_cols, key=lambda x: int(x))
                X_hands_numeric = df[numeric_cols].to_numpy(dtype=np.float32, copy=False)
                # The numeric columns are already in the right order (x0,y0,z0,...,x20,y20,z20)
                X_hands = X_hands_numeric
            else:
                return None
        else:
            X_hands = df[hand_cols].to_numpy(dtype=np.float32, copy=False)
        
        # Normalize to 126-dim (pad if one hand)
        X_hands_126 = to_126d(X_hands)
        if X_hands_126 is None:
            return None
        
        # Extract face features using NMM indices
        X_face = extract_face_features(df)
        
        # Combine hand + face features
        X_combined = np.concatenate([X_hands_126, X_face], axis=1)
        
        y = df["label"].astype(str).to_list()
        return X_combined, y
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None

def to_126d(X):
    # If 63-dim (one hand) -> pad to 126 with zeros
    if X.shape[1] == TWO_HAND_DIM:
        return X
    if X.shape[1] == ONE_HAND_DIM:
        pad = np.zeros((X.shape[0], ONE_HAND_DIM), dtype=np.float32)
        return np.concatenate([X, pad], axis=1)
    # Unknown layout; skip
    return None

def extract_face_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract face features using NMM-specific indices from CSV columns.
    Returns face features as (N_frames, FACE_DIM) array, or zeros if not found.
    """
    face_features = []
    n_rows = len(df)
    
    # Try to extract face columns in format Fx{index}, Fy{index}, Fz{index}
    for idx in FACE_INDICES:
        x_col = f"Fx{idx}"
        y_col = f"Fy{idx}"
        z_col = f"Fz{idx}"
        
        if x_col in df.columns and y_col in df.columns and z_col in df.columns:
            face_features.extend([df[x_col].values, df[y_col].values, df[z_col].values])
        else:
            # Face data not available for this index, pad with zeros
            face_features.extend([
                np.zeros(n_rows, dtype=np.float32),
                np.zeros(n_rows, dtype=np.float32),
                np.zeros(n_rows, dtype=np.float32)
            ])
    
    if face_features:
        face_array = np.column_stack(face_features).astype(np.float32)
    else:
        # No face data found, return zeros
        face_array = np.zeros((n_rows, FACE_DIM), dtype=np.float32)
    
    return face_array

def uniform_sample_or_interpolate(frames: np.ndarray, seq_length: int = SEQ_LENGTH, 
                                   start_index: int = None, stride_factor: float = None) -> np.ndarray:
    """
    Speed-invariant uniform sampling/interpolation for variable-length video sequences.
    
    - If > seq_length frames: Uniformly sample seq_length frames using np.linspace
    - If < seq_length frames: Use linear interpolation (indices) to stretch to seq_length
    - If == seq_length: Return as-is
    
    Args:
        frames: Array of shape (N_frames, NUM_FEATURES)
        seq_length: Target sequence length (default 30)
        start_index: Optional start offset for temporal shift augmentation
        stride_factor: Optional stride multiplier (0.8-1.2) for temporal shift augmentation
    
    Returns:
        Array of shape (seq_length, NUM_FEATURES)
    """
    n_frames = len(frames)
    
    # Calculate start and end indices for temporal shift
    if start_index is None:
        start_idx = 0
    else:
        start_idx = start_index
    
    if stride_factor is None:
        stride_factor = 1.0
    
    if n_frames == seq_length:
        # If exactly seq_length frames, can't really shift, just return as-is
        return frames.copy()
    elif n_frames > seq_length:
        # Uniform sampling with temporal shift: adjust range based on start_index and stride_factor
        max_start = max(0, n_frames - seq_length)
        start_idx = min(start_idx, max_start)
        
        # Adjust end index based on stride_factor
        # stride_factor > 1.0 means faster (sample fewer frames), < 1.0 means slower (sample more frames)
        effective_length = int(seq_length / stride_factor)
        end_idx = min(start_idx + effective_length, n_frames)
        
        # Ensure we have a valid range
        if end_idx <= start_idx:
            end_idx = min(start_idx + seq_length, n_frames)
        
        # Sample indices with adjusted stride
        indices = np.linspace(start_idx, end_idx - 1, seq_length, dtype=int)
        return frames[indices]
    else:
        # Linear interpolation: use indices to stretch (will repeat frames)
        # Apply stride_factor to adjust sampling speed
        # For short sequences, we can still vary the stride to affect interpolation
        start_idx = min(start_idx, n_frames - 1) if n_frames > 0 else 0
        effective_length = max(1, int(seq_length / stride_factor))
        end_idx = min(start_idx + effective_length, n_frames) if n_frames > 0 else 0
        
        if end_idx > start_idx:
            indices = np.linspace(start_idx, end_idx - 1, seq_length, dtype=int)
        else:
            indices = np.linspace(0, n_frames - 1, seq_length, dtype=int) if n_frames > 0 else np.zeros(seq_length, dtype=int)
        return frames[indices]

def augment_spatial_noise(frames: np.ndarray, scale: float = 0.01) -> np.ndarray:
    """
    Add random Gaussian noise to all coordinates.
    
    Args:
        frames: Array of shape (seq_length, NUM_FEATURES)
        scale: Standard deviation of Gaussian noise (default 0.01)
    
    Returns:
        Array of same shape as frames
    """
    noise = np.random.normal(0, scale, frames.shape)
    return frames + noise

def augment_scale(frames: np.ndarray, min_scale: float = 0.9, max_scale: float = 1.1) -> np.ndarray:
    """
    Scale skeleton relative to center (0.5, 0.5).
    
    Args:
        frames: Array of shape (seq_length, NUM_FEATURES)
        min_scale: Minimum scale factor (default 0.9)
        max_scale: Maximum scale factor (default 1.1)
    
    Returns:
        Array of same shape as frames
    """
    scale = np.random.uniform(min_scale, max_scale)
    
    # Reshape to (seq_length, 79, 3) to access x,y,z components
    # 237 features = 79 landmarks × 3 coords
    reshaped = frames.reshape(frames.shape[0], -1, 3)
    
    # Center is usually 0.5 for normalized coordinates
    center = 0.5
    
    # Apply scale: new_coord = center + (coord - center) * scale
    reshaped[..., 0] = center + (reshaped[..., 0] - center) * scale  # X
    reshaped[..., 1] = center + (reshaped[..., 1] - center) * scale  # Y
    # Z is usually relative, just scaling it is fine
    reshaped[..., 2] = reshaped[..., 2] * scale
    
    return reshaped.reshape(frames.shape)

def augment_rotation(frames: np.ndarray, max_angle_degrees: float = 10) -> np.ndarray:
    """
    Rotate skeleton around center (0.5, 0.5) in 2D plane.
    
    Args:
        frames: Array of shape (seq_length, NUM_FEATURES)
        max_angle_degrees: Maximum rotation angle in degrees (default 10)
    
    Returns:
        Array of same shape as frames
    """
    angle_rad = np.radians(np.random.uniform(-max_angle_degrees, max_angle_degrees))
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Reshape to (seq_length, 79, 3) to access x,y,z components
    reshaped = frames.reshape(frames.shape[0], -1, 3)
    center = 0.5
    
    x = reshaped[..., 0] - center
    y = reshaped[..., 1] - center
    
    # Rotation Matrix
    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a
    
    reshaped[..., 0] = new_x + center
    reshaped[..., 1] = new_y + center
    # Z remains unchanged
    
    return reshaped.reshape(frames.shape)

def augment_temporal_shift(raw_frames: np.ndarray, seq_length: int = SEQ_LENGTH) -> np.ndarray:
    """
    MVP Feature: Vary sampling speed and start point to create different "speeds" of the same sign.
    
    Args:
        raw_frames: Array of shape (N_frames, NUM_FEATURES)
        seq_length: Target sequence length (default 30)
    
    Returns:
        Array of shape (seq_length, NUM_FEATURES)
    """
    n_frames = len(raw_frames)
    
    # Random start_index offset (0 to max_offset based on frame count)
    max_start = max(0, n_frames - seq_length)
    start_index = np.random.randint(0, max_start + 1) if max_start > 0 else 0
    
    # Random stride_factor (0.8 to 1.2) to vary sampling speed
    stride_factor = np.random.uniform(0.8, 1.2)
    
    # Use modified uniform_sample_or_interpolate with temporal shift parameters
    return uniform_sample_or_interpolate(raw_frames, seq_length, start_index=start_index, stride_factor=stride_factor)

def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int = SEQ_LENGTH, 
                     num_augmentations: int = 50) -> tuple:
    """
    Group frames into sequences of fixed length using speed-invariant sampling with data augmentation.
    
    Frames are grouped by label. For each label, generate multiple augmented sequences using:
    - Temporal shift (always applied - MVP feature)
    - Random spatial augmentations (noise, scale, rotation)
    
    Args:
        X: Feature array of shape (N_frames, NUM_FEATURES)
        y: Label array of shape (N_frames,)
        seq_length: Target sequence length (default 30)
        num_augmentations: Number of augmented sequences to generate per label (default 50)
    
    Returns:
        sequences: Array of shape (N_sequences, seq_length, NUM_FEATURES)
        labels: Array of shape (N_sequences,) with one label per sequence
    """
    sequences = []
    labels = []
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        # Get all frames with this label
        label_mask = y == label
        label_frames = X[label_mask]
        
        if len(label_frames) == 0:
            continue
        
        # Generate num_augmentations sequences per label
        for _ in range(num_augmentations):
            # Always apply temporal shift (MVP feature - most important augmentation)
            sampled_sequence = augment_temporal_shift(label_frames, seq_length)
            
            # Randomly apply spatial augmentations (can apply multiple)
            if np.random.random() < 0.5:  # 50% chance to apply noise
                sampled_sequence = augment_spatial_noise(sampled_sequence)
            
            if np.random.random() < 0.5:  # 50% chance to apply scale
                sampled_sequence = augment_scale(sampled_sequence)
            
            if np.random.random() < 0.5:  # 50% chance to apply rotation
                sampled_sequence = augment_rotation(sampled_sequence)
            
            sequences.append(sampled_sequence)
            labels.append(label)
    
    if not sequences:
        raise ValueError("No sequences could be created from the data")
    
    sequences_array = np.array(sequences, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int64)
    
    return sequences_array, labels_array

def normalize_label(s: str) -> str:
    # Lowercase, strip, replace spaces with underscores to match Concept-Key style
    return s.strip().lower().replace(" ", "_")

def main():
    all_X = []
    all_y = []
    all_origins = []
    label_origin_counts = defaultdict(lambda: defaultdict(int))  # {label: {origin: count}}

    for pat in CSV_GLOBS:
        csv_count = 0
        loaded_count = 0
        for p in ROOT.glob(pat):
            if not p.is_file() or not p.suffix.lower() == ".csv":
                continue
            csv_count += 1
            loaded = load_csv(p)
            if not loaded:
                continue
            loaded_count += 1
            X, y = loaded
            # X already has combined hand+face features (237 columns), don't call to_126d again!
            # The load_csv function already handles normalization and combination
            
            # Verify the feature dimension
            if X.shape[1] != NUM_FEATURES:
                print(f"Warning: {p.name} has {X.shape[1]} features, expected {NUM_FEATURES}. Skipping.")
                continue
            
            # Infer origin for this file
            origin = infer_origin(p)
            
            y_norm = [normalize_label(lbl) for lbl in y]
            all_X.append(X)  # Use X directly, not X126
            all_y.extend(y_norm)
            all_origins.extend([origin] * len(y_norm))
            
            # Track label-origin statistics
            for lbl in y_norm:
                if origin != ORIGIN_UNKNOWN:
                    label_origin_counts[lbl][ORIGIN_LABELS[origin]] += 1
        
        if csv_count > 0:
            print(f"Pattern {pat}: Found {csv_count} CSVs, loaded {loaded_count}")

    if not all_X:
        raise SystemExit("No usable CSVs found for unified build.")

    X_all = np.concatenate(all_X, axis=0)
    origins_all = np.array(all_origins, dtype=np.int32)
    
    # Verify feature dimension matches expected NUM_FEATURES
    if X_all.shape[1] != NUM_FEATURES:
        print(f"Warning: Expected {NUM_FEATURES} features but got {X_all.shape[1]}")
        print(f"Padding or truncating to {NUM_FEATURES} features...")
        if X_all.shape[1] < NUM_FEATURES:
            pad = np.zeros((X_all.shape[0], NUM_FEATURES - X_all.shape[1]), dtype=np.float32)
            X_all = np.concatenate([X_all, pad], axis=1)
        else:
            X_all = X_all[:, :NUM_FEATURES]
    
    labels_sorted = sorted(set(all_y))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels_sorted)}
    y_idx = np.array([label_to_idx[lbl] for lbl in all_y], dtype=np.int64)

    # Filter out UNKNOWN origins (keep indices for consistent filtering)
    valid_mask = origins_all != ORIGIN_UNKNOWN
    X_filtered = X_all[valid_mask]
    y_filtered = y_idx[valid_mask]
    origins_filtered = origins_all[valid_mask]
    
    print(f"\nTotal frames: {len(X_all)}")
    print(f"Samples with unknown origin (excluded): {(~valid_mask).sum()}")
    print(f"Frames with valid origin: {len(X_filtered)}")
    print(f"Feature dimension: {X_filtered.shape[1]} (expected {NUM_FEATURES})")
    
    # Group frames by label and create sequences
    print(f"\nCreating {SEQ_LENGTH}-frame sequences with speed-invariant sampling...")
    X_sequences, y_sequences = create_sequences(X_filtered, y_filtered, SEQ_LENGTH)
    
    print(f"Created {len(X_sequences)} sequences from {len(X_filtered)} frames")
    print(f"Sequence shape: {X_sequences.shape} (N_sequences, {SEQ_LENGTH}, {NUM_FEATURES})")
    
    # Shuffle sequences together
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y_sequences))
    X_sequences = X_sequences[idx]
    y_sequences = y_sequences[idx]
    # Note: origins_filtered is per-frame, not per-sequence
    # We'll need to map origins to sequences (using majority vote per sequence label)
    # For now, we'll reconstruct origins from sequence labels if needed
    
    # Map sequence labels back to origins (approximate - use label origin stats)
    origins_sequences = np.zeros(len(y_sequences), dtype=np.int32)
    for i, label_idx in enumerate(y_sequences):
        label_name = labels_sorted[label_idx]
        # Find origin for this label from filtered data
        label_frame_mask = (y_filtered == label_idx)
        if label_frame_mask.sum() > 0:
            label_origins = origins_filtered[label_frame_mask]
            # Use most common origin for this label
            unique_origins, counts = np.unique(label_origins, return_counts=True)
            origins_sequences[i] = unique_origins[np.argmax(counts)]
    
    # Save main artifacts (sequences for LSTM model)
    np.save(OUT_DIR / "unified_X.npy", X_sequences)
    np.save(OUT_DIR / "unified_y.npy", y_sequences)
    with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_sorted, f, ensure_ascii=False, indent=2)

    # Save origin artifacts (per sequence)
    np.save(OUT_DIR / "unified_origin.npy", origins_sequences)
    with open(OUT_DIR / "origin_labels.json", "w", encoding="utf-8") as f:
        json.dump(ORIGIN_LABELS, f, ensure_ascii=False, indent=2)
    
    # Save label-origin statistics
    label_origin_stats = {
        lbl: dict(origins) for lbl, origins in label_origin_counts.items()
    }
    with open(OUT_DIR / "label_origin_stats.json", "w", encoding="utf-8") as f:
        json.dump(label_origin_stats, f, ensure_ascii=False, indent=2)
    
    # Save metadata for LSTM model
    face_indices_metadata = {
        "eyebrows": EYEBROW_INDICES,
        "lips": LIP_INDICES,
        "all": FACE_INDICES,
        "total_points": len(FACE_INDICES),
        "face_dim": FACE_DIM
    }
    with open(OUT_DIR / "face_indices.json", "w", encoding="utf-8") as f:
        json.dump(face_indices_metadata, f, ensure_ascii=False, indent=2)
    
    num_features_metadata = {
        "num_features": NUM_FEATURES,
        "hand_dim": TWO_HAND_DIM,
        "face_dim": FACE_DIM,
        "seq_length": SEQ_LENGTH
    }
    with open(OUT_DIR / "num_features.json", "w", encoding="utf-8") as f:
        json.dump(num_features_metadata, f, ensure_ascii=False, indent=2)

    print("\nWrote:", OUT_DIR / "unified_X.npy")
    print("Wrote:", OUT_DIR / "unified_y.npy")
    print("Wrote:", OUT_DIR / "labels.json")
    print("Wrote:", OUT_DIR / "unified_origin.npy")
    print("Wrote:", OUT_DIR / "origin_labels.json")
    print("Wrote:", OUT_DIR / "label_origin_stats.json")
    print("Wrote:", OUT_DIR / "face_indices.json")
    print("Wrote:", OUT_DIR / "num_features.json")
    print("\nFinal shapes:")
    print(f"  X_sequences: {X_sequences.shape} (N_sequences, {SEQ_LENGTH}, {NUM_FEATURES})")
    print(f"  y_sequences: {y_sequences.shape}")
    print(f"  origins_sequences: {origins_sequences.shape}")
    
    # Print origin distribution
    print("\nOrigin distribution (sequences):")
    for i, origin_name in enumerate(ORIGIN_LABELS):
        count = (origins_sequences == i).sum()
        pct = 100.0 * count / len(origins_sequences) if len(origins_sequences) > 0 else 0.0
        print(f"  {origin_name}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    main()

