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
    "Tensorflow/**/*.csv",
]

ONE_HAND_DIM = 21*3   # 63
TWO_HAND_DIM = ONE_HAND_DIM*2  # 126

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
        # Collect feature columns (order-stable)
        feat_cols = [c for c in df.columns if c != "label"]
        X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
        y = df["label"].astype(str).to_list()
        return X, y
    except Exception:
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

def normalize_label(s: str) -> str:
    # Lowercase, strip, replace spaces with underscores to match Concept-Key style
    return s.strip().lower().replace(" ", "_")

def main():
    all_X = []
    all_y = []
    all_origins = []
    label_origin_counts = defaultdict(lambda: defaultdict(int))  # {label: {origin: count}}

    for pat in CSV_GLOBS:
        for p in ROOT.glob(pat):
            if not p.is_file() or not p.suffix.lower() == ".csv":
                continue
            loaded = load_csv(p)
            if not loaded:
                continue
            X, y = loaded
            X126 = to_126d(X)
            if X126 is None:
                continue
            
            # Infer origin for this file
            origin = infer_origin(p)
            
            y_norm = [normalize_label(lbl) for lbl in y]
            all_X.append(X126)
            all_y.extend(y_norm)
            all_origins.extend([origin] * len(y_norm))
            
            # Track label-origin statistics
            for lbl in y_norm:
                if origin != ORIGIN_UNKNOWN:
                    label_origin_counts[lbl][ORIGIN_LABELS[origin]] += 1

    if not all_X:
        raise SystemExit("No usable CSVs found for unified build.")

    X_all = np.concatenate(all_X, axis=0)
    origins_all = np.array(all_origins, dtype=np.int32)
    
    labels_sorted = sorted(set(all_y))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels_sorted)}
    y_idx = np.array([label_to_idx[lbl] for lbl in all_y], dtype=np.int64)

    # Filter out UNKNOWN origins (keep indices for consistent filtering)
    valid_mask = origins_all != ORIGIN_UNKNOWN
    X_filtered = X_all[valid_mask]
    y_filtered = y_idx[valid_mask]
    origins_filtered = origins_all[valid_mask]
    
    print(f"Total samples: {len(X_all)}")
    print(f"Samples with unknown origin (excluded): {(~valid_mask).sum()}")
    print(f"Samples with valid origin: {len(X_filtered)}")

    # Shuffle together
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y_filtered))
    X_filtered = X_filtered[idx]
    y_filtered = y_filtered[idx]
    origins_filtered = origins_filtered[idx]

    # Save main artifacts (gloss classification)
    np.save(OUT_DIR / "unified_X.npy", X_filtered)
    np.save(OUT_DIR / "unified_y.npy", y_filtered)
    with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_sorted, f, ensure_ascii=False, indent=2)

    # Save origin artifacts
    np.save(OUT_DIR / "unified_origin.npy", origins_filtered)
    with open(OUT_DIR / "origin_labels.json", "w", encoding="utf-8") as f:
        json.dump(ORIGIN_LABELS, f, ensure_ascii=False, indent=2)
    
    # Save label-origin statistics
    label_origin_stats = {
        lbl: dict(origins) for lbl, origins in label_origin_counts.items()
    }
    with open(OUT_DIR / "label_origin_stats.json", "w", encoding="utf-8") as f:
        json.dump(label_origin_stats, f, ensure_ascii=False, indent=2)

    print("\nWrote:", OUT_DIR / "unified_X.npy")
    print("Wrote:", OUT_DIR / "unified_y.npy")
    print("Wrote:", OUT_DIR / "labels.json")
    print("Wrote:", OUT_DIR / "unified_origin.npy")
    print("Wrote:", OUT_DIR / "origin_labels.json")
    print("Wrote:", OUT_DIR / "label_origin_stats.json")
    print("\nShapes:", X_filtered.shape, y_filtered.shape, origins_filtered.shape)
    
    # Print origin distribution
    print("\nOrigin distribution:")
    for i, origin_name in enumerate(ORIGIN_LABELS):
        count = (origins_filtered == i).sum()
        pct = 100.0 * count / len(origins_filtered)
        print(f"  {origin_name}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    main()

