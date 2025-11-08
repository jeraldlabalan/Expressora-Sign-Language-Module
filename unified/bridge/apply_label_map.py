"""
Apply concept-key mapping to labels.
Maps model labels (e.g., thank_you, i_love_you) to translation module gloss tokens.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "unified" / "data"
BRIDGE_DIR = ROOT / "unified" / "bridge"
MODELS_DIR = ROOT / "unified" / "models"

def apply_rules(label, overrides):
    """
    Apply standard rules plus explicit overrides.
    
    Standard rules:
    - lowercase
    - underscore → space
    
    Args:
        label: Original label
        overrides: Dictionary of explicit overrides
    
    Returns:
        Mapped label
    """
    # Check for explicit override first
    if label in overrides:
        return overrides[label]
    
    # Apply standard rules
    mapped = label.strip().lower().replace("_", " ")
    return mapped

def main():
    # Load original labels
    labels_path = DATA_DIR / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}\n"
            f"Run dataset builder first: python unified/data/build_unified_dataset.py"
        )
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    # Load concept-key mapping
    map_path = BRIDGE_DIR / "concept_key_map.json"
    if not map_path.exists():
        raise FileNotFoundError(f"Concept-key map not found: {map_path}")
    
    with open(map_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    overrides = mapping.get('overrides', {})
    
    print("="*60)
    print("Applying Concept-Key Mapping")
    print("="*60)
    print(f"\nInput: {labels_path}")
    print(f"Map: {map_path}")
    print(f"Total labels: {len(labels)}")
    print(f"Explicit overrides: {len(overrides)}")
    
    # Apply mapping
    mapped_labels = []
    changes = []
    
    for label in labels:
        mapped = apply_rules(label, overrides)
        mapped_labels.append(mapped)
        
        if mapped != label:
            changes.append((label, mapped))
    
    # Save mapped labels
    output_path = MODELS_DIR / "expressora_labels_mapped.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapped_labels, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved mapped labels to: {output_path}")
    print(f"  Labels changed: {len(changes)}")
    
    # Show some examples
    if changes:
        print("\nExample mappings:")
        for orig, mapped in changes[:10]:
            print(f"  {orig:20s} → {mapped}")
        
        if len(changes) > 10:
            print(f"  ... and {len(changes) - 10} more")
    
    print("\n" + "="*60)
    print("Mapping Complete!")
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

