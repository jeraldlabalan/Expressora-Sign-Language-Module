"""Quick verification script to check if repository is ready for Kaggle"""
from pathlib import Path

print("="*70)
print("KAGGLE READINESS VERIFICATION")
print("="*70)

# Check required directories
required_dirs = [
    "unified/data",
    "unified/training", 
    "unified/export",
    "tools",
    "AlphabetSignLanguages",
    "ASLBasicPhrasesSignLanguages",
    "FSLBasicPhrasesSignLanguages",
    "ASLFacialExpressionsLanguages",
    "FSLFacialExpressionsLanguages",
    "Tensorflow"
]

print("\n1. Checking required directories...")
missing_dirs = []
for dir_path in required_dirs:
    if Path(dir_path).exists():
        print(f"  [OK] {dir_path}")
    else:
        print(f"  [MISSING] {dir_path}")
        missing_dirs.append(dir_path)

# Check required files
required_files = [
    "unified/data/build_unified_dataset.py",
    "unified/training/train_unified_tf.py",
    "unified/export/export_unified_tflite.py",
    "tools/monitor_training.py",
    "requirements_kaggle.txt"
]

print("\n2. Checking required files...")
missing_files = []
for file_path in required_files:
    if Path(file_path).exists():
        print(f"  [OK] {file_path}")
    else:
        print(f"  [MISSING] {file_path}")
        missing_files.append(file_path)

# Check CSV files in data directories
print("\n3. Checking CSV data files...")
csv_dirs = [
    "AlphabetSignLanguages",
    "ASLBasicPhrasesSignLanguages",
    "FSLBasicPhrasesSignLanguages",
    "ASLFacialExpressionsLanguages",
    "FSLFacialExpressionsLanguages",
    "Tensorflow"
]

total_csvs = 0
for dir_path in csv_dirs:
    if Path(dir_path).exists():
        csv_files = list(Path(dir_path).glob("**/*.csv"))
        count = len(csv_files)
        total_csvs += count
        if count > 0:
            print(f"  [OK] {dir_path}: {count} CSV files")
        else:
            print(f"  [WARN] {dir_path}: No CSV files found")
    else:
        print(f"  [MISSING] {dir_path}: Directory missing")

print(f"\n  Total CSV files found: {total_csvs}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if missing_dirs:
    print(f"[ERROR] Missing directories: {len(missing_dirs)}")
    for d in missing_dirs:
        print(f"   - {d}")
else:
    print("[OK] All required directories present")

if missing_files:
    print(f"[ERROR] Missing files: {len(missing_files)}")
    for f in missing_files:
        print(f"   - {f}")
else:
    print("[OK] All required files present")

if total_csvs == 0:
    print("[WARNING] No CSV files found! Dataset building will fail.")
elif total_csvs < 100:
    print(f"[WARNING] Only {total_csvs} CSV files found. Expected many more.")
else:
    print(f"[OK] Found {total_csvs} CSV files - looks good!")

if not missing_dirs and not missing_files and total_csvs > 0:
    print("\n[SUCCESS] REPOSITORY APPEARS READY FOR KAGGLE!")
    print("\nNext steps:")
    print("1. Zip the entire repository")
    print("2. Upload to Kaggle")
    print("3. Run: python unified/data/build_unified_dataset.py")
    print("4. Run: python unified/training/train_unified_tf.py")
else:
    print("\n[ERROR] REPOSITORY NOT READY - Fix issues above before zipping")

