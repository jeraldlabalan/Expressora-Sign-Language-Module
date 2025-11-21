"""
Monitor training progress and automatically handle completion.
"""
import json
import time
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "unified" / "models"
TRAINING_SCRIPT = ROOT / "unified" / "training" / "train_unified_tf.py"

def check_training_status():
    """Check if training has completed"""
    keras_model = MODELS / "expressora_unified.keras"
    training_log = MODELS / "training_log.json"
    diversity_history = MODELS / "diversity_history.json"
    
    # Check if model exists and is recent
    if keras_model.exists():
        import os
        import datetime
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(keras_model))
        age_minutes = (datetime.datetime.now() - mod_time).total_seconds() / 60
        
        if age_minutes < 10:  # Model updated in last 10 minutes
            print(f"✅ Model file exists and was updated {age_minutes:.1f} minutes ago")
            return True
    
    # Check if training log exists
    if training_log.exists():
        try:
            with open(training_log, 'r') as f:
                log = json.load(f)
            if 'loss' in log and len(log['loss']) > 0:
                print(f"✅ Training log exists with {len(log['loss'])} epochs")
                return True
        except:
            pass
    
    return False

def wait_for_training(max_wait_hours=3, check_interval=300):
    """Wait for training to complete, checking every 5 minutes"""
    print(f"Monitoring training progress...")
    print(f"Will check every {check_interval/60:.1f} minutes")
    print(f"Maximum wait time: {max_wait_hours} hours")
    print("="*70)
    
    start_time = time.time()
    max_wait_seconds = max_wait_hours * 3600
    
    while True:
        elapsed = time.time() - start_time
        elapsed_hours = elapsed / 3600
        
        if check_training_status():
            print(f"\n✅ Training appears to have completed!")
            return True
        
        if elapsed > max_wait_seconds:
            print(f"\n⏰ Maximum wait time reached ({max_wait_hours} hours)")
            return False
        
        print(f"\n[{elapsed_hours:.2f}h elapsed] Training still in progress...")
        time.sleep(check_interval)

def run_post_training():
    """Run post-training steps: export and test"""
    print("\n" + "="*70)
    print("POST-TRAINING AUTOMATION")
    print("="*70)
    
    # 1. Export to TFLite
    print("\n[1/3] Exporting to TFLite...")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "unified" / "export" / "export_unified_tflite.py")],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode == 0:
            print("✅ TFLite export successful")
        else:
            print(f"❌ TFLite export failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Export error: {e}")
        return False
    
    # 2. Test model performance
    print("\n[2/3] Testing model performance...")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "test_unified_models.py")],
            capture_output=True,
            text=True,
            timeout=300
        )
        print(result.stdout)
        if result.returncode == 0:
            print("✅ Performance test completed")
        else:
            print(f"⚠️ Performance test had issues: {result.stderr}")
    except Exception as e:
        print(f"⚠️ Test error: {e}")
    
    # 3. Test on real data
    print("\n[3/3] Testing on real data samples...")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "test_real_data.py")],
            capture_output=True,
            text=True,
            timeout=300
        )
        print(result.stdout)
        if result.returncode == 0:
            print("✅ Real data test completed")
        else:
            print(f"⚠️ Real data test had issues: {result.stderr}")
    except Exception as e:
        print(f"⚠️ Test error: {e}")
    
    return True

if __name__ == "__main__":
    print("="*70)
    print("TRAINING MONITOR & AUTOMATION")
    print("="*70)
    
    # Check if training is already complete
    if check_training_status():
        print("\nTraining appears complete. Running post-training steps...")
        run_post_training()
    else:
        print("\nTraining not yet complete. Waiting...")
        if wait_for_training(max_wait_hours=3):
            print("\nTraining completed! Running post-training steps...")
            run_post_training()
        else:
            print("\n⚠️ Training may still be running or encountered an issue.")
            print("Please check manually.")

