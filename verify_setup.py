#!/usr/bin/env python
"""
Quick verification script for RecycloBot setup.
Run this after installation to make sure everything works.
"""

import sys
import subprocess

def run_command(cmd, check=True):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"❌ Command failed: {cmd}")
            print(f"   Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"❌ Exception running command: {e}")
        return False

print("=" * 60)
print("RecycloBot Setup Verification")
print("=" * 60)

# 1. Check Python version
print("\n1. Checking Python version...")
python_version = sys.version.split()[0]
if python_version.startswith("3.10"):
    print(f"   ✅ Python {python_version}")
else:
    print(f"   ⚠️  Python {python_version} (expected 3.10)")

# 2. Check CUDA
print("\n2. Checking CUDA/GPU...")
cuda_check = """
import torch
print(f'   PyTorch version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA version: {torch.version.cuda}')
else:
    print('   ⚠️  No CUDA available')
"""
run_command(f'python -c "{cuda_check}"')

# 3. Check imports
print("\n3. Checking RecycloBot imports...")
import_check = """
try:
    import lerobot
    print('   ✅ LeRobot imported')
except ImportError:
    print('   ❌ LeRobot not installed')
    
try:
    import recyclobot
    print('   ✅ RecycloBot imported')
except ImportError:
    print('   ❌ RecycloBot not installed')
    
try:
    from recyclobot.planning.direct_smolvla_planner import plan
    print('   ✅ Planners available')
except ImportError:
    print('   ❌ Planners not available')
"""
run_command(f'python -c "{import_check}"')

# 4. Check gym environments
print("\n4. Checking gym environments...")
for env in ["gym_aloha", "gym_pusht", "gym_xarm"]:
    try:
        __import__(env)
        print(f"   ✅ {env} available")
    except ImportError:
        print(f"   ⚠️  {env} not installed (optional)")

# 5. Check model weights
print("\n5. Checking model weights...")
import os
model_path = os.path.expanduser("~/.cache/lerobot/koch_aloha")
if os.path.exists(model_path):
    print(f"   ✅ Model weights found at {model_path}")
else:
    print(f"   ❌ Model weights not found. Run:")
    print(f"      huggingface-cli download lerobot/koch_aloha --local-dir ~/.cache/lerobot")

# 6. Quick functionality test
print("\n6. Running quick functionality test...")
test_code = """
from PIL import Image
from recyclobot.planning.direct_smolvla_planner import plan
img = Image.new('RGB', (640, 480))
result = plan(img, 'Test task')
print(f'   ✅ Planning test passed: {result}')
"""
run_command(f'python -c "{test_code}"')

# Summary
print("\n" + "=" * 60)
print("Setup Summary")
print("=" * 60)

all_good = True

# Check critical components
critical_checks = [
    ("Python 3.10", python_version.startswith("3.10")),
    ("CUDA available", run_command('python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"', check=False)),
    ("LeRobot installed", run_command('python -c "import lerobot"', check=False)),
    ("RecycloBot installed", run_command('python -c "import recyclobot"', check=False)),
    ("Model weights", os.path.exists(os.path.expanduser("~/.cache/lerobot/koch_aloha")))
]

for name, status in critical_checks:
    symbol = "✅" if status else "❌"
    print(f"{symbol} {name}")
    if not status:
        all_good = False

if all_good:
    print("\n🎉 RecycloBot is ready to use!")
    print("\nTry running:")
    print("  python examples/run_recyclobot_demo.py --robot sim")
else:
    print("\n⚠️  Some components need attention. Check the messages above.")

print("\n" + "=" * 60)