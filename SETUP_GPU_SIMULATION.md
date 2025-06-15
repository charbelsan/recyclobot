# RecycloBot GPU Setup & Simulation Testing Guide

This guide helps you set up and test RecycloBot on a GPU workstation without the physical SO-ARM100 robot.

## 🖥️ System Requirements

- Ubuntu 20.04/22.04 (recommended)
- NVIDIA GPU with CUDA support
- CUDA 11.8+ and cuDNN 8.6+
- Python 3.10
- 16GB+ RAM
- 50GB+ free disk space

## 📦 Step 1: Clone and Setup Environment

```bash
# Clone RecycloBot repository
git clone https://github.com/your-username/recyclobot.git
cd recyclobot

# Create conda environment
conda create -n recyclobot python=3.10 -y
conda activate recyclobot

# Install CUDA dependencies (if not already installed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install LeRobot with simulation support and gym environments from GitHub
pip install "lerobot[smolvla,sim] @ git+https://github.com/huggingface/lerobot.git@main"

# Install specific gym environments (choose based on your needs)
pip install "lerobot[aloha]"    # Dual-arm manipulation tasks
pip install "lerobot[pusht]"    # 2D pushing tasks
pip install "lerobot[xarm]"     # Single-arm manipulation

# Install RecycloBot
pip install -e .

# Install additional dependencies for simulation
pip install gymnasium opencv-python-headless imageio[ffmpeg]
```

## 📥 Step 2: Download Required Models

```bash
# Download SmolVLA weights (REQUIRED!)
huggingface-cli download lerobot/smolvla_base --local-dir ~/.cache/lerobot

# Optional: Install local VLM planner (Qwen)
pip install "transformers[vision]>=4.44.0" "accelerate>=0.26.0" "bitsandbytes>=0.41.0"
```

## 🧪 Step 3: Test Basic Functionality

### Test 1: Verify GPU and Imports
```python
python -c "
import torch
import lerobot
from recyclobot.planning.direct_smolvla_planner import plan
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'LeRobot version: {lerobot.__version__}')
print('✅ All imports successful!')
"
```

### Test 2: Test Planners
```python
# Test direct planner (default)
python -c "
from PIL import Image
from recyclobot.planning.direct_smolvla_planner import plan
img = Image.new('RGB', (640, 480))
result = plan(img, 'Sort the recycling')
print(f'Direct planner result: {result}')
"

# Test Qwen planner (local, requires model download)
python -c "
from PIL import Image
from recyclobot.planning.qwen_planner import plan
img = Image.new('RGB', (640, 480))
try:
    result = plan(img, 'Sort the recycling')
    print(f'Qwen planner result: {result}')
except Exception as e:
    print(f'Qwen not available (expected if model not downloaded): {e}')
"
```

## 🎮 Step 4: Run Simulation Demo

### Option 1: LeRobot Gym Environments (Recommended)

LeRobot comes with several gym environments for robotic manipulation:

#### Aloha Environment (Dual-arm manipulation)
```bash
# Test dual-arm insertion task
python examples/run_recyclobot_gym_demo.py \
    --env aloha \
    --task AlohaInsertion-v0 \
    --prompt "Insert the peg into the hole" \
    --render \
    --episodes 1

# Test cube transfer task
python examples/run_recyclobot_gym_demo.py \
    --env aloha \
    --task AlohaTransferCube-v0 \
    --prompt "Transfer the cube to the target location" \
    --render
```

#### PushT Environment (2D manipulation)
```bash
# Simpler 2D pushing task
python examples/run_recyclobot_gym_demo.py \
    --env pusht \
    --task PushT-v0 \
    --prompt "Push the T-shaped block to the target" \
    --render
```

#### Xarm Environment (Single-arm manipulation)
```bash
# Single arm lifting task
python examples/run_recyclobot_gym_demo.py \
    --env xarm \
    --task XarmLift-v0 \
    --prompt "Lift the object" \
    --render
```

### Option 2: Mock Environment (Fallback)
```bash
# Run with simulated robot (default direct mode)
python examples/run_recyclobot_demo.py \
    --robot sim \
    --prompt "Sort all plastic bottles into the recycling bin" \
    --episodes 1 \
    --output test_output

# Run with different planners
python examples/run_recyclobot_demo.py \
    --robot sim \
    --planner qwen \
    --prompt "Pick up the aluminum can and place it in recycling"
```

## 📊 Step 5: Test Data Collection

### Collect Simulated Dataset
```bash
# Create mock teleoperated dataset
python scripts/collect_recyclobot_dataset_v3.py \
    --robot-path lerobot/configs/robot/sim.yaml \
    --repo-id test-user/recyclobot-sim-demos \
    --num-episodes 5 \
    --autonomous \
    --planner direct

# Check generated dataset
ls -la data/recyclobot-sim-demos/
```

### Verify Dataset Format
```python
python -c "
from datasets import load_from_disk
dataset = load_from_disk('data/recyclobot-sim-demos')
print(f'Dataset size: {len(dataset)}')
print(f'Features: {dataset.features}')
print(f'First sample keys: {list(dataset[0].keys())}')
"
```

## 🚀 Step 6: Full Pipeline Test

Create a test script `test_pipeline.py`:

```python
#!/usr/bin/env python
"""Test RecycloBot pipeline without physical robot."""

import numpy as np
from PIL import Image
import torch

# Test 1: Planning
print("1. Testing planning...")
from recyclobot.planning.direct_smolvla_planner import plan
test_image = Image.new('RGB', (640, 480), color=(200, 200, 200))
skills = plan(test_image, "Sort the recycling")
print(f"   Generated skills: {skills}")

# Test 2: Policy Loading
print("\n2. Testing SmolVLA policy loading...")
try:
    from lerobot.common.policies.factory import make_policy
    policy = make_policy(
        "smolvla",
        policy_kwargs={
            "pretrained": "lerobot/smolvla_base",
            "config_overrides": {
                "input_shapes": {
                    "observation.images.top": [3, 480, 640],
                    "observation.state": [14],
                },
                "output_shapes": {
                    "action": [7],
                }
            }
        }
    )
    print("   ✅ SmolVLA loaded successfully")
except Exception as e:
    print(f"   ❌ SmolVLA loading failed: {e}")

# Test 3: Skill Runner
print("\n3. Testing skill runner...")
from recyclobot.control.skill_runner import SkillRunner
runner = SkillRunner(policy if 'policy' in locals() else None)
instruction = runner.skill_to_language_prompt("pick(plastic_bottle)")
print(f"   Skill mapping: pick(plastic_bottle) → '{instruction}'")

# Test 4: Mock Execution
print("\n4. Testing mock execution...")
class MockEnv:
    def get_observation(self):
        return {
            "observation.images.top": torch.randn(3, 480, 640),
            "observation.state": torch.randn(14)
        }
    def send_action(self, action):
        pass
    def close(self):
        pass

env = MockEnv()
if 'policy' in locals():
    obs = env.get_observation()
    obs["task"] = instruction
    with torch.no_grad():
        action = policy.select_action(obs)
    print(f"   Generated action shape: {action.shape}")
    print("   ✅ Full pipeline test passed!")
else:
    print("   ⚠️  Skipping execution test (no policy loaded)")

print("\n✅ All tests completed!")
```

Run it:
```bash
python test_pipeline.py
```

## 🐛 Step 7: Troubleshooting

### Common Issues and Solutions

#### GPU Memory Issues
```bash
# Monitor GPU usage
nvidia-smi

# Reduce batch size if OOM
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### Missing Dependencies
```bash
# Install all optional dependencies from GitHub
pip install "lerobot[smolvla,sim,dev] @ git+https://github.com/huggingface/lerobot.git@main"
pip install matplotlib plotly pandas
```

#### Model Loading Errors
```bash
# Clear cache and re-download
rm -rf ~/.cache/lerobot
huggingface-cli download lerobot/smolvla_base --local-dir ~/.cache/lerobot
```

#### Simulation Display Issues
```bash
# For headless servers, use virtual display
sudo apt-get install xvfb
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
```

## 📈 Step 8: Performance Testing

```python
# benchmark.py
import time
import torch
from recyclobot.planning.direct_smolvla_planner import plan
from PIL import Image

# Warm up
img = Image.new('RGB', (640, 480))
_ = plan(img, "test")

# Benchmark
times = []
for i in range(10):
    start = time.time()
    _ = plan(img, f"Sort recycling task {i}")
    times.append(time.time() - start)

print(f"Average planning time: {np.mean(times)*1000:.2f}ms")
print(f"Planning FPS: {1/np.mean(times):.2f}")
```

## 🎯 Step 9: Next Steps

1. **Train on Custom Data**:
   ```bash
   python scripts/train_recyclobot.py \
       --dataset-name test-user/recyclobot-sim-demos \
       --output-dir outputs/recyclobot_sim \
       --use-lora \
       --num-epochs 5
   ```

2. **Evaluate Performance**:
   ```bash
   python scripts/evaluate_recyclobot.py \
       --dataset test-user/recyclobot-sim-demos \
       --checkpoint outputs/recyclobot_sim \
       --mode planning
   ```

3. **Create Custom Simulation**:
   - Extend the mock environment
   - Add visual feedback
   - Implement success metrics

## 📝 Quick Test Commands Summary

```bash
# 1. Quick functionality test
python -c "import recyclobot; print('✅ RecycloBot imported successfully')"

# 2. Run simulation demo
python examples/run_recyclobot_demo.py --robot sim --episodes 1

# 3. Test data collection
python scripts/collect_recyclobot_dataset_v3.py \
    --robot-path lerobot/configs/robot/sim.yaml \
    --repo-id test/sim-demos \
    --num-episodes 2

# 4. Full pipeline test
python test_pipeline.py
```

## 🔗 Resources

- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [SmolVLA Blog Post](https://huggingface.co/blog/smolvla)
- [RecycloBot Issues](https://github.com/your-username/recyclobot/issues)

---

**Note**: This simulation setup allows you to test and develop RecycloBot's AI components without physical hardware. When you're ready to deploy on real hardware, the same code will work with minimal changes!