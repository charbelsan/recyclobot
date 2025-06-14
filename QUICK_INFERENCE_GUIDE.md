# 🚀 Quick Inference Guide - Test RecycloBot Without Training

This guide shows how to test RecycloBot's base capabilities immediately after installation.

## 📋 Prerequisites

```bash
# 1. Clone the repository
git clone https://github.com/charbelsan/recyclobot.git
cd recyclobot

# 2. Create environment
conda create -n recyclobot python=3.10 -y
conda activate recyclobot

# 3. Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. Install LeRobot and RecycloBot
pip install "lerobot[smolvla]==0.4.0"
pip install -e .

# 5. Download SmolVLA weights (REQUIRED!)
huggingface-cli download lerobot/smolvla_base --local-dir ~/.cache/lerobot
```

## 🧪 Test 1: Basic Functionality Test

```bash
# Run the base model test
python test_base_model.py
```

This will test:
- ✅ Installation verification
- ✅ Planning capabilities
- ✅ Language understanding
- ✅ Action generation

## 🎮 Test 2: Simulated Robot Demo

```bash
# Run with mock environment (no GPU needed)
python examples/run_recyclobot_demo.py --robot sim --episodes 1 --prompt "Sort the recycling"
```

What happens:
1. Uses direct SmolVLA mode (no separate planner)
2. Generates a skill sequence from your prompt
3. Executes with a simulated robot
4. Saves data to `recyclobot_data/`

## 🎯 Test 3: Different Planning Modes

```bash
# Test different prompts
python examples/run_recyclobot_demo.py --robot sim --prompt "Pick up all plastic bottles"
python examples/run_recyclobot_demo.py --robot sim --prompt "Put the can in the blue bin"
python examples/run_recyclobot_demo.py --robot sim --prompt "Clean the workspace"
```

## 💡 Test 4: Quick Python Test

```python
# test_inference.py
from PIL import Image
from recyclobot.planning.direct_smolvla_planner import plan

# Create test image
img = Image.new('RGB', (640, 480), color=(100, 100, 100))

# Test planning
result = plan(img, "Sort the plastic bottles into recycling")
print(f"Generated plan: {result}")

# Output: ['pick(plastic_bottle)', 'place(recycling_bin)', ...]
```

## 🔥 Test 5: Full Pipeline Test

```python
import torch
from lerobot.common.policies.factory import make_policy
from recyclobot.control.skill_runner import SkillRunner

# Load SmolVLA
policy = make_policy(
    "smolvla",
    pretrained="lerobot/smolvla_base",
    config_overrides={
        "input_shapes": {
            "observation.images.top": [3, 480, 640],
            "observation.state": [14],
        },
        "output_shapes": {"action": [7]}
    }
)

# Create skill runner
runner = SkillRunner(policy)

# Test skill to language conversion
lang = runner.skill_to_language_prompt("pick(plastic_bottle)")
print(f"Language instruction: {lang}")
# Output: "pick up the plastic bottle"

# Test inference
obs = {
    "observation.images.top": torch.randn(3, 480, 640),
    "observation.state": torch.randn(14),
    "task": lang
}

with torch.no_grad():
    action = policy.select_action(obs)
print(f"Action shape: {action.shape}")  # (7,)
```

## 📊 What You'll See Without Training

### ✅ Base Model CAN:
- Understand natural language commands
- Generate 7-DOF robot actions
- Map high-level skills to language
- Respond to recycling-related prompts
- Produce reasonable motion primitives

### ❌ Base Model CANNOT (needs fine-tuning):
- Know your specific workspace layout
- Identify your exact bin locations
- Recognize your specific objects
- Execute optimal sorting strategies
- Handle your particular robot dynamics

## 🎯 Expected Output Examples

```bash
$ python test_base_model.py

🔍 Checking installations...
✅ LeRobot version: 0.4.0
✅ RecycloBot modules loaded

🧠 Testing planning capabilities...
Prompt: 'Pick up the plastic bottle and put it in the recycling bin'
Plan: ['pick(plastic_bottle)', 'place(recycling_bin)']

🗣️ Testing language understanding...
Skill to language mapping:
  pick(plastic_bottle) → 'pick up the plastic bottle'
  place(recycling_bin) → 'place the object in the recycling bin (blue)'

🤖 Testing model inference...
Generated action shape: (7,)
✅ Model can generate actions!
```

## 🚨 Troubleshooting

### "Model not found" error
```bash
# Download the model manually
huggingface-cli download lerobot/smolvla_base --local-dir ~/.cache/lerobot
```

### "CUDA out of memory" error
```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""
python examples/run_recyclobot_demo.py --robot sim
```

### "Module not found" error
```bash
# Make sure you're in the recyclobot directory
cd recyclobot
pip install -e .
```

## 🎬 Next Steps

1. **Collect Data**: Record demonstrations with your robot
   ```bash
   python scripts/collect_recyclobot_dataset_v3.py \
       --robot-path lerobot/configs/robot/so101.yaml \
       --repo-id your-username/recycling-demos
   ```

2. **Fine-tune**: Train on your specific setup
   ```bash
   python scripts/train_recyclobot.py \
       --dataset-name your-username/recycling-demos \
       --use-lora
   ```

3. **Deploy**: Use your fine-tuned model
   ```bash
   python examples/run_recyclobot_demo.py \
       --robot so101 \
       --checkpoint outputs/recyclobot_lora
   ```

---

**Remember**: The base model provides general manipulation capabilities. Fine-tuning teaches it your specific recycling setup! 🤖♻️