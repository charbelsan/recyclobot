# 🎮 RecycloBot Simulation Guide

Run RecycloBot **in simulation** (no real arm needed)

> Works on any laptop/desktop — only needs MuJoCo and the LeRobot "sim" extras.

## 📋 Prerequisites

- Linux/macOS (Windows via WSL2)
- Python 3.10+
- OpenGL support (for rendering)

## 🚀 Quick Setup

### Step 1: Install RecycloBot with Simulation Support

```bash
# Clone and setup
git clone https://github.com/charbelsan/recyclobot.git
cd recyclobot

# Create environment
conda create -n recyclobot python=3.10 -y
conda activate recyclobot

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install LeRobot from GitHub with simulation extras
pip install "lerobot[smolvla,sim,aloha,pusht,xarm] @ git+https://github.com/huggingface/lerobot.git@v0.4.0"

# Install RecycloBot
pip install -e .

# Linux dependencies (if needed)
sudo apt-get install libgl1-mesa-dev ffmpeg xvfb
```

### Step 2: Download SmolVLA Weights

```bash
huggingface-cli download lerobot/smolvla_base --local-dir ~/.cache/lerobot
```

### Step 3: Setup MuJoCo (One Time)

```bash
# Download MuJoCo
wget https://github.com/deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz
tar -xzf mujoco-2.3.7-linux-x86_64.tar.gz
mkdir -p ~/.mujoco && mv mujoco-2.3.7 ~/.mujoco/

# Set environment variable
echo 'export MUJOCO_GL=egl' >> ~/.bashrc  # or osmesa for headless
source ~/.bashrc
```

## 🎯 Running Simulations

### Option 1: Gym Environment Demo

```bash
# Aloha Environment (Dual-arm manipulation)
python examples/run_recyclobot_gym_demo.py \
    --env aloha \
    --task AlohaInsertion-v0 \
    --prompt "Insert the peg into the hole" \
    --render \
    --episodes 1

# PushT Environment (2D pushing)
python examples/run_recyclobot_gym_demo.py \
    --env pusht \
    --task PushT-v0 \
    --prompt "Push the T-shaped block to the target" \
    --render

# Xarm Environment (Single-arm)
python examples/run_recyclobot_gym_demo.py \
    --env xarm \
    --task XarmLift-v0 \
    --prompt "Lift the object" \
    --render
```

### Option 2: Mock Environment Demo

```bash
# Basic simulation with mock robot
python examples/run_recyclobot_demo.py \
    --robot sim \
    --prompt "Sort all plastic bottles into the recycling bin" \
    --episodes 1

# With different planners
python examples/run_recyclobot_demo.py \
    --robot sim \
    --planner qwen \
    --prompt "Pick up the aluminum can"
```

### Option 3: Headless Mode (Servers)

```bash
# For servers without display
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Or use software rendering
export MUJOCO_GL=osmesa

# Run without GUI
python examples/run_recyclobot_gym_demo.py \
    --env aloha \
    --task AlohaInsertion-v0 \
    --no-render \
    --save-video outputs/sim_demo.mp4
```

## 📊 Collect Simulation Data

```bash
# Autonomous data collection in simulation
python scripts/collect_recyclobot_dataset_v3.py \
    --robot-path lerobot/configs/robot/sim.yaml \
    --repo-id test-user/recyclobot-sim-demos \
    --num-episodes 10 \
    --autonomous \
    --planner direct

# With specific environment
python scripts/collect_recyclobot_dataset_v3.py \
    --robot-path lerobot/configs/robot/aloha.yaml \
    --repo-id test-user/aloha-recycling \
    --num-episodes 20 \
    --autonomous
```

## 🧪 Test Pipeline

```python
# test_sim.py
import gymnasium as gym
from lerobot.common.envs.factory import make_env
from recyclobot.control.skill_runner import SkillRunner
from lerobot.common.policies.factory import make_policy

# Create environment
env = make_env("aloha_sim_insertion_human")
obs, info = env.reset()

# Load policy
policy = make_policy("smolvla", pretrained="lerobot/smolvla_base")
runner = SkillRunner(policy)

# Execute skills
skills = ["pick(block)", "place(target)"]
runner.run(skills, env)

# Save video
env.save_video("output.mp4")
```

## 🎮 Available Environments

### Aloha (Dual-arm)
- `AlohaInsertion-v0`: Peg insertion task
- `AlohaTransferCube-v0`: Cube transfer between arms
- `AlohaPegTransfer-v0`: Multiple peg manipulation

### PushT (2D Navigation)
- `PushT-v0`: Push T-block to target
- `PushTImage-v0`: Image-based variant

### Xarm (Single-arm)
- `XarmLift-v0`: Object lifting
- `XarmPushCube-v0`: Cube pushing
- `XarmReachTarget-v0`: Target reaching

## 🚨 Troubleshooting

### "GLEW initialization failed"
```bash
# Use software renderer
export MUJOCO_GL=osmesa
# Or EGL for headless GPU
export MUJOCO_GL=egl
```

### "No module named mujoco"
```bash
pip install mujoco==2.3.7
```

### "Cannot connect to X server"
```bash
# Start virtual display
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

### Performance Issues
```bash
# Reduce simulation frequency
python examples/run_recyclobot_gym_demo.py --fps 10

# Disable rendering
python examples/run_recyclobot_gym_demo.py --no-render
```

## 📈 Next Steps

1. **Fine-tune in Simulation**:
   ```bash
   python scripts/train_recyclobot.py \
       --dataset-name test-user/recyclobot-sim-demos \
       --use-lora \
       --num-epochs 10
   ```

2. **Evaluate Performance**:
   ```bash
   python scripts/evaluate_recyclobot.py \
       --dataset test-user/recyclobot-sim-demos \
       --mode efficiency
   ```

3. **Transfer to Real Robot**:
   - Models trained in sim can be fine-tuned on real data
   - Use same commands but with `--robot so101`

## 🔗 Resources

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Gymnasium Environments](https://gymnasium.farama.org/)
- [LeRobot Sim Environments](https://github.com/huggingface/lerobot/tree/main/lerobot/envs)

---

That's it! You can iterate entirely in simulation, then switch to the real arm by replacing `--robot sim` with `--robot so101`. 🤖♻️