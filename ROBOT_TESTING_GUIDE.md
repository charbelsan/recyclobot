# Robot Testing Guide for RecycloBot

## For Team Members Testing with Real SO-ARM100 Robot

### Prerequisites Check

Before starting, ensure you have:
- ✅ SO-ARM100 robot connected via USB
- ✅ Webcam connected (robot's camera or external USB camera)
- ✅ NVIDIA GPU with CUDA support
- ✅ Ubuntu 20.04 or 22.04
- ✅ Python 3.10 with conda/mamba

### Step 1: Pull Latest Changes

```bash
cd recyclobot
git pull origin master
```

### Step 2: Environment Setup

```bash
# Create fresh conda environment
conda create -n recyclobot python=3.10 -y
conda activate recyclobot

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install the package
pip install -e .
```

### Step 3: Verify GPU Setup

```bash
# Check CUDA is working
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: Test SmolVLA Loading

```bash
# This will download the model (~2GB) on first run
python quick_validate.py
```

Expected output:
```
Loading SmolVLA model...
✅ All normalization stats look good!
✅ Inference works! Action shape: torch.Size([6])
```

### Step 5: Connect Robot

1. **Power on the SO-ARM100**
2. **Connect USB cable to computer**
3. **Check connection:**
   ```bash
   ls /dev/ttyUSB*
   # Should see /dev/ttyUSB0 or similar
   ```

4. **Set permissions (if needed):**
   ```bash
   sudo chmod 666 /dev/ttyUSB0
   ```

### Step 6: Run Real Robot Demo

```bash
# Basic sorting demo
python examples/run_recyclobot_demo.py --robot so101 --prompt "Sort the items on the table"

# Specific object pickup
python examples/run_recyclobot_demo.py --robot so101 --prompt "Pick up the plastic bottle"

# Multiple episodes
python examples/run_recyclobot_demo.py --robot so101 --episodes 5
```

### Step 7: Using Different Planners

```bash
# With Gemini (need GEMINI_API_KEY)
export GEMINI_API_KEY="your-key-here"
python examples/run_recyclobot_demo.py --robot so101 --planner gemini

# With local Qwen-VL (no API needed)
python examples/run_recyclobot_demo.py --robot so101 --planner qwen

# Direct SmolVLA (default, no separate planner)
python examples/run_recyclobot_demo.py --robot so101 --planner direct
```

### Step 8: Safety Setup

⚠️ **IMPORTANT SAFETY STEPS:**

1. **Clear the workspace** - Remove fragile items
2. **Set up bins** - Place recycling/trash bins within reach
3. **Test objects** - Use safe items (empty plastic bottles, paper)
4. **Emergency stop** - Keep hand on power switch
5. **Start slow** - Use `--fps 5` for slower movements

### Step 9: Troubleshooting

#### Robot Not Found
```bash
# Check connections
ls /dev/ttyUSB*
sudo dmesg | tail -20

# Try different port
python examples/run_recyclobot_demo.py --robot so101 --port /dev/ttyUSB1
```

#### CUDA/GPU Issues
```bash
# Force CPU (slower but works)
CUDA_VISIBLE_DEVICES="" python examples/run_recyclobot_demo.py --robot so101

# Check GPU memory
nvidia-smi
```

#### Model Loading Issues
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--lerobot--smolvla_base
python quick_validate.py
```

#### Robot Moving Too Fast
```bash
# Reduce speed
python examples/run_recyclobot_demo.py --robot so101 --fps 5
```

### Step 10: Data Collection Mode

To collect demonstration data for training:

```bash
# Teleoperated recording
python scripts/record_recyclobot_teleoperated.py \
    --robot so101 \
    --fps 10 \
    --episodes 50 \
    --output datasets/recycling_demos
```

### What to Expect

1. **First Run**: Model download (~2GB), takes 2-3 minutes
2. **Robot Behavior**: 
   - Looks at scene
   - Plans actions based on prompt
   - Executes pick/place movements
   - ~30-60 seconds per skill
3. **Success Rate**: ~60-80% on simple tasks
4. **Data Saved**: In `recyclobot_data/` directory

### Quick Test Commands

```bash
# 1. Minimal test (simulation)
python examples/run_recyclobot_demo.py

# 2. Real robot test
python examples/run_recyclobot_demo.py --robot so101

# 3. Full pipeline test
python examples/run_recyclobot_demo.py --robot so101 --planner gemini --episodes 3
```

### Reporting Issues

If you encounter problems:

1. **Save the error output**
2. **Note the command you ran**
3. **Check robot LED status**
4. **Take a photo of the setup**

### Tips for Success

- 🎯 Start with simple, clear objects (bottles, cans)
- 📦 Keep bins close to robot's workspace
- 💡 Good lighting helps vision model
- 🔄 Reset robot position between runs
- 📝 Take notes on what works/fails

---

**Need help?** Check the error, try the troubleshooting steps, then reach out with:
- Full error message
- Command you ran
- Photo of robot setup
- Output of `python quick_validate.py`