# RecycloBot Setup for Your GPU Station

Since you already have Python 3.10 and CUDA configured, here's the streamlined setup:

## 1. Clone and Setup RecycloBot

```bash
# Clone the repository
git clone https://github.com/charbelsan/recyclobot.git
cd recyclobot

# Create virtual environment (since you have Python 3.10)
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## 2. Install PyTorch with CUDA Support

```bash
# Install PyTorch for CUDA 11.8 (adjust if you have different CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA is working
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## 3. Install LeRobot with Simulation Support

```bash
# Install LeRobot with SmolVLA and simulation environments from GitHub
pip install "lerobot[smolvla,sim] @ git+https://github.com/huggingface/lerobot.git@main"

# Install gym environments (choose what you need)
pip install "lerobot[aloha]"    # Dual-arm manipulation
pip install "lerobot[pusht]"    # 2D pushing tasks
pip install "lerobot[xarm]"     # Single-arm tasks

# Install additional dependencies
pip install gymnasium opencv-python-headless imageio[ffmpeg] matplotlib
```

## 4. Install RecycloBot

```bash
# Install RecycloBot in development mode
pip install -e .

# Optional: Install local VLM planner dependencies
pip install "transformers[vision]>=4.44.0" "accelerate>=0.26.0"
```

## 5. Download Required Model Weights

```bash
# Download SmolVLA pretrained weights (REQUIRED!)
huggingface-cli download lerobot/smolvla_base --local-dir ~/.cache/lerobot

# Optional: Download base SmolVLA model
huggingface-cli download lerobot/smolvla_base --local-dir ~/.cache/lerobot
```

## 6. Quick Test

```bash
# Test RecycloBot imports and GPU
python test_recyclobot_gpu.py

# If successful, you should see:
# ✅ RecycloBot is ready for GPU simulation!
```

## 7. Run Your First Demo

### Option A: Mock Simulation (Quick Test)
```bash
# Run with direct SmolVLA (default, most efficient)
python examples/run_recyclobot_demo.py --robot sim --prompt "Sort the recycling"

# Or test with external planner
python examples/run_recyclobot_demo.py --robot sim --planner qwen --prompt "Pick up bottles"
```

### Option B: Gym Environment with Physics
```bash
# Test with Aloha dual-arm environment
python examples/run_recyclobot_gym_demo.py \
    --env aloha \
    --task AlohaInsertion-v0 \
    --prompt "Insert the peg into the hole" \
    --render

# Test with simpler PushT environment
python examples/run_recyclobot_gym_demo.py \
    --env pusht \
    --prompt "Push the block to the target" \
    --render
```

## 8. Collect Dataset for Training

```bash
# Collect data with mock simulation
python scripts/collect_recyclobot_dataset_v3.py \
    --robot-path lerobot/configs/robot/sim.yaml \
    --repo-id test/recyclobot-sim-demos \
    --num-episodes 10

# Or collect with Aloha environment
python scripts/collect_recyclobot_dataset_v3.py \
    --robot-path lerobot/configs/robot/aloha.yaml \
    --repo-id test/recyclobot-aloha-demos \
    --num-episodes 10
```

## 9. Train a Model (Optional)

```bash
# Fine-tune SmolVLA on your collected data
python scripts/train_recyclobot.py \
    --dataset-name test/recyclobot-sim-demos \
    --output-dir outputs/recyclobot_finetuned \
    --use-lora \
    --num-epochs 10
```

## Troubleshooting

### If you get CUDA/GPU errors:
```bash
# Check your CUDA version
nvcc --version
nvidia-smi

# Install matching PyTorch version
# For CUDA 12.1: pip install torch --index-url https://download.pytorch.org/whl/cu121
# For CUDA 11.8: pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### If model download fails:
```bash
# Login to HuggingFace (optional, for private models)
huggingface-cli login

# Clear cache and retry
rm -rf ~/.cache/lerobot
huggingface-cli download lerobot/smolvla_base --local-dir ~/.cache/lerobot
```

### If you're behind a proxy:
```bash
# Set proxy environment variables
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
export HF_HUB_OFFLINE=1  # For offline mode after downloading
```

## Performance Tips

1. **Monitor GPU Usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Reduce memory usage** if needed:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

3. **Use smaller batch sizes** for training:
   ```bash
   python scripts/train_recyclobot.py --batch-size 4
   ```

## Next Steps

1. **Experiment with different prompts**:
   - "Pick up all plastic bottles"
   - "Sort items by color"
   - "Place objects in the correct bins"

2. **Try different environments**:
   - Aloha: Complex dual-arm tasks
   - PushT: Simple 2D tasks
   - Xarm: Single-arm manipulation

3. **Collect your own data** and train custom policies

4. **Join the community**:
   - Report issues: https://github.com/charbelsan/recyclobot/issues
   - Share your results!

---

You're all set! RecycloBot should now be running on your GPU station. The system will use SmolVLA's built-in vision-language understanding by default, making it efficient and straightforward to test.