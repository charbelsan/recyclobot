# 🚀 RecycloBot Quick Reference Card

## Essential Commands

### 1️⃣ Update & Test
```bash
git pull
python quick_validate.py
```

### 2️⃣ Run Robot Demo
```bash
# Basic run
python examples/run_recyclobot_demo.py --robot so101

# Custom prompt
python examples/run_recyclobot_demo.py --robot so101 --prompt "Pick up the red can"

# Slower movements (safer)
python examples/run_recyclobot_demo.py --robot so101 --fps 5
```

### 3️⃣ Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| Robot not found | `sudo chmod 666 /dev/ttyUSB0` |
| GPU memory error | `export CUDA_VISIBLE_DEVICES=0` |
| Multi-GPU device error | Set `CUDA_VISIBLE_DEVICES=0` (SmolVLA requires single GPU) |
| Model download stuck | `rm -rf ~/.cache/huggingface/hub/models--lerobot--smolvla_base` |
| Robot moves too fast | Add `--fps 5` to command |

### 4️⃣ Environment Variables
```bash
# For Gemini planner
export GEMINI_API_KEY="your-key"

# For OpenAI planner  
export OPENAI_API_KEY="your-key"

# Force single GPU (REQUIRED for SmolVLA)
export CUDA_VISIBLE_DEVICES=0
```

### 5️⃣ Safety Checklist
- [ ] Clear workspace of fragile items
- [ ] Bins within robot reach
- [ ] Power switch accessible
- [ ] Test with empty bottles first
- [ ] Someone watching robot

### 6️⃣ Test Sequence
1. `python quick_validate.py` - Test model
2. `python examples/run_recyclobot_demo.py` - Test simulation
3. `python examples/run_recyclobot_demo.py --robot so101 --fps 5` - Test real robot slowly
4. `python examples/run_recyclobot_demo.py --robot so101` - Normal speed

---
**Full guide**: See `ROBOT_TESTING_GUIDE.md`