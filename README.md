# RecycloBot 🤖♻️

**Vision-Language Planning for Robotic Waste Sorting**

RecycloBot adds intelligent waste sorting capabilities to [LeRobot](https://github.com/huggingface/lerobot) using vision-language models for high-level planning.

## 🎯 Key Features

- **Vision-Language Planning**: Multiple AI models analyze scenes and generate sorting sequences
  - Google Gemini, OpenAI GPT-4V, Anthropic Claude, Qwen-VL (local)
  - Any OpenAI-compatible API (Ollama, vLLM, Together.ai)
- **SmolVLA Execution**: Natural language instructions → robot actions
- **Proper Dataset Format**: LeRobot-compatible data collection for training
- **Multi-Robot Support**: SO-ARM100 hardware and simulation (Aloha, PushT, Xarm gym environments)

## 🚀 Quick Start

### Installation

```bash
# Create environment
conda create -n recyclobot python=3.10 -y
conda activate recyclobot

# Install LeRobot with SmolVLA support
pip install "lerobot[smolvla,feetech]>=0.5.0"

# Download SmolVLA weights (REQUIRED!)
huggingface-cli download lerobot/koch_aloha --local-dir ~/.cache/lerobot

# Install RecycloBot
git clone https://github.com/your-username/recyclobot.git
cd recyclobot
pip install -e .

# Optional: Install planners
pip install google-generativeai  # For Gemini
pip install "transformers[vision]>=4.44.0" "accelerate>=0.26.0"  # For Qwen
```

### Basic Usage

```bash
# 1. Calibrate robot (first time only)
python -m lerobot.scripts.control_robot --robot.type so101 --control.type calibrate

# 2. Test teleoperation
python -m lerobot.scripts.control_robot --robot.type so101 --control.type teleoperate

# 3. Run autonomous demo (uses SmolVLA directly by default)
python examples/run_recyclobot_demo.py --robot so101 --prompt "Sort the recycling"

# Optional: Use external planner like Gemini
export GEMINI_API_KEY="your-key"
python examples/run_recyclobot_demo.py --robot so101 --planner gemini --prompt "Sort the recycling"

# 4. Collect dataset (CORRECTED VERSION!)
python scripts/collect_recyclobot_dataset_v3.py \
    --robot-path lerobot/configs/robot/so101.yaml \
    --repo-id your-username/recyclobot-demos \
    --autonomous --planner gemini \
    --num-episodes 50
```

## 📊 System Architecture

### Default: Direct SmolVLA Execution (Recommended)
```
User Task: "Sort the recycling"
    ↓
SmolVLA Policy (Vision + Language → Actions)
    ↓
Robot Execution (SO-101)
```

Since SmolVLA = SmolVLM + Action Expert, it already has vision-language understanding built-in!

### Alternative: Two-Stage Planning (Optional)
```
User Task: "Sort the recycling"
    ↓
Vision-Language Planner (Gemini/GPT-4V/Qwen)
    ↓
Skills: ["pick(plastic_bottle)", "place(recycling_bin)"]
    ↓
Natural Language: "pick up the plastic bottle"
    ↓
SmolVLA Policy (Vision + Language → Actions)
    ↓
Robot Execution (SO-101)
```

Use this if you want explicit planning steps or need to use cloud-based models.

### How SmolVLA Works

SmolVLA is a Vision-Language-Action model that:
1. Contains SmolVLM (500M param VLM) for understanding
2. Takes RGB images + natural language instructions
3. Outputs continuous robot actions
4. The language specifies WHAT to manipulate (e.g., "pick up the red block")
5. Uses vision to understand WHERE and HOW

### Usage Examples

```bash
# Default: Direct SmolVLA execution (no separate planner)
python examples/run_recyclobot_demo.py --robot so101 --prompt "Sort all the recycling"

# Optional: Use external planner
python examples/run_recyclobot_demo.py --robot so101 --planner gemini --prompt "Sort all the recycling"
```

## 🗂️ Critical: Dataset Format

### ⚠️ Important Discovery

After analyzing LeRobot's implementation, we found that:
- Tasks ARE stored **per-frame** (via `task_index`), not per-episode
- The dataset loader automatically injects task strings during training
- SmolVLA expects `batch["task"]` during training/inference

### Correct Data Collection

```python
# ✅ CORRECT API (v3):
dataset.add_frame(frame_data, task="Pick up the plastic bottle")
dataset.save_episode()

# ❌ WRONG (our initial understanding):
dataset.add_episode(task=task)  # This method doesn't exist!
dataset.add_frame(frame_data)   # Missing task parameter!
```

### Dataset Structure

```
dataset/
├── data/                    # Parquet files with frame data
│   └── chunk-000/
│       └── episode_000000.parquet
├── meta/                    # Metadata files
│   ├── tasks.jsonl         # Task dictionary
│   ├── episodes.jsonl      # Episode boundaries
│   └── stats.json          # Normalization stats
└── videos/                  # Optional video storage
```

**Frame data contains:**
```python
{
    "task_index": 0,  # References meta/tasks.jsonl
    "observation.images.top": tensor([3, 480, 640]),
    "observation.state": tensor([14]),  # SO-101: 7 joints x 2
    "action": tensor([7]),
    # NO "task" field - added dynamically during loading!
}
```

**Task storage (meta/tasks.jsonl):**
```json
{"task_index": 0, "task": "Pick up the plastic bottle and place it in recycling"}
{"task_index": 1, "task": "Sort all aluminum cans into the recycling bin"}
```

## 🤖 SmolVLA Integration

### Critical Fixes Made

1. **Observation Format**: Use `"observation.images.top"` (not `"main_camera"`)
2. **State Dimensions**: SO-101 has 14 dims (7 joints × 2 for pos+vel)
3. **Model Loading**: Use `lerobot/koch_aloha` (a real pretrained model)
4. **Task Handling**: Pass task during inference AND data collection

### Correct Inference Code

```python
# For policy inference
observation = {
    "observation.images.top": image_tensor,  # (C,H,W) normalized
    "observation.state": state_tensor,       # (14,) for SO-101
    "task": "pick up the plastic bottle"     # REQUIRED!
}
action = policy.select_action(observation)
```

## 🎓 Training SmolVLA

### Option 1: LeRobot Native Training
```bash
python -m lerobot.scripts.train \
    policy=smolvla \
    dataset_repo_id=your-username/recyclobot-demos \
    hydra.run.dir=outputs/train_recyclobot \
    training.num_epochs=100 \
    training.batch_size=8 \
    policy.use_lora=true
```

### Option 2: RecycloBot Script
```bash
python scripts/train_recyclobot.py \
    --dataset-name your-username/recyclobot-demos \
    --output-dir outputs/recyclobot_smolvla \
    --use-lora \
    --num-epochs 50
```

## 🔧 Configuration

Edit `recyclobot/config.yaml`:

```yaml
planners:
  openai:
    api_key: "sk-..."  # or use env var
    model: "gpt-4-vision-preview"
    
  ollama:
    api_base: "http://localhost:11434/v1"
    model: "llava:13b"
    
  anthropic:
    api_key: "your-key"
    model: "claude-3-opus-20240229"
```

## 🚨 Troubleshooting

### SmolVLA Won't Load
```bash
# Download weights manually
huggingface-cli download lerobot/koch_aloha --local-dir ~/.cache/lerobot
```

### Dataset Issues
- Ensure using `collect_recyclobot_dataset_v3.py` (not v2!)
- Check task is passed to `add_frame(data, task=...)`
- Verify `meta/tasks.jsonl` exists after collection

### Camera Problems
```bash
python -m lerobot.find_cameras  # Find correct index
# Use --robot-overrides cameras.top.index=1
```

## 📁 Project Structure

```
recyclobot/
├── recyclobot/              # Core modules
│   ├── planning/           # Vision-language planners
│   ├── control/            # Skill execution
│   └── logging/            # Dataset management
├── scripts/                # Data collection & training
├── examples/               # Demo scripts
└── tests/                  # Unit tests
```

## 🧪 Testing

```bash
pytest tests/
```

## 📚 Key Scripts

- `collect_recyclobot_dataset_v3.py`: Correct dataset collection
- `train_recyclobot.py`: Fine-tuning with LoRA
- `evaluate_recyclobot.py`: System evaluation
- `run_recyclobot_demo.py`: Live demo

## 🔍 Implementation Details

### Skill Mapping
Skills are converted to natural language for SmolVLA:
- `pick(plastic_bottle)` → `"pick up the plastic bottle"`
- `place(recycling_bin)` → `"place the object in the recycling bin"`

### Planning Metadata
Stored separately from dataset:
```json
{
    "planner_name": "gemini",
    "skill_sequence": ["pick(bottle)", "place(bin)"],
    "reasoning": "I see a plastic bottle that needs recycling..."
}
```

## 🤝 Contributing

Contributions welcome! Key areas:
- New planners (vision-language models)
- Additional recycling skills
- Multi-language support
- Simulation environments

## 📄 License

MIT License

## 🙏 Acknowledgments

- Built on [LeRobot](https://github.com/huggingface/lerobot) by HuggingFace
- Uses [SmolVLA](https://huggingface.co/blog/smolvla) for vision-language-action
- Hardware: [SO-ARM100](https://www.seeedstudio.com/blog/2024/05/09/getting-started-with-the-so-arm100-smart-robotic-arm-now-available-on-seeed/) by Seeed Studio

---

**RecycloBot** - Teaching robots to sort waste intelligently! 🌍