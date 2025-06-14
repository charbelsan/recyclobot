# RecycloBot Correct Usage Guide

## Overview

RecycloBot must integrate properly with LeRobot's data format to work with SmolVLA. This guide explains the correct way to collect, store, and use datasets.

## Key Principles

### 1. Task Descriptions Are Episode-Level
- ✅ One task description per episode
- ✅ Task stored in metadata, not frame data
- ❌ Don't change task within an episode
- ❌ Don't store task in every frame

### 2. Frame Data Has Fixed Schema
- ✅ Only standard fields: observations, actions, timestamps
- ❌ No custom fields like "metadata" or "language_instruction"
- ❌ No planning information in frames

### 3. SmolVLA Expects Natural Language
- ✅ Use clear imperative sentences
- ✅ Specify WHAT to manipulate: "pick up the plastic bottle"
- ❌ Don't use abstract commands: "execute skill 1"
- ❌ Don't use goal IDs

## Correct Data Collection Methods

### Method 1: Autonomous Collection (Recommended for RecycloBot)

```bash
# Use the corrected script
python scripts/collect_recyclobot_dataset_v2.py \
    --robot-path lerobot/configs/robot/so101.yaml \
    --repo-id your-username/recyclobot-demos \
    --autonomous \
    --planner gemini \
    --num-episodes 50 \
    --tasks-file recycling_tasks.json
```

**What happens:**
1. Planner analyzes scene and generates skill sequence
2. Skills are converted to natural language for SmolVLA
3. Episode is recorded with consistent task description
4. Planning metadata saved separately

### Method 2: Teleoperated Collection

```bash
# Option A: Use LeRobot directly
python -m lerobot.record \
    --robot-path lerobot/configs/robot/so101.yaml \
    --repo-id your-username/recyclobot-teleop \
    --num-episodes 50 \
    --single-task "Sort all items into recycling bins"

# Option B: Use RecycloBot wrapper for planning analysis
python scripts/record_recyclobot_teleoperated.py \
    --robot-path lerobot/configs/robot/so101.yaml \
    --repo-id your-username/recyclobot-teleop \
    --num-episodes 50 \
    --tasks-file recycling_tasks.json \
    --analyze-after \
    --planner gemini
```

### Method 3: Mixed Collection

```bash
# First collect some teleoperated demos
python -m lerobot.record \
    --robot-path lerobot/configs/robot/so101.yaml \
    --repo-id your-username/recyclobot-mixed \
    --num-episodes 20

# Then add autonomous demos to same dataset
python scripts/collect_recyclobot_dataset_v2.py \
    --robot-path lerobot/configs/robot/so101.yaml \
    --repo-id your-username/recyclobot-mixed \
    --autonomous \
    --num-episodes 30 \
    --start-episode 20  # Continue from episode 20
```

## Dataset Structure

### Correct Structure
```
recyclobot-demos/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  # Contains frames
│       └── episode_000001.parquet
├── meta/
│   ├── info.json                   # Dataset info
│   ├── tasks.jsonl                 # Task descriptions
│   ├── episodes.jsonl              # Episode boundaries
│   └── stats.json                  # Normalization stats
├── videos/                         # Optional
│   └── observation.images.top/
│       └── episode_000000.mp4
└── planning_metadata/              # RecycloBot addition
    ├── episode_000000.json         # Planning details
    └── episode_000001.json
```

### Frame Data (in Parquet files)
```python
{
    # Standard fields only
    "timestamp": 0.033,
    "frame_index": 0,
    "episode_index": 0,
    "index": 0,
    "task_index": 0,  # References meta/tasks.jsonl
    
    # Observations
    "observation.images.top": tensor([3, 480, 640]),
    "observation.state": tensor([14]),  # SO-101 state
    
    # Actions
    "action": tensor([7]),  # SO-101 commands
}
```

### Task Storage (meta/tasks.jsonl)
```json
{"task_index": 0, "task": "Sort all the trash on the table into appropriate bins"}
{"task_index": 1, "task": "Pick up plastic bottles and place them in recycling"}
```

### Planning Metadata (planning_metadata/episode_000000.json)
```json
{
    "episode_idx": 0,
    "task_description": "Sort all the trash on the table into appropriate bins",
    "planner_name": "gemini",
    "skill_sequence": [
        "pick(plastic_bottle)",
        "place(recycling_bin)",
        "pick(aluminum_can)",
        "place(recycling_bin)"
    ],
    "reasoning": "I can see a plastic bottle and aluminum can that need to be recycled...",
    "total_steps": 247,
    "timestamp": 1234567890.123
}
```

## Training SmolVLA

### Using LeRobot's Training Script
```bash
python -m lerobot.scripts.train \
    policy=smolvla \
    dataset_repo_id=your-username/recyclobot-demos \
    hydra.run.dir=outputs/train_recyclobot \
    training.num_epochs=100 \
    training.batch_size=8 \
    training.learning_rate=1e-5 \
    policy.use_lora=true
```

### Using RecycloBot's Training Script
```bash
python scripts/train_recyclobot.py \
    --dataset-name your-username/recyclobot-demos \
    --output-dir outputs/recyclobot_smolvla \
    --use-lora \
    --num-epochs 50
```

## Common Mistakes to Avoid

### ❌ Wrong: Changing task within episode
```python
# DON'T DO THIS
for skill in skills:
    observation["task"] = skill_to_language(skill)  # Changes every skill!
    dataset.add_frame(observation)
```

### ✅ Correct: Consistent task per episode
```python
# DO THIS
dataset.add_episode(task="Sort items into recycling bins")
for skill in skills:
    # Task is already set at episode level
    dataset.add_frame({
        "observation.images.top": image,
        "observation.state": state,
        "action": action
    })
```

### ❌ Wrong: Custom fields in frames
```python
# DON'T DO THIS
dataset.add_frame({
    "observation.images.top": image,
    "observation.state": state,
    "action": action,
    "current_skill": "pick(bottle)",  # NOT ALLOWED!
    "planner_output": skills,         # NOT ALLOWED!
})
```

### ✅ Correct: Only standard fields
```python
# DO THIS
dataset.add_frame({
    "observation.images.top": image,
    "observation.state": state,
    "action": action,
})
# Save custom data separately
save_metadata({"current_skill": "pick(bottle)"})
```

## Verification Checklist

Before training or deploying:

- [ ] Dataset loads without errors: `dataset = load_dataset("user/recyclobot-demos")`
- [ ] All episodes have task descriptions in `meta/tasks.jsonl`
- [ ] Frame data contains only standard fields
- [ ] Images are properly formatted (C,H,W) tensors
- [ ] State and action dimensions match robot config
- [ ] At least 50 episodes for fine-tuning
- [ ] Planning metadata stored separately (not in dataset)

## Quick Test

```python
from datasets import load_dataset

# Test loading
dataset = load_dataset("your-username/recyclobot-demos", split="train")
print(f"Dataset size: {len(dataset)}")
print(f"Features: {dataset.features}")

# Check first sample
sample = dataset[0]
assert "observation.images.top" in sample
assert "observation.state" in sample
assert "action" in sample
assert "task_index" in sample
assert "task" not in sample  # Task is NOT in frame data!

print("Dataset format is correct!")
```

Following this guide ensures your RecycloBot datasets work correctly with SmolVLA and LeRobot's training pipeline.