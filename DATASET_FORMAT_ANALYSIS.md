# RecycloBot Dataset Format Analysis

## Critical Issues with Original Implementation

### 1. ❌ WRONG: Task as Frame-Level Data
**Original code:**
```python
observation["task"] = language_instruction  # WRONG!
dataset.add_frame({
    **observation,
    "action": action,
    "metadata": json.dumps(frame_metadata)  # WRONG!
})
```

**Why it's wrong:**
- LeRobot stores task descriptions at the **episode level**, not frame level
- SmolVLA expects consistent task descriptions for all frames in an episode
- Adding "task" to frame data breaks the expected schema

**Correct approach:**
```python
# Task is set once per episode
dataset.add_episode(task=task_description)

# Frames only contain observations and actions
dataset.add_frame({
    "observation.images.top": image_tensor,
    "observation.state": state_tensor,
    "action": action_tensor,
})
```

### 2. ❌ WRONG: Custom Metadata in Frames
**Original code:**
```python
dataset.add_frame({
    **observation,
    "action": action,
    "metadata": json.dumps(frame_metadata)  # NOT ALLOWED!
})
```

**Why it's wrong:**
- LeRobot has a strict schema for frame data
- Custom fields like "metadata" are not allowed
- This would break dataset loading and training

**Correct approach:**
- Store metadata separately in accompanying files
- Keep frame data clean with only standard fields

### 3. ❌ WRONG: Dataset Initialization
**Original code:**
```python
dataset = LeRobotDataset(
    repo_id=args.repo_id,
    fps=args.fps,
    robot=robot,
    root=f"data/{args.repo_id.split('/')[-1]}"
)
```

**Why it's wrong:**
- Missing proper feature definitions
- Not using LeRobotDataset.create() method
- Missing robot_type specification

**Correct approach:**
```python
dataset = LeRobotDataset.create(
    repo_id=repo_id,
    fps=fps,
    features={
        "observation.images.top": {"dtype": "video", "shape": (480, 640, 3)},
        "observation.state": {"dtype": "float32", "shape": (14,)},
        "action": {"dtype": "float32", "shape": (7,)},
    },
    robot_type="so101"
)
```

### 4. ❌ WRONG: Episode Management
**Original code:**
```python
dataset.start_episode(task=task_description)
# ... collect data ...
dataset.end_episode()
```

**Why it's wrong:**
- These methods don't exist in LeRobotDataset
- Wrong API usage

**Correct approach:**
```python
dataset.add_episode(task=task_description)
# ... collect frames ...
dataset.save_episode()
```

## Correct LeRobot Dataset Structure

### File Structure
```
dataset_root/
├── data/                          # Parquet files with frame data
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── episode_000001.parquet
├── meta/                          # Dataset metadata
│   ├── info.json                  # Dataset info
│   ├── tasks.jsonl                # Task descriptions
│   ├── episodes.jsonl             # Episode boundaries
│   └── stats.json                 # Normalization statistics
└── videos/                        # Optional video storage
    └── chunk-000/
        └── observation.images.top/
            ├── episode_000000.mp4
            └── episode_000001.mp4
```

### Task Storage Format

**meta/tasks.jsonl:**
```json
{"task_index": 0, "task": "Pick up the plastic bottle and place it in recycling bin"}
{"task_index": 1, "task": "Sort all aluminum cans into the recycling bin"}
```

**meta/episodes.jsonl:**
```json
{"episode_index": 0, "task_index": 0, "length": 247}
{"episode_index": 1, "task_index": 1, "length": 312}
```

### Frame Data Schema

Each frame in the parquet files contains:
```python
{
    # Standard fields (required)
    "timestamp": 0.033,                    # float32
    "frame_index": 0,                      # int64
    "episode_index": 0,                    # int64
    "index": 0,                           # int64
    
    # Observations
    "observation.images.top": [...],       # Video frame or tensor
    "observation.state": [...],            # Robot state (14D for SO-101)
    
    # Actions
    "action": [...],                       # Robot commands (7D for SO-101)
    
    # Task reference (NOT the task string!)
    "task_index": 0,                       # int64 - references tasks.jsonl
}
```

## SmolVLA Training Expectations

When SmolVLA loads the dataset:

1. **Task Loading:**
   ```python
   # SmolVLA internally does:
   task_string = dataset.tasks[batch["task_index"]]
   # NOT: task_string = batch["task"]  # This field doesn't exist!
   ```

2. **Observation Format:**
   ```python
   observations = {
       "pixel_values": batch["observation.images.top"],  # (B, C, H, W)
       "states": batch["observation.state"],             # (B, 14)
       "language": task_strings,                         # List of strings
   }
   ```

3. **Action Format:**
   ```python
   actions = batch["action"]  # (B, 7) for SO-101
   ```

## Best Practices for RecycloBot

1. **Use Episode-Level Tasks:**
   - One task description per episode
   - Consistent language throughout the episode
   - Natural, imperative sentences

2. **Keep Frame Data Clean:**
   - Only standard LeRobot fields
   - No custom metadata in frames
   - Store planning info separately

3. **Proper Data Types:**
   - Images: uint8 or normalized float32
   - States/Actions: float32
   - All tensors properly shaped

4. **Planning Metadata:**
   - Store in separate JSON files
   - Link to episodes by index
   - Include reasoning traces

5. **Task Variety:**
   - Multiple phrasings for same task
   - Different object combinations
   - Varied complexity levels

## Example: Correct Data Collection

```python
# 1. Create dataset with proper schema
dataset = LeRobotDataset.create(
    repo_id="user/recyclobot-v2",
    fps=30,
    features={
        "observation.images.top": {"dtype": "video", "shape": (480, 640, 3)},
        "observation.state": {"dtype": "float32", "shape": (14,)},
        "action": {"dtype": "float32", "shape": (7,)},
    },
    robot_type="so101"
)

# 2. Add episode with task
task = "Pick up all plastic bottles and place them in the recycling bin"
dataset.add_episode(task=task)

# 3. Collect frames (no task in frame data!)
for step in range(episode_length):
    obs = robot.get_observation()
    action = policy.get_action(obs, task)  # Policy uses task internally
    
    dataset.add_frame({
        "observation.images.top": obs["image"],
        "observation.state": obs["state"],
        "action": action,
    })
    
    robot.send_action(action)

# 4. Save episode
dataset.save_episode()

# 5. Store planning metadata separately
metadata = {
    "episode_idx": 0,
    "planner": "gemini",
    "skills": ["pick(plastic_bottle)", "place(recycling_bin)"],
    "reasoning": "I see plastic bottles on the table..."
}
save_planning_metadata(metadata)
```

## Migration Guide

To fix existing datasets:

1. Remove "task" field from frame data
2. Move task to episode level
3. Remove custom metadata fields
4. Ensure proper feature shapes
5. Regenerate meta files

The corrected implementation in `collect_recyclobot_dataset_v2.py` follows all these requirements.