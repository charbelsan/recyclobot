# Final Dataset Format Corrections for RecycloBot

## Critical Discovery: How LeRobot Actually Handles Tasks

After deep analysis of the SmolVLA training notebook and LeRobot implementation, here's the correct understanding:

### 1. The Correct API Usage

**❌ WRONG (our v2 implementation):**
```python
dataset.add_episode(task=task_description)  # This method doesn't exist!
dataset.add_frame(frame_data)  # Missing task parameter!
```

**✅ CORRECT (v3 implementation):**
```python
# No add_episode() call - episodes are implicit
dataset.add_frame(frame_data, task=task_description)  # Pass task separately!
dataset.save_episode()  # This finalizes the episode
```

### 2. How Tasks Are Actually Stored

1. **During Collection**: Pass task string to `add_frame(frame_dict, task="...")`
2. **During Saving**: `save_episode()` extracts unique tasks and assigns indices
3. **In Storage**: Only `task_index` is stored in parquet files
4. **During Loading**: Dataset adds `item["task"] = self.meta.tasks[task_idx]`

### 3. What SmolVLA Expects

From the actual implementation:
```python
# In SmolVLA's prepare_language():
tasks = batch["task"]  # Expects task string in batch!
```

This means:
- During training: Dataset loader injects task strings
- During inference: We MUST pass "task" in the observation dict

### 4. Key Corrections Needed

#### In `collect_recyclobot_dataset_v3.py`:
```python
# CORRECT: Pass task as separate parameter
dataset.add_frame(frame_data, task=frame_task)

# For inference, still need task in observation:
policy_obs = {
    "observation.images.top": image,
    "observation.state": state,
    "task": language_instruction  # Required for inference!
}
```

#### In `skill_runner.py`:
Already correct! It properly passes task during inference:
```python
obs_formatted = {
    "observation.images.top": image,
    "observation.state": state,
    "task": language_instruction  # ✅ Correct!
}
```

### 5. Important Insights

1. **Task Flexibility**: LeRobot supports different tasks per frame within an episode
2. **Episode Consistency**: Typically, all frames in an episode have the same task
3. **Language Variations**: You can experiment with skill-specific language per frame
4. **Storage Efficiency**: Task strings are deduplicated via indexing

### 6. Dataset Structure Summary

**What gets stored in parquet files:**
```python
{
    "timestamp": 0.033,
    "frame_index": 0,
    "episode_index": 0,
    "index": 0,
    "task_index": 0,  # Integer reference to task
    "observation.images.top": [...],
    "observation.state": [...],
    "action": [...],
}
```

**What gets returned by dataset[idx]:**
```python
{
    # All the above fields PLUS:
    "task": "Pick up the plastic bottle",  # Added by dataset loader!
}
```

### 7. Verification Tests

To verify correct implementation:
```python
# Test 1: Check dataset saving
dataset = LeRobotDataset.create(...)
dataset.add_frame({"observation.images.top": img, ...}, task="Pick up bottle")
dataset.save_episode()
assert len(dataset.meta.tasks) > 0  # Should have tasks

# Test 2: Check dataset loading
item = dataset[0]
assert "task" in item  # Should have task string
assert isinstance(item["task"], str)  # Should be string, not index

# Test 3: Check metadata files
tasks_file = Path(dataset.root) / "meta" / "tasks.jsonl"
assert tasks_file.exists()  # Should have task dictionary
```

### 8. Migration from v2 to v3

For existing code using v2:
1. Remove `dataset.add_episode()` calls
2. Add `task=...` parameter to all `dataset.add_frame()` calls
3. Ensure `dataset.save_episode()` is called after each episode
4. Keep passing "task" in observations for inference

### 9. Why This Matters

- **Training**: SmolVLA training expects task strings in batches
- **Generalization**: Proper task handling enables multi-task learning
- **Flexibility**: Supports experimenting with different instruction phrasings
- **Efficiency**: Deduplication saves storage space

### 10. Summary of Changes

| Component | v2 (Wrong) | v3 (Correct) |
|-----------|------------|--------------|
| Episode start | `dataset.add_episode(task=...)` | No explicit call |
| Frame adding | `dataset.add_frame(data)` | `dataset.add_frame(data, task=...)` |
| Task storage | Thought it was episode-level | Actually per-frame with dedup |
| Documentation | Said "no task in frames" | Tasks ARE in frames (via index) |

The v3 implementation in `collect_recyclobot_dataset_v3.py` correctly follows LeRobot's actual API and will produce datasets compatible with SmolVLA training.