# SmolVLA Implementation Correction

## Critical Finding: Tasks ARE Stored Per-Frame!

After analyzing the actual SmolVLA implementation, I discovered that our understanding was **incorrect**. Here's what actually happens:

### How SmolVLA Really Works

1. **Tasks ARE stored per-frame** in the dataset
2. Each frame has a `task_index` field
3. The dataset has a global task dictionary
4. When loading, the dataset automatically adds `item["task"] = self.meta.tasks[task_idx]`

### Evidence from LeRobot Code

From the SmolVLA model implementation:
```python
def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
    """Tokenize the text input"""
    tasks = batch["task"]  # Expects "task" in batch!
```

From dataset loading:
```python
# The dataset loader adds task string dynamically:
item["task"] = self.meta.tasks[task_idx]
```

### What This Means

1. **Our current implementation is WRONG** - we're not storing tasks per frame
2. **LeRobot's dataset.add_episode(task=...)** likely handles this internally
3. **The task string IS expected in the batch during inference**

### Correct Dataset Format

```python
# Each frame in the dataset should have:
{
    "timestamp": 0.033,
    "frame_index": 0,
    "episode_index": 0,
    "index": 0,
    "task_index": 0,  # This maps to task dictionary
    
    # When loaded, the dataset adds:
    "task": "Pick up the plastic bottle",  # Added dynamically from task_index
    
    # Observations
    "observation.images.top": tensor([3, 480, 640]),
    "observation.state": tensor([14]),
    
    # Actions
    "action": tensor([7]),
}
```

### Why Our Implementation Might Still Work

If we're using `dataset.add_episode(task=task_description)`, LeRobot might be:
1. Automatically adding the task to the task dictionary
2. Setting the correct `task_index` for all frames in that episode
3. Handling the task string injection when loading

### Key Differences from Our Understanding

1. **Task flexibility**: SmolVLA actually supports different tasks per frame (though typically constant per episode)
2. **Storage method**: Tasks are referenced by index, with strings stored separately
3. **Loading behavior**: The task string is injected during dataset loading, not stored directly

### Implications for RecycloBot

1. **For inference**: We MUST pass "task" in the observation dict
2. **For training**: The dataset loader will handle task injection
3. **For collection**: Using `dataset.add_episode(task=...)` should work correctly

### Verification Needed

We should verify that LeRobot's `add_episode(task=...)` method:
1. Adds the task to the global task dictionary
2. Sets the correct task_index for all frames in the episode
3. Properly formats the data for SmolVLA training

### Correct Policy Inference Code

```python
# For inference, we MUST include task in the observation:
policy_obs = {
    "observation.images.top": image_tensor,
    "observation.state": state_tensor,
    "task": "Pick up the plastic bottle and place it in recycling bin"  # REQUIRED!
}
action = policy.select_action(policy_obs)
```

### Summary

Our implementation needs adjustment:
- ✅ We correctly pass "task" during inference in skill_runner.py
- ❓ We need to verify dataset.add_episode() sets task_index correctly
- ❌ Our documentation incorrectly states tasks aren't in frame data
- ✅ We correctly don't manually add "task" to frame data (dataset handles it)