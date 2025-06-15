"""
RecycloBot Dataset Logger
Records robot trajectories with planning metadata for recycling tasks.
Uses HuggingFace datasets for efficient storage and sharing.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Features, Image, Sequence, Value


# Dataset schema with recycling-specific fields
RECYCLOBOT_FEATURES = Features({
    # Standard robot data
    "episode_id": Value("int32"),
    "step_id": Value("int32"),
    "timestamp": Value("float64"),
    "image": Image(),  # HF datasets Image feature handles encoding
    "state": Sequence(Value("float32")),  # Joint positions
    "action": Sequence(Value("float32")),  # Joint velocities/torques
    
    # RecycloBot specific
    "planner_name": Value("string"),  # "gemini" or "qwen"
    "planner_log": Value("string"),   # Full skill sequence as JSON string
    "current_skill": Value("string"),  # Current executing skill
    "task": Value("string"),  # Natural language instruction for SmolVLA (expects 'task' key)
    
    # Recycling metrics
    "detected_objects": Value("string"),  # JSON list of detected items
    "target_bin": Value("string"),        # Target bin for current action
    "skill_completed": Value("bool"),     # Whether skill finished
})


class RecycloBotLogger:
    """Dataset logger for recycling robot demonstrations."""
    
    def __init__(self, output_dir: str, dataset_name: str = "recyclobot_demo"):
        """
        Initialize logger.
        
        Args:
            output_dir: Directory to save dataset
            dataset_name: Name for the dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_name = dataset_name
        self.episode_id = 0
        self.step_id = 0
        
        # Buffers for current episode
        self.episode_buffer = {key: [] for key in RECYCLOBOT_FEATURES.keys()}
        
        # Metadata
        self.metadata = {
            "dataset_name": dataset_name,
            "created_at": datetime.now().isoformat(),
            "features": RECYCLOBOT_FEATURES.to_dict(),
            "recycling_categories": ["recycling", "compost", "trash"],
            "recorded_episodes": []
        }
        
    def start_episode(self, planner_name: str, planner_log: List[str]):
        """
        Start a new episode.
        
        Args:
            planner_name: Name of planner used ("gemini" or "qwen")
            planner_log: Full skill sequence from planner
        """
        self.episode_id += 1
        self.step_id = 0
        self.episode_buffer = {key: [] for key in RECYCLOBOT_FEATURES.keys()}
        
        # Store episode metadata
        episode_meta = {
            "episode_id": self.episode_id,
            "planner_name": planner_name,
            "skill_sequence": planner_log,
            "start_time": datetime.now().isoformat(),
            "total_skills": len(planner_log)
        }
        self.metadata["recorded_episodes"].append(episode_meta)
        
        print(f"Started episode {self.episode_id} with {len(planner_log)} skills")
        
    def record(self, obs: Dict[str, Any], action: Optional[np.ndarray], 
               done: bool, extra: Dict[str, Any]):
        """
        Record a single timestep.
        
        Args:
            obs: Observation dict with 'image' and 'state' keys
            action: Action array (can be None for first step)
            done: Whether episode is complete
            extra: Additional metadata (current_skill, goal_id, etc.)
        """
        self.step_id += 1
        
        # Process image (resize to standard size)
        if isinstance(obs.get("image"), np.ndarray):
            image = cv2.resize(obs["image"], (224, 224))
        else:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Process state
        state = obs.get("state", np.zeros(6))  # SmolVLA uses 6-dim state
        
        # Handle torch tensors
        if hasattr(state, "cpu"):
            state = state.cpu().numpy()
        
        # Convert to list
        if hasattr(state, "tolist"):
            state = state.tolist()
        elif isinstance(state, (list, tuple)):
            state = list(state)
        elif isinstance(state, np.ndarray):
            state = state.tolist()
        else:
            state = [float(state)]
        
        # Process action
        if action is None:
            action = [0.0] * len(state)  # Create list of zeros matching state length
        else:
            # Handle torch tensors
            if hasattr(action, "cpu"):
                action = action.cpu().numpy()
            
            # Convert to list
            if hasattr(action, "tolist"):
                action = action.tolist()
            elif isinstance(action, (list, tuple)):
                action = list(action)
            elif isinstance(action, np.ndarray):
                action = action.tolist()
            else:
                action = [float(action)]
        
        # Add to buffer
        self.episode_buffer["episode_id"].append(self.episode_id)
        self.episode_buffer["step_id"].append(self.step_id)
        self.episode_buffer["timestamp"].append(self.step_id * 0.1)  # 10Hz
        self.episode_buffer["image"].append(image)
        self.episode_buffer["state"].append(state)
        self.episode_buffer["action"].append(action)
        
        # Add RecycloBot specific fields
        self.episode_buffer["planner_name"].append(extra.get("planner_name", ""))
        self.episode_buffer["planner_log"].append(extra.get("planner_log", "[]"))
        self.episode_buffer["current_skill"].append(extra.get("current_skill", ""))
        self.episode_buffer["task"].append(extra.get("task", extra.get("language_instruction", "")))
        
        # Recycling metrics
        self.episode_buffer["detected_objects"].append(extra.get("detected_objects", "[]"))
        self.episode_buffer["target_bin"].append(extra.get("target_bin", ""))
        # Handle skill_completed which might be a string (skill name) or bool
        skill_completed = extra.get("skill_completed", False)
        if isinstance(skill_completed, str):
            # If it's a skill name string, treat as True
            self.episode_buffer["skill_completed"].append(True)
        else:
            self.episode_buffer["skill_completed"].append(bool(skill_completed))
        
        # Save episode if done
        if done:
            self.save_episode()
            
    def save_episode(self):
        """Save current episode buffer to disk."""
        if not any(len(v) > 0 for v in self.episode_buffer.values()):
            return
        
        # Debug: Check for problematic data
        for key in ["state", "action"]:
            if key in self.episode_buffer and self.episode_buffer[key]:
                for i, item in enumerate(self.episode_buffer[key]):
                    if not isinstance(item, list):
                        print(f"WARNING: {key}[{i}] is not a list: type={type(item)}, value={item}")
                        # Try to fix it
                        if hasattr(item, "tolist"):
                            self.episode_buffer[key][i] = item.tolist()
                        elif isinstance(item, (tuple, np.ndarray)):
                            self.episode_buffer[key][i] = list(item)
                        else:
                            self.episode_buffer[key][i] = [item]
                    # Also check for nested lists
                    elif isinstance(item, list) and item and isinstance(item[0], list):
                        print(f"WARNING: {key}[{i}] is a nested list, flattening...")
                        self.episode_buffer[key][i] = item[0]
            
        # Create HuggingFace dataset from buffer
        try:
            dataset = Dataset.from_dict(self.episode_buffer, features=RECYCLOBOT_FEATURES)
        except TypeError as e:
            print(f"\nError creating dataset: {e}")
            print("\nExpected features schema:")
            for key, feature in RECYCLOBOT_FEATURES.items():
                print(f"  {key}: {feature}")
            print("\nDebugging episode buffer:")
            for key, values in self.episode_buffer.items():
                if values:
                    print(f"\n{key}:")
                    print(f"  Length: {len(values)}")
                    print(f"  First item type: {type(values[0])}")
                    print(f"  First item value: {repr(values[0])[:100]}...")  # Show first 100 chars
                    if isinstance(values[0], list) and values[0]:
                        print(f"  First item[0] type: {type(values[0][0])}")
            raise
        
        # Save as parquet
        episode_path = self.output_dir / f"episode_{self.episode_id:04d}.parquet"
        dataset.to_parquet(str(episode_path))
        
        # Also save as arrow for compatibility
        arrow_path = self.output_dir / f"episode_{self.episode_id:04d}.arrow" 
        dataset.save_to_disk(str(arrow_path))
        
        print(f"Saved episode {self.episode_id} with {len(self.episode_buffer['step_id'])} steps")
        
        # Update and save metadata
        self.metadata["recorded_episodes"][-1]["end_time"] = datetime.now().isoformat()
        self.metadata["recorded_episodes"][-1]["total_steps"] = len(self.episode_buffer['step_id'])
        self.save_metadata()
        
    def save_metadata(self):
        """Save dataset metadata."""
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
            
    def create_dataset_card(self):
        """Create a dataset card for HuggingFace Hub."""
        # Gather metadata safely
        episodes = self.metadata.get('recorded_episodes', [])
        planners = set(ep.get('planner_name', 'unknown') for ep in episodes) if episodes else {'unknown'}
        num_episodes = len(episodes)
        total_steps = sum(ep.get('total_steps', 0) for ep in episodes) if episodes else 0
        
        card_content = f"""---
task_categories:
- robotics
- recycling
tags:
- lerobot
- recyclobot
- waste-sorting
- so-arm100
size_categories:
- n<1K
---

# {self.dataset_name}

## Dataset Description

This dataset contains robot demonstrations for recycling/waste sorting tasks using the RecycloBot system.

### Dataset Summary

- **Robot**: SO-ARM100 (6-DOF robotic arm)
- **Task**: Waste sorting into recycling, compost, and trash bins
- **Planner**: {', '.join(planners)}
- **Episodes**: {num_episodes}
- **Total Steps**: {total_steps}

### Supported Tasks

The dataset includes demonstrations of:
- Picking up various waste items (bottles, cans, food waste)
- Placing items in appropriate bins
- Inspecting items to determine type
- General sorting behaviors

## Dataset Structure

### Data Fields

- `episode_id`: Episode identifier
- `step_id`: Step within episode
- `timestamp`: Time in seconds
- `image`: RGB camera observation (224x224)
- `state`: Joint positions (7-DOF)
- `action`: Joint commands (7-DOF)
- `planner_name`: Vision-language planner used
- `planner_log`: Full skill sequence
- `current_skill`: Currently executing skill
- `language_instruction`: Natural language instruction for SmolVLA
- `detected_objects`: Objects identified in scene
- `target_bin`: Target bin for placement
- `skill_completed`: Skill completion flag

### Data Splits

Currently contains a single split with all demonstrations.

## Additional Information

### Dataset Curators

RecycloBot Team - AI Hackathon 2024

### Licensing Information

MIT License

### Citation Information

```bibtex
@misc{{recyclobot2024,
  title={{RecycloBot: Vision-Language Planning for Waste Sorting}},
  author={{RecycloBot Team}},
  year={{2024}},
  publisher={{HuggingFace}}
}}
```
"""
        
        card_path = self.output_dir / "README.md"
        with open(card_path, "w") as f:
            f.write(card_content)
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_episodes": len(self.metadata["recorded_episodes"]),
            "total_steps": sum(ep['total_steps'] for ep in self.metadata['recorded_episodes']),
            "planners_used": list(set(ep['planner_name'] for ep in self.metadata['recorded_episodes'])),
            "average_episode_length": 0
        }
        
        if stats["total_episodes"] > 0:
            stats["average_episode_length"] = stats["total_steps"] / stats["total_episodes"]
            
        # Count skills used
        skill_counts = {}
        for ep in self.metadata["recorded_episodes"]:
            for skill in ep.get("skill_sequence", []):
                base_skill = skill.split("(")[0]
                skill_counts[base_skill] = skill_counts.get(base_skill, 0) + 1
        stats["skill_usage"] = skill_counts
        
        return stats