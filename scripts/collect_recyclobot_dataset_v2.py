#!/usr/bin/env python
"""
Collect RecycloBot dataset in proper LeRobot format with vision-language planning

This script properly integrates with LeRobot's data collection pipeline
while adding RecycloBot's planning capabilities.

Usage:
    # Teleoperated collection with planning annotation
    python scripts/collect_recyclobot_dataset_v2.py \
        --robot-path lerobot/configs/robot/so101.yaml \
        --repo-id your-username/recyclobot-demos \
        --num-episodes 50
        
    # Autonomous collection with planner
    python scripts/collect_recyclobot_dataset_v2.py \
        --robot-path lerobot/configs/robot/so101.yaml \
        --repo-id your-username/recyclobot-demos \
        --autonomous \
        --planner gemini
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# Add parent directory for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from recyclobot.control.skill_runner import SkillRunner
from recyclobot.planning.gemini_planner import plan as gemini_plan
from recyclobot.planning.qwen_planner import plan as qwen_plan


def create_lerobot_dataset(repo_id: str, robot_config: Dict, fps: int = 30):
    """
    Create a LeRobot dataset with proper format for SmolVLA.
    
    Args:
        repo_id: HuggingFace repo ID
        robot_config: Robot configuration dictionary
        fps: Recording frequency
        
    Returns:
        LeRobotDataset instance
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    
    # Define features based on robot config
    # SO-101 specific configuration
    features = {
        "observation.images.top": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32", 
            "shape": (14,),  # 7 joints x 2 (pos + vel)
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),  # 7 joint commands
            "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"],
        },
    }
    
    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type="so101"
    )
    
    return dataset


def collect_episode_with_planning(
    robot,
    dataset,
    planner_fn,
    planner_name: str,
    task_description: str,
    episode_idx: int,
    use_language_per_frame: bool = False
):
    """
    Collect a single episode with planning annotations in LeRobot format.
    
    Args:
        robot: Robot instance
        dataset: LeRobotDataset instance
        planner_fn: Planning function
        planner_name: Name of planner
        task_description: Natural language task
        episode_idx: Episode number
        use_language_per_frame: Whether to use different language per skill
    """
    print(f"\n{'='*60}")
    print(f"Episode {episode_idx + 1}")
    print(f"Task: {task_description}")
    print(f"{'='*60}")
    
    # Reset robot and get initial observation
    obs_dict, info = robot.reset()
    
    # Extract image for planning
    if "observation.images.top" in obs_dict:
        image_tensor = obs_dict["observation.images.top"]
        if isinstance(image_tensor, torch.Tensor):
            # Convert to numpy: (C, H, W) -> (H, W, C)
            image_np = image_tensor.permute(1, 2, 0).numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_tensor
        image = Image.fromarray(image_np)
    else:
        logging.warning("No camera image found, using placeholder")
        image = Image.new('RGB', (640, 480))
    
    # Generate plan
    print(f"Planning with {planner_name}...")
    try:
        skills = planner_fn(image, task_description)
        print(f"Generated plan: {skills}")
    except Exception as e:
        logging.error(f"Planning failed: {e}")
        skills = ["pick(object)", "place(bin)"]
    
    # Initialize policy and skill runner
    from lerobot.common.policies.factory import make_policy
    
    policy_kwargs = {
        "pretrained": "lerobot/koch_aloha",
        "config_overrides": {
            "input_shapes": {
                "observation.images.top": [3, 480, 640],
                "observation.state": [14],
            },
            "output_shapes": {
                "action": [7],
            }
        }
    }
    
    try:
        policy = make_policy("smolvla", policy_kwargs=policy_kwargs)
    except Exception as e:
        logging.error(f"Failed to load SmolVLA policy: {e}")
        # Fallback to mock policy for testing
        class MockPolicy:
            def select_action(self, obs):
                return torch.zeros(7)
        policy = MockPolicy()
    
    runner = SkillRunner(policy)
    
    # Start recording episode
    # Important: use task description at episode level, not per frame
    dataset.add_episode(task=task_description)
    
    # Execute skills and record
    total_steps = 0
    for skill_idx, skill in enumerate(skills):
        print(f"\nExecuting skill {skill_idx + 1}/{len(skills)}: {skill}")
        
        # Convert skill to natural language
        language_instruction = runner.skill_to_language_prompt(skill)
        
        # Execute skill with timeout
        skill_start_time = time.time()
        timeout = 5.0  # seconds per skill
        
        while time.time() - skill_start_time < timeout:
            # Get current observation
            obs_dict = robot.get_observation()
            
            # Prepare observation for policy
            policy_obs = {
                "observation.images.top": obs_dict["observation.images.top"],
                "observation.state": obs_dict["observation.state"],
                "task": language_instruction if use_language_per_frame else task_description
            }
            
            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(policy_obs)
            
            # Ensure action is numpy array
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            # Send action to robot
            robot.send_action(action)
            
            # Record frame in LeRobot format
            # Do NOT include task in frame data - it's stored at episode level
            frame_data = {
                "observation.images.top": obs_dict["observation.images.top"],
                "observation.state": obs_dict["observation.state"],
                "action": action,
            }
            
            # Add frame to dataset
            dataset.add_frame(frame_data)
            
            # Optional: store skill metadata separately
            # This goes in a separate metadata file, not in the dataset
            skill_metadata = {
                "episode_idx": episode_idx,
                "frame_idx": total_steps,
                "skill_idx": skill_idx,
                "skill": skill,
                "language_instruction": language_instruction,
                "time_in_skill": time.time() - skill_start_time
            }
            
            total_steps += 1
            
            # Simple completion check
            if total_steps % 50 == 0:  # Every ~5 seconds at 10Hz
                break
        
        # Brief pause between skills
        if skill_idx < len(skills) - 1:
            time.sleep(0.5)
    
    # End episode
    dataset.save_episode()
    
    # Save planning metadata separately (not in dataset)
    metadata_dir = Path(dataset.root) / "planning_metadata"
    metadata_dir.mkdir(exist_ok=True)
    
    planning_metadata = {
        "episode_idx": episode_idx,
        "task_description": task_description,
        "planner_name": planner_name,
        "skill_sequence": skills,
        "total_steps": total_steps,
        "timestamp": time.time()
    }
    
    metadata_path = metadata_dir / f"episode_{episode_idx:06d}.json"
    with open(metadata_path, 'w') as f:
        json.dump(planning_metadata, f, indent=2)
    
    print(f"\nEpisode complete! Total steps: {total_steps}")
    
    return planning_metadata


def main():
    parser = argparse.ArgumentParser(description="Collect RecycloBot dataset in LeRobot format")
    
    # Robot configuration
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/so101.yaml",
        help="Path to robot config file"
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Robot config overrides (e.g., cameras.top.index=0)"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., 'user/recyclobot-demos')"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to collect"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Control frequency in Hz"
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Local directory for dataset (default: auto)"
    )
    
    # Planning configuration
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Use autonomous collection with planner"
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="gemini",
        choices=["gemini", "qwen", "openai"],
        help="Planner to use for autonomous collection"
    )
    parser.add_argument(
        "--tasks-file",
        type=str,
        help="JSON file with task descriptions"
    )
    
    # Collection options
    parser.add_argument(
        "--warmup-time",
        type=float,
        default=2.0,
        help="Warmup time before recording (seconds)"
    )
    parser.add_argument(
        "--reset-time",
        type=float,
        default=5.0,
        help="Reset time between episodes (seconds)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load task descriptions
    if args.tasks_file:
        with open(args.tasks_file, 'r') as f:
            tasks = json.load(f)
    else:
        # Default recycling tasks
        tasks = [
            "Sort all the trash on the table into appropriate bins",
            "Pick up all plastic bottles and put them in the recycling bin",
            "Put the aluminum cans in the recycling bin",
            "Place all organic waste in the compost bin",
            "Clean up the workspace by sorting items into their correct bins",
        ]
    
    # Initialize robot using LeRobot's proper initialization
    from lerobot.common.robot_devices.robots.factory import make_robot
    from lerobot.common.robot_devices.robots.utils import get_robot_config
    
    # Load robot config
    robot_config = get_robot_config(args.robot_path, args.robot_overrides)
    
    # Create robot
    robot = make_robot(robot_config)
    robot.connect()
    
    print(f"Connected to robot: {robot_config['robot_type']}")
    print(f"Cameras: {list(robot_config.get('cameras', {}).keys())}")
    
    # Create dataset with proper LeRobot format
    root = args.root or f"data/{args.repo_id.split('/')[-1]}"
    dataset = create_lerobot_dataset(args.repo_id, robot_config, args.fps)
    
    # Select planner
    if args.autonomous:
        if args.planner == "gemini":
            planner_fn = gemini_plan
        elif args.planner == "qwen":
            planner_fn = qwen_plan
        else:
            from recyclobot.planning.openai_planner import plan as openai_plan
            planner_fn = openai_plan
        planner_name = args.planner
    else:
        print("Teleoperation mode - use LeRobot's record command instead")
        print("Example:")
        print(f"  python -m lerobot.record \\")
        print(f"    --robot-path {args.robot_path} \\")
        print(f"    --repo-id {args.repo_id} \\")
        print(f"    --num-episodes {args.num_episodes}")
        return
    
    # Collect episodes
    print(f"\nCollecting {args.num_episodes} episodes...")
    print(f"Robot: {robot_config['robot_type']}")
    print(f"Mode: Autonomous with {planner_name} planner")
    print(f"FPS: {args.fps}")
    
    try:
        for episode_idx in range(args.num_episodes):
            # Select task
            task = tasks[episode_idx % len(tasks)]
            
            # Warmup
            if episode_idx > 0:
                print(f"\nReset time ({args.reset_time}s)...")
                time.sleep(args.reset_time)
            
            # Collect episode
            collect_episode_with_planning(
                robot=robot,
                dataset=dataset,
                planner_fn=planner_fn,
                planner_name=planner_name,
                task_description=task,
                episode_idx=episode_idx
            )
            
            # Save checkpoint every 10 episodes
            if (episode_idx + 1) % 10 == 0:
                dataset.consolidate()
                print(f"Checkpoint saved at episode {episode_idx + 1}")
    
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    except Exception as e:
        logging.error(f"Collection failed: {e}")
        raise
    finally:
        # Finalize dataset
        dataset.consolidate()
        
        print(f"\n{'='*60}")
        print("Dataset collection complete!")
        print(f"{'='*60}")
        print(f"Total episodes: {dataset.num_episodes}")
        print(f"Dataset saved to: {dataset.root}")
        
        # Push to HuggingFace
        if input("\nPush to HuggingFace Hub? (y/n): ").lower() == 'y':
            print("Pushing to HuggingFace Hub...")
            dataset.push_to_hub(args.repo_id)
            print(f"Dataset available at: https://huggingface.co/datasets/{args.repo_id}")
        
        # Disconnect robot
        robot.disconnect()


if __name__ == "__main__":
    main()