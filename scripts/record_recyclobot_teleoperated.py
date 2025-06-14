#!/usr/bin/env python
"""
Record RecycloBot demonstrations using teleoperation with planning annotations.

This script wraps LeRobot's record functionality to add planning metadata
for teleoperated demonstrations.

Usage:
    # Record with keyboard teleoperation
    python scripts/record_recyclobot_teleoperated.py \
        --robot-path lerobot/configs/robot/so101.yaml \
        --control-mode keyboard \
        --repo-id user/recyclobot-teleop \
        --task "Sort the recycling into appropriate bins" \
        --num-episodes 10
        
    # Record with gamepad
    python scripts/record_recyclobot_teleoperated.py \
        --robot-path lerobot/configs/robot/so101.yaml \
        --control-mode gamepad \
        --repo-id user/recyclobot-teleop \
        --tasks-file recycling_tasks.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from recyclobot.planning.gemini_planner import plan as gemini_plan
from recyclobot.planning.qwen_planner import plan as qwen_plan


def analyze_teleoperated_episode(
    episode_dir: Path,
    task: str,
    planner_name: str = "gemini"
) -> dict:
    """
    Analyze a teleoperated episode to extract planning information.
    
    Args:
        episode_dir: Directory containing episode data
        task: Task description
        planner_name: Which planner to use for analysis
        
    Returns:
        Dictionary with planning metadata
    """
    # Load first frame to get scene image
    from datasets import load_from_disk
    
    episode_data = load_from_disk(str(episode_dir))
    first_frame = episode_data[0]
    
    # Extract image
    if "observation.images.top" in first_frame:
        image_data = first_frame["observation.images.top"]
        if isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        else:
            # Handle tensor format
            image_np = image_data.numpy()
            if image_np.shape[0] == 3:  # CHW -> HWC
                image_np = np.transpose(image_np, (1, 2, 0))
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            image = Image.fromarray(image_np)
    else:
        print("Warning: No image found for planning analysis")
        return {"error": "No image found"}
    
    # Run planner on initial scene
    if planner_name == "gemini":
        planner_fn = gemini_plan
    elif planner_name == "qwen":
        planner_fn = qwen_plan
    else:
        from recyclobot.planning.openai_planner import plan as openai_plan
        planner_fn = openai_plan
    
    try:
        planned_skills = planner_fn(image, task)
        print(f"Planner analysis: {planned_skills}")
    except Exception as e:
        print(f"Planning analysis failed: {e}")
        planned_skills = ["unknown"]
    
    # Analyze demonstrated trajectory
    # This is simplified - in practice you'd want more sophisticated analysis
    episode_length = len(episode_data)
    
    metadata = {
        "task": task,
        "planner_name": planner_name,
        "planned_skills": planned_skills,
        "episode_length": episode_length,
        "demonstration_type": "teleoperated",
        "timestamp": time.time()
    }
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Record teleoperated RecycloBot demonstrations"
    )
    
    # Robot configuration
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/so101.yaml",
        help="Path to robot config"
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Robot config overrides"
    )
    
    # Control configuration  
    parser.add_argument(
        "--control-mode",
        type=str,
        default="keyboard",
        choices=["keyboard", "gamepad", "spacemouse"],
        help="Teleoperation control mode"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repo ID"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to record"
    )
    
    # Task configuration
    parser.add_argument(
        "--task",
        type=str,
        help="Single task description"
    )
    parser.add_argument(
        "--tasks-file",
        type=str,
        help="JSON file with task descriptions"
    )
    
    # Planning analysis
    parser.add_argument(
        "--analyze-after",
        action="store_true",
        help="Run planning analysis after recording"
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="gemini",
        choices=["gemini", "qwen", "openai"],
        help="Planner for post-hoc analysis"
    )
    
    # LeRobot passthrough options
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording frequency"
    )
    parser.add_argument(
        "--warmup-time",
        type=float,
        default=2.0,
        help="Warmup time before recording"
    )
    parser.add_argument(
        "--reset-time", 
        type=float,
        default=5.0,
        help="Reset time between episodes"
    )
    parser.add_argument(
        "--display-cameras",
        action="store_true",
        help="Display camera feeds during recording"
    )
    
    args = parser.parse_args()
    
    # Load tasks
    if args.tasks_file:
        with open(args.tasks_file, 'r') as f:
            tasks = json.load(f)
    elif args.task:
        tasks = [args.task] * args.num_episodes
    else:
        print("Error: Either --task or --tasks-file must be provided")
        sys.exit(1)
    
    # Ensure we have enough tasks
    while len(tasks) < args.num_episodes:
        tasks.extend(tasks)
    tasks = tasks[:args.num_episodes]
    
    # Build LeRobot record command
    lerobot_cmd = [
        "python", "-m", "lerobot.record",
        "--robot-path", args.robot_path,
        "--repo-id", args.repo_id,
        "--num-episodes", str(args.num_episodes),
        "--fps", str(args.fps),
        "--warmup-time", str(args.warmup_time),
        "--reset-time", str(args.reset_time),
        "--push-to-hub", "0",  # We'll push after adding metadata
    ]
    
    # Add robot overrides
    if args.robot_overrides:
        for override in args.robot_overrides:
            lerobot_cmd.extend(["--robot-overrides", override])
    
    # Add control mode
    if args.control_mode == "keyboard":
        lerobot_cmd.extend(["--teleop-mode", "keyboard"])
    elif args.control_mode == "gamepad":
        lerobot_cmd.extend(["--teleop-mode", "gamepad"])
    
    # Add display option
    if args.display_cameras:
        lerobot_cmd.append("--display-cameras")
    
    # Add tasks
    for i, task in enumerate(tasks):
        lerobot_cmd.extend([f"--task-{i}", task])
    
    print("Starting LeRobot teleoperated recording...")
    print(f"Command: {' '.join(lerobot_cmd)}")
    
    # Run LeRobot record
    try:
        result = subprocess.run(lerobot_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Recording failed: {e}")
        sys.exit(1)
    
    # Post-processing: Add planning metadata
    if args.analyze_after:
        print("\nAnalyzing recorded episodes with planner...")
        
        dataset_dir = Path(f"data/{args.repo_id.split('/')[-1]}")
        metadata_dir = dataset_dir / "planning_metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # Analyze each episode
        for episode_idx in range(args.num_episodes):
            print(f"\nAnalyzing episode {episode_idx + 1}/{args.num_episodes}")
            
            episode_dir = dataset_dir / "data" / f"chunk-000" / f"episode_{episode_idx:06d}"
            task = tasks[episode_idx]
            
            if episode_dir.exists():
                metadata = analyze_teleoperated_episode(
                    episode_dir,
                    task,
                    args.planner
                )
                
                # Save metadata
                metadata_path = metadata_dir / f"episode_{episode_idx:06d}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            else:
                print(f"Warning: Episode directory not found: {episode_dir}")
    
    print(f"\nRecording complete!")
    print(f"Dataset saved to: data/{args.repo_id.split('/')[-1]}")
    
    # Offer to push to hub
    if input("\nPush to HuggingFace Hub? (y/n): ").lower() == 'y':
        push_cmd = [
            "huggingface-cli", "upload",
            args.repo_id,
            f"data/{args.repo_id.split('/')[-1]}",
            "--repo-type", "dataset"
        ]
        subprocess.run(push_cmd, check=True)
        print(f"Dataset available at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()