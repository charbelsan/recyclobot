#!/usr/bin/env python
"""
Collect RecycloBot dataset in LeRobot format with vision-language planning

This script properly integrates with LeRobot's data collection pipeline
while adding RecycloBot's planning capabilities.

Usage:
    # Teleoperated collection with planning annotation
    python scripts/collect_recyclobot_dataset.py \
        --robot-type so101 \
        --repo-id your-username/recyclobot-demos \
        --num-episodes 50
        
    # Autonomous collection with planner
    python scripts/collect_recyclobot_dataset.py \
        --robot-type so101 \
        --repo-id your-username/recyclobot-demos \
        --autonomous \
        --planner gemini
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from recyclobot.control.skill_runner import SkillRunner
from recyclobot.planning.gemini_planner import plan as gemini_plan
from recyclobot.planning.qwen_planner import plan as qwen_plan


def collect_episode_with_planning(robot, dataset, planner_fn, planner_name, task_description):
    """
    Collect a single episode with planning annotations.
    
    This integrates RecycloBot's planning with LeRobot's data collection.
    """
    print(f"\n{'='*60}")
    print(f"Episode {dataset.num_episodes + 1}")
    print(f"Task: {task_description}")
    print(f"{'='*60}")
    
    # Reset robot
    observation, info = robot.reset()
    
    # Get initial image for planning
    if "observation.images.top" in observation:
        image_array = observation["observation.images.top"]
        # Convert tensor to numpy if needed
        if hasattr(image_array, 'numpy'):
            image_array = image_array.numpy()
        # Convert CHW to HWC for PIL
        if image_array.shape[0] == 3:
            image_array = np.transpose(image_array, (1, 2, 0))
        # Denormalize if needed
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        image = Image.fromarray(image_array)
    else:
        print("Warning: No camera image found, using placeholder")
        image = Image.new('RGB', (640, 480))
    
    # Generate plan
    print(f"Planning with {planner_name}...")
    try:
        skills = planner_fn(image, task_description)
        print(f"Generated plan: {skills}")
    except Exception as e:
        print(f"Planning failed: {e}")
        skills = ["pick(object)", "place(bin)"]
    
    # Create skill runner
    from lerobot.common.policies.factory import make_policy
    policy = make_policy("smolvla", pretrained="lerobot/smolvla_base")
    runner = SkillRunner(policy)
    
    # Store planning metadata
    planning_metadata = {
        "planner_name": planner_name,
        "task_description": task_description,
        "skill_sequence": skills,
        "planning_success": True
    }
    
    # Start episode
    dataset.start_episode(task=task_description)
    
    # Execute skills and record
    for skill_idx, skill in enumerate(skills):
        print(f"\nExecuting skill {skill_idx + 1}/{len(skills)}: {skill}")
        
        # Convert skill to natural language
        language_instruction = runner.skill_to_language_prompt(skill)
        
        # Execute skill with timeout
        skill_start_step = dataset.num_steps
        timeout = 5.0  # seconds per skill
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Get observation
            observation = robot.get_observation()
            
            # Add language instruction to observation
            observation["task"] = language_instruction
            
            # Get action from policy
            action = policy.select_action(observation)
            
            # Send action to robot
            robot.send_action(action)
            
            # Record frame with metadata
            frame_metadata = {
                "skill_index": skill_idx,
                "current_skill": skill,
                "language_instruction": language_instruction,
                "time_in_skill": time.time() - start_time
            }
            
            # Add frame to dataset
            dataset.add_frame({
                **observation,
                "action": action,
                "metadata": json.dumps(frame_metadata)
            })
            
            # Check for skill completion (simplified)
            if dataset.num_steps - skill_start_step > 50:  # ~5 seconds at 10Hz
                break
    
    # End episode
    dataset.end_episode()
    
    # Save planning metadata
    metadata_path = Path(dataset.root) / "planning_metadata" / f"episode_{dataset.num_episodes:06d}.json"
    metadata_path.parent.mkdir(exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(planning_metadata, f, indent=2)
    
    print(f"\nEpisode complete! Total steps: {dataset.num_steps}")
    
    return planning_metadata


def main():
    parser = argparse.ArgumentParser(description="Collect RecycloBot dataset with planning")
    
    # Robot configuration
    parser.add_argument(
        "--robot-type",
        type=str,
        default="so101",
        choices=["so101", "aloha", "sim"],
        help="Robot type"
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        help="Path to robot configuration file"
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
        default=10,
        help="Control frequency in Hz"
    )
    
    # Planning configuration
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Use autonomous collection with planner (vs teleoperation)"
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
    
    args = parser.parse_args()
    
    # Load task descriptions
    if args.tasks_file:
        with open(args.tasks_file, 'r') as f:
            tasks = json.load(f)
    else:
        # Default recycling tasks
        tasks = [
            "Sort all the trash on the table into appropriate bins",
            "Pick up all plastic bottles and put them in the recycling bin",
            "Separate organic waste from recyclables",
            "Clean up the workspace by sorting items",
            "Put all aluminum cans in the recycling bin"
        ]
    
    # Initialize robot
    if args.robot_type == "sim":
        print("Note: Simulation mode - using mock robot")
        # Create mock robot for testing
        class MockRobot:
            def reset(self):
                return {"observation.images.top": np.zeros((3, 480, 640))}, {}
            def get_observation(self):
                return {"observation.images.top": np.zeros((3, 480, 640))}
            def send_action(self, action):
                pass
        robot = MockRobot()
    else:
        from lerobot.common.robot_devices.robots.factory import make_robot
        robot = make_robot(args.robot_type)
        robot.connect()
    
    # Initialize dataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        fps=args.fps,
        robot=robot,
        root=f"data/{args.repo_id.split('/')[-1]}"
    )
    
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
        print("Teleoperation mode - planning will be done post-hoc")
        planner_fn = None
        planner_name = "human"
    
    # Collect episodes
    print(f"\nCollecting {args.num_episodes} episodes...")
    print(f"Robot: {args.robot_type}")
    print(f"Mode: {'Autonomous' if args.autonomous else 'Teleoperated'}")
    print(f"Planner: {planner_name}")
    
    for episode_idx in range(args.num_episodes):
        # Select task
        task = tasks[episode_idx % len(tasks)]
        
        if args.autonomous and planner_fn:
            # Autonomous collection with planning
            collect_episode_with_planning(
                robot, dataset, planner_fn, planner_name, task
            )
        else:
            # Teleoperated collection
            print(f"\n{'='*60}")
            print(f"Episode {episode_idx + 1}/{args.num_episodes}")
            print(f"Task: {task}")
            print("Teleoperate the robot to complete the task")
            print("Press 'q' to end episode")
            print(f"{'='*60}")
            
            # Use LeRobot's teleoperation
            from lerobot.scripts.control_robot import record_episode
            record_episode(
                robot=robot,
                dataset=dataset,
                task=task,
                fps=args.fps
            )
    
    # Finalize dataset
    dataset.consolidate()
    
    print(f"\n{'='*60}")
    print("Dataset collection complete!")
    print(f"{'='*60}")
    print(f"Total episodes: {dataset.num_episodes}")
    print(f"Total steps: {dataset.num_steps}")
    print(f"Dataset saved to: {dataset.root}")
    
    # Push to HuggingFace
    print("\nPushing to HuggingFace Hub...")
    dataset.push_to_hub(args.repo_id)
    print(f"Dataset available at: https://huggingface.co/datasets/{args.repo_id}")
    
    # Disconnect robot
    if hasattr(robot, 'disconnect'):
        robot.disconnect()


if __name__ == "__main__":
    main()