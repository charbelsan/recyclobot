#!/usr/bin/env python
"""
RecycloBot demo using LeRobot's gym environments.

This uses actual simulation environments that come with LeRobot,
particularly the Aloha environment which is suitable for manipulation tasks.
"""

import argparse
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# LeRobot imports
from lerobot.common.envs.factory import make_env
from lerobot.common.policies.factory import make_policy

# RecycloBot imports  
from recyclobot.planning.direct_smolvla_planner import plan as direct_plan
from recyclobot.control.skill_runner import SkillRunner
from recyclobot.logging.dataset_logger import RecycloBotLogger


def create_gym_environment(env_name="aloha", task=None, render=True):
    """
    Create a LeRobot gym environment.
    
    Args:
        env_name: Environment type ("aloha", "pusht", "xarm")
        task: Specific task (e.g., "AlohaInsertion-v0")
        render: Whether to render the environment
        
    Returns:
        Environment instance
    """
    from lerobot.common.envs.configs import (
        AlohaConfig, 
        PushtConfig,
        XarmConfig
    )
    
    # Select appropriate config
    if env_name == "aloha":
        config = AlohaConfig(
            task=task or "AlohaInsertion-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array" if render else None,
            visualization=render
        )
    elif env_name == "pusht":
        config = PushtConfig(
            task=task or "PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array" if render else None,
            visualization=render
        )
    elif env_name == "xarm":
        config = XarmConfig(
            task=task or "XarmLift-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array" if render else None,
            visualization=render
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    # Create environment
    env = make_env(config)
    print(f"Created {env_name} environment: {config.task}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    return env, config


def run_gym_demo(args):
    """Run RecycloBot demo with gym environment."""
    
    print("RecycloBot Gym Environment Demo")
    print("=" * 60)
    
    # Create environment
    try:
        env, env_config = create_gym_environment(
            env_name=args.env,
            task=args.task,
            render=args.render
        )
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\nMake sure you've installed the environment:")
        print(f"  pip install 'lerobot[{args.env}]'")
        return
    
    # Create policy
    print("\nLoading SmolVLA policy...")
    try:
        # Determine action dimension based on environment
        if args.env == "aloha":
            action_dim = 14  # Dual 7-DOF arms
            state_dim = 14
        elif args.env == "pusht":
            action_dim = 2   # 2D movement
            state_dim = 2
        elif args.env == "xarm":
            action_dim = 4   # 3D + gripper
            state_dim = 4
        else:
            action_dim = 7   # Default
            state_dim = 14
        
        policy = make_policy(
            "smolvla",
            policy_kwargs={
                "pretrained": "lerobot/koch_aloha",
                "config_overrides": {
                    "input_shapes": {
                        "observation.images.top": [3, 480, 640],
                        "observation.state": [state_dim],
                    },
                    "output_shapes": {
                        "action": [action_dim],
                    }
                }
            }
        )
        print("✅ SmolVLA policy loaded")
    except Exception as e:
        print(f"Error loading policy: {e}")
        policy = None
    
    # Create planner
    planner_name = args.planner
    if planner_name == "direct":
        plan_fn = direct_plan
    else:
        # Import other planners as needed
        plan_fn = direct_plan
    
    # Create logger
    logger = RecycloBotLogger(
        dataset_name=f"recyclobot_{args.env}_demo",
        robot_name=args.env,
        fps=env_config.fps,
        output_dir=args.output
    )
    
    # Run episodes
    for episode in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{args.episodes}")
        print(f"Task: {args.prompt}")
        print(f"{'='*60}")
        
        # Reset environment
        obs, info = env.reset()
        
        # Get initial image for planning
        if "pixels" in obs:
            # Extract image from observation
            image_array = obs["pixels"]["top"]
            if isinstance(image_array, torch.Tensor):
                image_array = image_array.cpu().numpy()
            
            # Convert to PIL Image
            if image_array.shape[0] == 3:  # CHW format
                image_array = np.transpose(image_array, (1, 2, 0))
            image = Image.fromarray((image_array * 255).astype(np.uint8))
        else:
            # Create placeholder image
            image = Image.new('RGB', (640, 480))
        
        # Generate plan
        print(f"Planning with {planner_name}...")
        skills = plan_fn(image, args.prompt)
        print(f"Generated plan: {skills}")
        
        # Execute plan
        if policy and isinstance(skills[0], str) and skills[0].startswith("task:"):
            # Direct execution mode
            print("Executing direct task...")
            
            done = False
            total_reward = 0
            steps = 0
            max_steps = 300
            
            while not done and steps < max_steps:
                # Format observation for SmolVLA
                if "pixels" in obs:
                    image_tensor = obs["pixels"]["top"]
                    if isinstance(image_tensor, np.ndarray):
                        image_tensor = torch.from_numpy(image_tensor).float()
                else:
                    image_tensor = torch.randn(3, 480, 640)
                
                if "agent_pos" in obs:
                    state_tensor = torch.from_numpy(obs["agent_pos"]).float()
                else:
                    state_tensor = torch.zeros(state_dim)
                
                policy_obs = {
                    "observation.images.top": image_tensor,
                    "observation.state": state_tensor,
                    "task": args.prompt  # Use original prompt
                }
                
                # Get action from policy
                with torch.no_grad():
                    action = policy.select_action(policy_obs)
                
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
                # Log data
                logger.record(
                    obs={"image": image_tensor.numpy(), "state": state_tensor.numpy()},
                    action=action,
                    done=done,
                    extra={
                        "reward": reward,
                        "step": steps,
                        "task": args.prompt
                    }
                )
                
                # Render if enabled
                if args.render and hasattr(env, 'render'):
                    env.render()
                
                # Small delay for visualization
                if args.render:
                    time.sleep(0.05)
            
            print(f"Episode complete: {steps} steps, reward: {total_reward:.2f}")
            print(f"Success: {info.get('success', False)}")
            
        else:
            # Skill-based execution
            print("Executing skill sequence...")
            runner = SkillRunner(policy)
            
            # Simple execution loop
            for skill_idx, skill in enumerate(skills):
                print(f"\n[{skill_idx+1}/{len(skills)}] {skill}")
                
                # Convert to natural language
                instruction = runner.skill_to_language_prompt(skill)
                print(f"Instruction: '{instruction}'")
                
                # Execute for a fixed number of steps
                for step in range(50):  # 5 seconds at 10 Hz
                    if "pixels" in obs:
                        image_tensor = torch.from_numpy(obs["pixels"]["top"]).float()
                    else:
                        image_tensor = torch.randn(3, 480, 640)
                    
                    if "agent_pos" in obs:
                        state_tensor = torch.from_numpy(obs["agent_pos"]).float()
                    else:
                        state_tensor = torch.zeros(state_dim)
                    
                    policy_obs = {
                        "observation.images.top": image_tensor,
                        "observation.state": state_tensor,
                        "task": instruction
                    }
                    
                    with torch.no_grad():
                        action = policy.select_action(policy_obs)
                    
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if args.render and hasattr(env, 'render'):
                        env.render()
                        time.sleep(0.05)
                    
                    if terminated or truncated:
                        break
    
    # Cleanup
    env.close()
    
    # Save dataset
    logger.create_dataset_card()
    stats = logger.get_statistics()
    
    print(f"\n{'='*60}")
    print("Demo Complete!")
    print(f"{'='*60}")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Dataset saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="RecycloBot demo with LeRobot gym environments"
    )
    
    parser.add_argument(
        "--env",
        default="aloha",
        choices=["aloha", "pusht", "xarm"],
        help="Gym environment to use"
    )
    parser.add_argument(
        "--task",
        help="Specific task (e.g., AlohaInsertion-v0)"
    )
    parser.add_argument(
        "--prompt",
        default="Insert the peg into the hole",
        help="Task instruction"
    )
    parser.add_argument(
        "--planner",
        default="direct",
        choices=["direct", "gemini", "qwen"],
        help="Planner type"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment"
    )
    parser.add_argument(
        "--output",
        default="gym_demo_output",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Run demo
    run_gym_demo(args)


if __name__ == "__main__":
    main()