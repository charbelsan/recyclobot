#!/usr/bin/env python
"""
RecycloBot Demo - Waste Sorting with Vision-Language Planning

Usage:
    # Simulation (default)
    python run_recyclobot_demo.py --prompt "Sort the trash on the table"
    
    # Real robot (SO-ARM100)
    python run_recyclobot_demo.py --robot so101 --prompt "Put bottles in recycling"
    
    # Force specific planner
    python run_recyclobot_demo.py --planner gemini --prompt "Clean up the workspace"
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from recyclobot.control.skill_runner import SkillRunner
from recyclobot.logging.dataset_logger import RecycloBotLogger


def select_planner(force_planner=None, config_path=None):
    """
    Select appropriate planner based on availability and user preference.
    
    Args:
        force_planner: Force specific planner ("gemini", "qwen", "openai", "smolvlm", etc.)
        config_path: Path to config file for OpenAI-style planners
        
    Returns:
        Tuple of (planner_name, planner_function)
    """
    # Handle direct SmolVLA execution (no separate planner)
    if force_planner == "direct":
        from recyclobot.planning.direct_smolvla_planner import plan
        print("Using direct SmolVLA execution (no separate planner)")
        return "direct", plan
    
    # Handle SmolVLM planner (same model as in SmolVLA but used separately)
    if force_planner == "smolvlm":
        try:
            from recyclobot.planning.smolvlm_planner import plan
            print("Using SmolVLM planner (same model as in SmolVLA)")
            return "smolvlm", plan
        except ImportError as e:
            print(f"Error: Cannot use SmolVLM planner: {e}")
            print("Install with: pip install transformers>=4.44.0")
            sys.exit(1)
    
    # Handle OpenAI-style planners
    if force_planner in ["openai", "anthropic", "local", "ollama", "vllm", "together"] or os.getenv("OPENAI_API_KEY"):
        try:
            from recyclobot.planning.openai_planner import OpenAIPlanner
            
            # Load config to get planner name
            config = {}
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        import yaml
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
            
            # Determine which OpenAI-style planner to use
            if force_planner:
                planner_name = force_planner
            elif config.get("default_planner"):
                planner_name = config["default_planner"]
            else:
                planner_name = "openai"
            
            # Create planner instance
            planner = OpenAIPlanner(config_path)
            print(f"Using {planner_name} planner (OpenAI-compatible API)")
            return planner_name, lambda img, prompt: planner.plan(img, prompt)
            
        except (ImportError, ValueError, RuntimeError) as e:
            if force_planner in ["openai", "anthropic", "local"]:
                print(f"Error: Cannot use {force_planner} planner: {e}")
                sys.exit(1)
            print(f"OpenAI-style planner unavailable ({e})")
    
    # Try Gemini
    if force_planner == "gemini" or os.getenv("GEMINI_API_KEY"):
        try:
            from recyclobot.planning.gemini_planner import plan
            print("Using Gemini planner")
            return "gemini", plan
        except (ImportError, RuntimeError) as e:
            if force_planner == "gemini":
                print(f"Error: Cannot use Gemini planner: {e}")
                sys.exit(1)
            print(f"Gemini unavailable ({e})")
    
    # Try Qwen
    if force_planner == "qwen":
        from recyclobot.planning.qwen_planner import plan
        print("Using Qwen-VL planner (local)")
        return "qwen", plan
    
    # Default fallback to Qwen
    from recyclobot.planning.qwen_planner import plan
    print("Using Qwen-VL planner (local) as fallback")
    return "qwen", plan


def create_environment(robot_type="sim"):
    """
    Create robot environment (simulation or real).
    
    Args:
        robot_type: "sim" for simulation or "so101" for real robot
        
    Returns:
        Environment instance
    """
    if robot_type == "sim":
        try:
            # Try to use LeRobot's simulation environment
            from lerobot.common.envs.factory import make_env
            
            env = make_env("aloha_sim_insertion_human")
            print("Using LeRobot simulation environment")
            return env
            
        except (ImportError, Exception) as e:
            print(f"Note: Using mock environment ({e})")
            
            # Fallback mock environment
            class MockEnv:
                def __init__(self):
                    self.step_count = 0
                    
                def reset(self, seed=None):
                    self.step_count = 0
                    # Return observation in LeRobot format
                    obs = {
                        "observation.images.main_camera": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                        "observation.state": np.random.randn(7).astype(np.float32),
                        "image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),  # Legacy support
                        "state": np.random.randn(7).astype(np.float32)  # Legacy support
                    }
                    return obs, {}
                
                def get_observation(self):
                    # Return observation in LeRobot format
                    obs = {
                        "observation.images.main_camera": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                        "observation.state": np.random.randn(7).astype(np.float32),
                        "image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),  # Legacy support
                        "state": np.random.randn(7).astype(np.float32)  # Legacy support
                    }
                    return obs
                
                def send_action(self, action):
                    self.step_count += 1
                    return action
                
                def close(self):
                    pass
                    
            return MockEnv()
    
    elif robot_type == "so101":
        try:
            from lerobot.common.robots.factory import make_robot
            
            robot = make_robot(
                "so101",
                robot_kwargs={
                    "use_cameras": True,
                    "use_leader": False
                }
            )
            robot.connect()
            print("Connected to SO-101 robot")
            return robot
            
        except ImportError as e:
            print(f"Error: LeRobot not properly installed: {e}")
            print("Install with: pip install 'lerobot[feetech]'")
            sys.exit(1)
        except Exception as e:
            print(f"Error connecting to robot: {e}")
            print("Make sure the robot is connected and powered on")
            sys.exit(1)
    
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")


def create_policy(robot_type="sim"):
    """
    Create SmolVLA policy for control.
    
    Args:
        robot_type: Robot type to determine device
        
    Returns:
        Policy instance
    """
    try:
        from lerobot.common.policies.factory import make_policy
        from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        
        # Load pre-trained SmolVLA policy
        device = "cuda" if robot_type != "sim" and torch.cuda.is_available() else "cpu"
        
        print("Loading SmolVLA from HuggingFace...")
        
        try:
            # Try to load pretrained model with make_policy
            policy = make_policy(
                "smolvla",
                pretrained="lerobot/koch_aloha",  # Use a working pretrained model
                config_overrides={
                    "input_shapes": {
                        "observation.images.top": [3, 480, 640],
                        "observation.state": [14],  # SO-101 has 14 state dims
                    },
                    "output_shapes": {
                        "action": [7],  # 6 joints + gripper
                    }
                }
            )
            print("Loaded pretrained SmolVLA model")
        except Exception as e:
            print(f"Could not load pretrained model: {e}")
            print("Creating SmolVLA policy from scratch...")
            
            # Create config with proper features for RecycloBot
            config = SmolVLAConfig(
                input_features={
                    "observation.images.top": {"shape": [3, 480, 640], "type": "image"},
                    "observation.state": {"shape": [14], "type": "state"}
                },
                output_features={
                    "action": {"shape": [7], "type": "action"}
                },
                # Use smaller chunk size for demo
                chunk_size=10,
                n_action_steps=10,
            )
            
            # Create policy without pretrained weights
            policy = SmolVLAPolicy(config)
        
        policy.to(device)
        policy.eval()  # Set to evaluation mode
        
        print(f"SmolVLA policy ready on {device}")
        print("Note: SmolVLA uses natural language instructions, not goal IDs!")
        return policy
        
    except ImportError as e:
        print(f"Error: LeRobot not installed properly: {e}")
        print("Please run: pip install 'lerobot[smolvla]'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading SmolVLA: {e}")
        print("Make sure you have the latest LeRobot version")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="RecycloBot - AI-powered waste sorting demonstration"
    )
    parser.add_argument(
        "--robot",
        default="sim",
        choices=["sim", "so101"],
        help="Robot type: simulation or real SO-101"
    )
    parser.add_argument(
        "--prompt",
        default="Sort all the trash into the appropriate bins",
        help="Task instruction for the planner"
    )
    parser.add_argument(
        "--planner",
        choices=["direct", "gemini", "qwen", "openai", "anthropic", "local", "ollama", "vllm", "together", "smolvlm"],
        help="Force specific planner (default: auto-select, 'direct' uses SmolVLA without separate planner)"
    )
    parser.add_argument(
        "--config",
        default="recyclobot/config.yaml",
        help="Path to configuration file (YAML or JSON) for OpenAI-style planners"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to record"
    )
    parser.add_argument(
        "--output",
        default="recyclobot_data",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Control frequency in Hz"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("RecycloBot Demo - Waste Sorting with Vision-Language Planning")
    print("=" * 60)
    print(f"Robot: {args.robot}")
    print(f"Task: {args.prompt}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Initialize components
    planner_name, planner_fn = select_planner(args.planner, args.config)
    env = create_environment(args.robot)
    policy = create_policy(args.robot)
    
    # Create skill runner and logger
    runner = SkillRunner(policy, fps=args.fps)
    logger = RecycloBotLogger(args.output, dataset_name=f"recyclobot_{args.robot}")
    
    # Run episodes
    for episode in range(args.episodes):
        print(f"\n--- Episode {episode + 1}/{args.episodes} ---")
        
        # Reset environment
        obs, _ = env.reset(seed=episode)
        
        # Get initial image for planning
        image = Image.fromarray(obs["image"])
        
        # Generate skill plan
        print(f"Planning with {planner_name}...")
        try:
            skills = planner_fn(image, args.prompt)
            print(f"Generated plan: {skills}")
        except Exception as e:
            print(f"Planning failed: {e}")
            skills = ["inspect(items)", "sort()"]  # Fallback plan
            print(f"Using fallback plan: {skills}")
        
        # Validate skills
        if not isinstance(skills, list) or not skills:
            print("Warning: Invalid plan, using default")
            skills = ["inspect(items)", "sort()"]
        
        # Start episode logging
        logger.start_episode(planner_name, skills)
        
        # Log initial observation
        logger.record(
            obs=obs,
            action=None,
            done=False,
            extra={
                "planner_name": planner_name,
                "planner_log": str(skills),
                "current_skill": "planning",
                "language_instruction": "initializing"
            }
        )
        
        # Execute skill sequence
        runner.run(skills, env, logger)
        
        # Final logging
        final_obs = env.get_observation()
        logger.record(
            obs=final_obs,
            action=np.zeros(7),
            done=True,
            extra={
                "planner_name": planner_name,
                "planner_log": str(skills),
                "episode_complete": True
            }
        )
        
        print(f"Episode {episode + 1} complete!")
    
    # Cleanup
    env.close()
    
    # Generate dataset card and statistics
    logger.create_dataset_card()
    stats = logger.get_statistics()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Average episode length: {stats['average_episode_length']:.1f} steps")
    print(f"Skills used: {stats['skill_usage']}")
    print(f"\nDataset saved to: {args.output}")
    print("\nTo upload to HuggingFace Hub:")
    print(f"  cd {args.output}")
    print("  huggingface-cli upload recyclobot/waste-sorting-demos .")
    

if __name__ == "__main__":
    main()