"""
RecycloBot Skill Runner
Maps high-level recycling skills to SmolVLA natural language instructions.

IMPORTANT: SmolVLA Architecture (from HF blog)
- SmolVLA is a Vision-Language-Action model
- Takes RGB images + natural language instructions
- The language DOES specify what to manipulate (e.g., "pick up the red block")
- Uses SmolVLM-500M backbone for vision-language understanding
- Action expert outputs continuous robot actions
- No goal IDs - it's all natural language!
"""

import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# SmolVLA uses natural language, not goal IDs!
# These are skill templates for generating instructions

# Common recycling targets and their descriptions
RECYCLING_VOCAB = {
    # Objects to pick
    "plastic_bottle": "plastic bottle",
    "aluminum_can": "aluminum can", 
    "glass_bottle": "glass bottle",
    "paper": "paper waste",
    "cardboard": "cardboard",
    "food_waste": "food waste",
    "banana_peel": "banana peel",
    "apple_core": "apple core",
    
    # Bins to place in
    "recycling_bin": "recycling bin (blue)",
    "compost_bin": "compost bin (green)", 
    "trash_bin": "trash bin (black)",
    "paper_bin": "paper recycling bin",
}


class SkillRunner:
    """Execute high-level recycling skills using SmolVLA policy."""
    
    def __init__(self, policy, fps: int = 10, skill_timeout: float = 5.0):
        """
        Initialize skill runner.
        
        Args:
            policy: SmolVLA policy instance
            fps: Control frequency in Hz
            skill_timeout: Maximum time per skill in seconds
        """
        self.policy = policy
        self.dt = 1.0 / fps
        self.skill_timeout = skill_timeout
        self.device = policy.device if hasattr(policy, 'device') else 'cpu'
        
    def parse_skill(self, skill_str: str) -> Tuple[str, Optional[str]]:
        """
        Parse skill string into action and parameter.
        
        Args:
            skill_str: String like "pick(plastic_bottle)" or "sort()"
            
        Returns:
            Tuple of (action, parameter) where parameter may be None
        """
        match = re.match(r'(\w+)\(([^)]*)\)', skill_str)
        if match:
            action = match.group(1)
            param = match.group(2).strip() if match.group(2) else None
            return action, param
        return skill_str, None
    
    def skill_to_language_prompt(self, skill_str: str) -> str:
        """
        Convert skill string to natural language instruction for SmolVLA.
        
        SmolVLA uses natural language to understand WHAT to manipulate!
        The instruction specifies both the action AND the target object.
        
        Args:
            skill_str: Skill string like "pick(plastic_bottle)"
            
        Returns:
            Natural language instruction for SmolVLA
        """
        action, param = self.parse_skill(skill_str)
        
        if action == "pick" and param:
            # Convert parameter to natural description
            obj_desc = RECYCLING_VOCAB.get(param, param.replace("_", " "))
            return f"pick up the {obj_desc}"
        
        elif action == "place" and param:
            # Convert bin parameter to natural description
            bin_desc = RECYCLING_VOCAB.get(param, param.replace("_", " "))
            return f"place the object in the {bin_desc}"
            
        elif action == "inspect":
            if param:
                obj_desc = RECYCLING_VOCAB.get(param, param.replace("_", " "))
                return f"pick up the {obj_desc} to examine it"
            return "pick up the object to examine it"
            
        elif action == "sort":
            return "pick up items and place them in appropriate bins"
            
        elif action == "highfive":
            return "give a high five"
            
        else:
            # Fallback: try to make it natural
            return skill_str.replace("_", " ").replace("(", " ").replace(")", "")
    
    def execute_skill(self, skill_str: str, env, logger=None) -> bool:
        """
        Execute a single recycling skill.
        
        Args:
            skill_str: Skill string to execute
            env: Environment or robot interface
            logger: Optional dataset logger
            
        Returns:
            Success status
        """
        action_type, param = self.parse_skill(skill_str)
        
        # Convert skill to natural language instruction for SmolVLA
        language_instruction = self.skill_to_language_prompt(skill_str)
        
        print(f"Executing: {skill_str} -> Instruction: '{language_instruction}'")
        
        # Execute skill with timeout
        start_time = time.time()
        step_count = 0
        
        while time.time() - start_time < self.skill_timeout:
            # Get observation
            obs = env.get_observation() if hasattr(env, 'get_observation') else env.observation
            
            # SmolVLA expects specific observation format
            # Convert observation to LeRobot format
            if isinstance(obs, dict):
                # Get image - check multiple possible keys (prioritize LeRobot format)
                image = obs.get("observation.images.top", 
                               obs.get("observation.images.top",
                               obs.get("image", obs.get("observation.image", None))))
                if image is None:
                    # Look for any image key
                    for key in obs:
                        if "image" in key:
                            image = obs[key]
                            break
                    if image is None:
                        raise ValueError(f"No image found in observation. Keys: {obs.keys()}")
                
                # Get state - check multiple possible keys
                state = obs.get("observation.state", obs.get("state", np.zeros(14)))
                
                # Ensure proper types and shapes
                if isinstance(image, np.ndarray):
                    # Convert to torch tensor and ensure shape is (C, H, W)
                    if image.ndim == 3 and image.shape[-1] == 3:  # (H, W, C)
                        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                    elif image.ndim == 3 and image.shape[0] == 3:  # (C, H, W)
                        image = torch.from_numpy(image).float() / 255.0
                    else:
                        raise ValueError(f"Unexpected image shape: {image.shape}")
                
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state).float()
                
                obs_formatted = {
                    "observation.images.top": image,
                    "observation.state": state,
                    "task": language_instruction  # SmolVLA expects 'task', not 'instruction'
                }
            else:
                # If obs is just an image, create proper format
                if isinstance(obs, np.ndarray):
                    if obs.ndim == 3 and obs.shape[-1] == 3:  # (H, W, C)
                        obs = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
                    elif obs.ndim == 3 and obs.shape[0] == 3:  # (C, H, W)
                        obs = torch.from_numpy(obs).float() / 255.0
                
                obs_formatted = {
                    "observation.images.top": obs,
                    "observation.state": torch.zeros(14),  # Default state for SO-101 (6 joints + gripper) x2 (pos + vel)
                    "task": language_instruction
                }
            
            # Get action from policy
            with torch.no_grad():
                # SmolVLA forward pass with properly formatted observation
                if hasattr(self.policy, 'select_action'):
                    # Use select_action for inference (manages action queue)
                    action = self.policy.select_action(obs_formatted)
                elif hasattr(self.policy, 'predict_action'):
                    # Some implementations use predict_action
                    action = self.policy.predict_action(obs_formatted)
                else:
                    # Standard forward pass
                    action = self.policy(obs_formatted)
            
            # Execute action
            if hasattr(action, 'cpu'):
                action = action.cpu().numpy()
            
            if hasattr(env, 'send_action'):
                env.send_action(action)
            elif hasattr(env, 'step'):
                env.step(action)
            else:
                raise ValueError("Environment must have send_action or step method")
            
            # Log if logger provided
            if logger:
                # Convert back to original format for logging
                log_obs = {
                    "image": obs.get("image", obs.get("observation.images.top")),
                    "state": obs.get("state", obs.get("observation.state"))
                }
                logger.record(
                    obs=log_obs,
                    action=action,
                    done=False,
                    extra={
                        "current_skill": skill_str,
                        "language_instruction": language_instruction,
                        "step_in_skill": step_count
                    }
                )
            
            # Sleep to maintain control frequency
            time.sleep(self.dt)
            step_count += 1
            
            # Simple completion heuristics based on skill type and time
            if action_type == "pick" and step_count > 30:  # ~3 seconds at 10Hz
                break
            elif action_type == "place" and step_count > 40:  # ~4 seconds
                break
        
        # Log skill completion
        if logger:
            # Convert back to original format for logging
            log_obs = {
                "image": obs.get("image", obs.get("observation.images.top")),
                "state": obs.get("state", obs.get("observation.state"))
            }
            logger.record(
                obs=log_obs,
                action=action,
                done=True,
                extra={
                    "skill_completed": skill_str,
                    "duration": time.time() - start_time,
                    "total_steps": step_count
                }
            )
        
        return True
    
    def run(self, skills: List[str], env, logger=None):
        """
        Execute a sequence of recycling skills.
        
        Args:
            skills: List of skill strings
            env: Environment or robot interface  
            logger: Optional dataset logger
        """
        print(f"Executing skill sequence: {skills}")
        
        for i, skill in enumerate(skills):
            print(f"\n[{i+1}/{len(skills)}] {skill}")
            success = self.execute_skill(skill, env, logger)
            
            if not success:
                print(f"Warning: Skill {skill} may have failed")
            
            # Brief pause between skills
            if i < len(skills) - 1:
                time.sleep(0.5)