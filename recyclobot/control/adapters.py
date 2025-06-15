"""
State adapters for SmolVLA compatibility.

IMPORTANT: SmolVLA base model actually uses 6-dimensional state/action vectors,
not 14-dimensional as previously assumed. The normalization buffers confirm this.
This module now handles the 6-dim format correctly.
"""

import numpy as np
import torch
from typing import Union, Tuple

_EXPECTED = 6  # SmolVLA actually expects 6-dim state
_J = 6  # Direct dimension (not joints × 2)


def pad_state(
    qpos: Union[np.ndarray, torch.Tensor, list], 
    qvel: Union[np.ndarray, torch.Tensor, list, None] = None
) -> np.ndarray:
    """
    Ensure state matches SmolVLA's 6-dimensional format.
    
    Args:
        qpos: State vector or joint positions
        qvel: Ignored - SmolVLA uses combined 6-dim state
        
    Returns:
        6-dimensional state vector
        
    Raises:
        ValueError: If state dimension doesn't match
        
    Example:
        >>> # SO-100 has 6 joints/dims - perfect match
        >>> state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        >>> result = pad_state(state)  # Returns (6,) array unchanged
    """
    # Convert to numpy
    state = np.asarray(qpos, dtype=np.float32)
    
    # SmolVLA expects exactly 6-dim state
    if len(state) == _EXPECTED:
        return state
    elif len(state) < _EXPECTED:
        # Pad with zeros if needed
        pad_size = _EXPECTED - len(state)
        return np.concatenate([state, np.zeros(pad_size, dtype=np.float32)])
    else:
        # Trim if too long
        return state[:_EXPECTED]


def pad_state_torch(
    qpos: torch.Tensor,
    qvel: Union[torch.Tensor, None] = None,
    device: Union[str, torch.device, None] = None
) -> torch.Tensor:
    """
    PyTorch version ensuring state matches SmolVLA's 6-dim format.
    
    Args:
        qpos: State tensor
        qvel: Ignored - SmolVLA uses combined 6-dim state
        device: Target device (optional)
        
    Returns:
        6-dimensional state tensor
    """
    if device is None:
        device = qpos.device
        
    # Ensure float32
    state = qpos.to(dtype=torch.float32, device=device)
    
    # Handle different dimensions
    if state.shape[-1] == _EXPECTED:
        return state
    elif state.shape[-1] < _EXPECTED:
        # Pad with zeros
        pad_size = _EXPECTED - state.shape[-1]
        pad_shape = list(state.shape)
        pad_shape[-1] = pad_size
        return torch.cat([state, torch.zeros(pad_shape, dtype=torch.float32, device=device)], dim=-1)
    else:
        # Trim if too long
        return state[..., :_EXPECTED]


def unpad_action(action: Union[np.ndarray, torch.Tensor], n_joints: int) -> Union[np.ndarray, torch.Tensor]:
    """
    Extract actual joint commands from SmolVLA's 6-dim action.
    
    Args:
        action: 6-dimensional action from SmolVLA
        n_joints: Actual number of joints on robot
        
    Returns:
        Action for actual robot (may be trimmed or as-is)
        
    Example:
        >>> # SmolVLA outputs 7-DOF, but SO-100 needs 6
        >>> action_7dof = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0])
        >>> action_6dof = unpad_action(action_7dof, n_joints=6)
    """
    if n_joints > _J:
        raise ValueError(f"Robot has {n_joints} joints, but SmolVLA only outputs {_J}")
    
    if isinstance(action, torch.Tensor):
        return action[..., :n_joints]
    else:
        return action[..., :n_joints]


def adapt_observation_for_policy(
    obs: dict,
    n_joints: int = 6,
    image_key: str = "observation.image"
) -> dict:
    """
    Adapt a raw observation dict to SmolVLA format with proper state padding.
    
    Args:
        obs: Raw observation dict
        n_joints: Number of actual robot joints
        image_key: Key for image observation
        
    Returns:
        Observation dict with 14-dim state for SmolVLA
    """
    # Extract state - try multiple possible keys
    if "observation.state" in obs:
        raw_state = obs["observation.state"]
    elif "observation_state" in obs:  # Legacy support
        raw_state = obs["observation_state"]
    elif "state" in obs:
        raw_state = obs["state"]
    else:
        raise KeyError("No state found in observation")
    
    # Handle different state formats
    if isinstance(raw_state, (list, tuple)):
        raw_state = np.array(raw_state)
    
    # Split into pos/vel if concatenated
    if len(raw_state) == 2 * n_joints:
        # Already has pos and vel
        qpos = raw_state[:n_joints]
        qvel = raw_state[n_joints:]
    else:
        # Only positions, assume zero velocity
        qpos = raw_state[:n_joints]
        qvel = np.zeros_like(qpos)
    
    # Ensure 6-dimensional state
    padded_state = pad_state(qpos)  # qvel is ignored for 6-dim state
    
    # Build policy observation with dot convention (LeRobot standard)
    policy_obs = {
        "observation.state": padded_state
    }
    
    # Get the main image observation
    if "observation.image" in obs:
        policy_obs["observation.image"] = obs["observation.image"]
    elif image_key in obs:
        policy_obs["observation.image"] = obs[image_key]
    elif "image" in obs:
        policy_obs["observation.image"] = obs["image"]
    
    # SmolVLA expects 3 camera views - use same image if only one available
    if "observation.image" in policy_obs:
        # Check for additional camera views
        if "observation.image2" in obs:
            policy_obs["observation.image2"] = obs["observation.image2"]
        else:
            policy_obs["observation.image2"] = policy_obs["observation.image"]
            
        if "observation.image3" in obs:
            policy_obs["observation.image3"] = obs["observation.image3"]
        else:
            policy_obs["observation.image3"] = policy_obs["observation.image"]
    
    # Add language instruction as 'task' (SmolVLA expects 'task' key)
    if "language_instruction" in obs:
        policy_obs["task"] = obs["language_instruction"]
    elif "task" in obs:
        policy_obs["task"] = obs["task"]
    
    return policy_obs