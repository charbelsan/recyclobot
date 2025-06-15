"""Tests for state adapter functionality."""

import numpy as np
import pytest
import torch

from recyclobot.control.adapters import (
    adapt_observation_for_policy,
    pad_state,
    pad_state_torch,
    unpad_action,
)


class TestStateAdapter:
    """Test state padding and adaptation functions."""
    
    def test_pad_state_6dof(self):
        """Test padding 6-DOF state to 14 dimensions."""
        qpos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        qvel = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
        
        padded = pad_state(qpos, qvel)
        
        assert padded.shape == (14,)
        assert np.allclose(padded[:6], qpos)
        assert padded[6] == 0.0  # 7th joint position (padded)
        assert np.allclose(padded[7:13], qvel)
        assert padded[13] == 0.0  # 7th joint velocity (padded)
    
    def test_pad_state_no_velocity(self):
        """Test padding when velocity is not provided."""
        qpos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        padded = pad_state(qpos)
        
        assert padded.shape == (14,)
        assert np.allclose(padded[:6], qpos)
        assert np.allclose(padded[7:], 0.0)  # All velocities zero
    
    def test_pad_state_7dof(self):
        """Test that 7-DOF state passes through unchanged."""
        qpos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        qvel = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
        
        padded = pad_state(qpos, qvel)
        
        assert padded.shape == (14,)
        assert np.allclose(padded[:7], qpos)
        assert np.allclose(padded[7:], qvel)
    
    def test_pad_state_too_many_joints(self):
        """Test error when more than 7 joints provided."""
        qpos = np.zeros(8)
        
        with pytest.raises(ValueError, match="SmolVLA base expects ≤7"):
            pad_state(qpos)
    
    @pytest.mark.gpu
    def test_pad_state_torch(self):
        """Test PyTorch version of state padding."""
        qpos = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        qvel = torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
        
        padded = pad_state_torch(qpos, qvel)
        
        assert padded.shape == (14,)
        assert torch.allclose(padded[:6], qpos)
        assert padded[6].item() == 0.0
        assert torch.allclose(padded[7:13], qvel)
        assert padded[13].item() == 0.0
    
    def test_unpad_action(self):
        """Test unpadding 7-DOF action to 6-DOF."""
        action_7dof = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0])
        
        action_6dof = unpad_action(action_7dof, n_joints=6)
        
        assert action_6dof.shape == (6,)
        assert np.allclose(action_6dof, action_7dof[:6])
    
    def test_adapt_observation_for_policy(self):
        """Test full observation adaptation."""
        obs = {
            "image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),  # 6 positions only
            "task": "pick up the bottle"
        }
        
        adapted = adapt_observation_for_policy(obs, n_joints=6)
        
        assert "observation.image" in adapted
        assert "observation.state" in adapted
        assert adapted["observation.state"].shape == (14,)
        assert adapted["observation.state"][:6].tolist() == obs["state"].tolist()
        assert np.allclose(adapted["observation.state"][6:], 0.0)
        assert adapted["language_instruction"] == "pick up the bottle"
    
    def test_adapt_observation_with_full_state(self):
        """Test adaptation when state already includes velocities."""
        obs = {
            "observation.image": np.random.randint(0, 255, (256, 256, 3)),
            "observation.state": np.concatenate([
                np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),  # positions
                np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])   # velocities
            ])
        }
        
        adapted = adapt_observation_for_policy(obs, n_joints=6)
        
        assert adapted["observation.state"].shape == (14,)
        # Check positions
        assert np.allclose(adapted["observation.state"][:6], obs["observation.state"][:6])
        # Check velocities
        assert np.allclose(adapted["observation.state"][7:13], obs["observation.state"][6:12])