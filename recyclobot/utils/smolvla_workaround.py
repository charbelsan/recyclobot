"""Workaround for SmolVLA normalization stats issue"""

import os
# Force single GPU usage before importing
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def create_dataset_stats():
    """Create manual dataset stats for SmolVLA"""
    # Based on the actual dimensions from the model (6-dim state and action)
    dataset_stats = {
        "observation.image": {
            "mean": torch.tensor([[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32),  # CHW format
            "std": torch.tensor([[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32),
            "min": torch.tensor([[[0.0]], [[0.0]], [[0.0]]], dtype=torch.float32),
            "max": torch.tensor([[[1.0]], [[1.0]], [[1.0]]], dtype=torch.float32)
        },
        "observation.image2": {
            "mean": torch.tensor([[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32),
            "std": torch.tensor([[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32),
            "min": torch.tensor([[[0.0]], [[0.0]], [[0.0]]], dtype=torch.float32),
            "max": torch.tensor([[[1.0]], [[1.0]], [[1.0]]], dtype=torch.float32)
        },
        "observation.image3": {
            "mean": torch.tensor([[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32),
            "std": torch.tensor([[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32),
            "min": torch.tensor([[[0.0]], [[0.0]], [[0.0]]], dtype=torch.float32),
            "max": torch.tensor([[[1.0]], [[1.0]], [[1.0]]], dtype=torch.float32)
        },
        "observation.state": {
            "mean": torch.zeros(6, dtype=torch.float32),  # 6-dim based on state dict
            "std": torch.ones(6, dtype=torch.float32),
            "min": torch.full((6,), -1.0, dtype=torch.float32),
            "max": torch.full((6,), 1.0, dtype=torch.float32)
        },
        "action": {
            "mean": torch.zeros(6, dtype=torch.float32),  # 6-dim based on state dict
            "std": torch.ones(6, dtype=torch.float32) * 0.1,
            "min": torch.full((6,), -1.0, dtype=torch.float32),
            "max": torch.full((6,), 1.0, dtype=torch.float32)
        }
    }
    return dataset_stats


def load_smolvla_with_manual_stats(model_id="lerobot/smolvla_base", device=None):
    """
    Load SmolVLA with manual normalization stats as a workaround.
    
    This fixes the "mean is infinity" error by directly replacing the infinity values
    in the normalization buffers after loading.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Force single GPU to avoid multi-device issues
    if device == "cuda":
        device = "cuda:0"
        # Set CUDA device before loading to ensure all components use same device
        torch.cuda.set_device(0)
    
    # Load the model first with device_map to force single device
    print("Loading SmolVLA model...")
    
    # Try to load with device_map if supported
    try:
        policy = SmolVLAPolicy.from_pretrained(
            model_id,
            device_map={"": device},  # Force everything to single device
            torch_dtype=torch.float32
        )
    except TypeError:
        # Fallback if device_map not supported
        policy = SmolVLAPolicy.from_pretrained(model_id)
    
    # Manually fix the normalization buffers with infinity
    print("Fixing normalization buffers...")
    
    with torch.no_grad():
        # Fix normalize_inputs buffers (6-dim based on actual model)
        if hasattr(policy, 'normalize_inputs'):
            state_dict = policy.normalize_inputs.state_dict()
            if 'buffer_observation_state.mean' in state_dict:
                # Use copy_ to update buffer values in-place
                policy.normalize_inputs.buffer_observation_state.mean.copy_(torch.zeros(6, dtype=torch.float32))
                policy.normalize_inputs.buffer_observation_state.std.copy_(torch.ones(6, dtype=torch.float32))
        
        # Fix normalize_targets buffers
        if hasattr(policy, 'normalize_targets'):
            state_dict = policy.normalize_targets.state_dict() 
            if 'buffer_action.mean' in state_dict:
                policy.normalize_targets.buffer_action.mean.copy_(torch.zeros(6, dtype=torch.float32))
                policy.normalize_targets.buffer_action.std.copy_(torch.ones(6, dtype=torch.float32) * 0.1)
        
        # Fix unnormalize_outputs buffers
        if hasattr(policy, 'unnormalize_outputs'):
            state_dict = policy.unnormalize_outputs.state_dict()
            if 'buffer_action.mean' in state_dict:
                policy.unnormalize_outputs.buffer_action.mean.copy_(torch.zeros(6, dtype=torch.float32))
                policy.unnormalize_outputs.buffer_action.std.copy_(torch.ones(6, dtype=torch.float32) * 0.1)
    
    # Move to device and set to eval
    print(f"Moving model to {device}...")
    
    # First move the entire policy
    policy = policy.to(device)
    
    # Force all submodules to the same device
    for name, module in policy.named_modules():
        # Move each module to the device
        module.to(device)
        
        # Also check for any buffers that might be on wrong device
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer.device != torch.device(device):
                module.register_buffer(buffer_name, buffer.to(device))
        
        # Check parameters too
        for param_name, param in module.named_parameters(recurse=False):
            if param.device != torch.device(device):
                param.data = param.data.to(device)
    
    # Specifically handle VLM components if they exist
    if hasattr(policy, 'model') and hasattr(policy.model, 'vlm_with_expert'):
        policy.model.vlm_with_expert = policy.model.vlm_with_expert.to(device)
    
    policy.eval()
    
    # Verify the fix
    print("\nVerifying normalization stats...")
    issues = []
    for name, param in policy.state_dict().items():
        if "normalize" in name and ("mean" in name or "std" in name):
            if torch.isinf(param).any():
                issues.append(f"  ❌ {name} still has infinity!")
            elif torch.isnan(param).any():
                issues.append(f"  ❌ {name} has NaN!")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(issue)
    else:
        print("  ✅ All normalization stats look good!")
    
    return policy


def create_policy_with_workaround(device=None):
    """
    Create SmolVLA policy with workaround for the normalization issue.
    
    This is a drop-in replacement for the standard policy creation.
    Always uses manual stats since the pretrained model lacks them.
    """
    try:
        return load_smolvla_with_manual_stats(device=device)
    except Exception as e:
        print(f"Error loading policy: {e}")
        raise