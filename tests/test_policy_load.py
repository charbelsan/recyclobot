"""Test that SmolVLA policy loads correctly with expected API."""

import pytest
import torch


@pytest.mark.cpu
def test_smolvla_base_loads():
    """Verify SmolVLA base model loads with correct dimensions."""
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    # Load policy then move to CPU
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to("cpu")
    
    # Verify expected state dimension
    assert policy.config.state_dim == 14, f"Expected state_dim=14, got {policy.config.state_dim}"
    
    # Verify policy is in eval mode
    assert not policy.training, "Policy should be in eval mode"
    
    # Test that it can process a mock observation
    mock_obs = {
        "observation.image": torch.randn(1, 3, 256, 256),
        "observation.state": torch.randn(1, 14),
        "language_instruction": "pick up the object"
    }
    
    with torch.no_grad():
        # Should not raise an error
        action = policy.predict_action(mock_obs)
        assert action.shape[-1] == 7, f"Expected 7-DOF action, got {action.shape}"


@pytest.mark.cpu
def test_device_placement():
    """Test that device placement works correctly."""
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    # Load then move to CPU
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to("cpu")
    
    # Check all parameters are on CPU
    for param in policy.parameters():
        assert param.device.type == "cpu", f"Parameter on wrong device: {param.device}"


@pytest.mark.gpu
def test_gpu_device_placement():
    """Test GPU device placement if available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    # Load then move to GPU
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to("cuda")
    
    # Check all parameters are on GPU
    for param in policy.parameters():
        assert param.device.type == "cuda", f"Parameter on wrong device: {param.device}"


@pytest.mark.cpu
def test_smolvla_load_device_transfer():
    """Test that loading and device transfer works (guards against API changes)."""
    import torch
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    # Load model without any device kwargs
    p = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    # Then transfer to device
    p.to("cpu")
    assert next(p.parameters()).device.type == "cpu"


@pytest.mark.cpu
def test_feature_names_match_checkpoint():
    """Test that feature names match checkpoint normalization stats."""
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    p = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    
    # Get feature names from config
    keys_cfg = set(p.config.input_features.keys()) if hasattr(p.config, 'input_features') else set()
    
    # Get feature names from normalization stats
    keys_stat = {n.split(".")[-2] for n, _ in p.state_dict().items()
                 if n.startswith("normalize_inputs.buffer_") and n.endswith(".mean")}
    
    # Every config feature should have stats
    missing = keys_cfg - keys_stat
    assert not missing, f"Features missing normalization stats: {missing}"
    
    # Print for debugging
    print(f"Config features: {keys_cfg}")
    print(f"Stats features: {keys_stat}")