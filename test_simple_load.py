#!/usr/bin/env python
"""Simple test to load SmolVLA without assumptions"""

import torch
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Testing SmolVLA loading...")
print("="*60)

# Try the simplest approach first
try:
    print("1. Loading SmolVLA directly...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to(device)
    policy.eval()
    print("✓ Policy loaded successfully")
    
    # Check normalization stats
    print("\n2. Checking normalization stats...")
    has_stats = False
    for name, param in policy.state_dict().items():
        if "normalize_inputs" in name and "mean" in name:
            has_stats = True
            is_inf = torch.isinf(param).any().item()
            print(f"  - {name}: {'INFINITY!' if is_inf else 'OK'}")
    
    if not has_stats:
        print("  ✗ No normalization stats found!")
    
    # Check expected features
    print("\n3. Expected features:")
    if hasattr(policy.config, 'input_features'):
        for feature in policy.config.input_features:
            print(f"  - {feature}")
    
    # Try inference
    print("\n4. Testing inference...")
    obs = {
        "observation.image": torch.randn(1, 3, 256, 256, device=device),
        "observation.state": torch.randn(1, 14, device=device),
        "language_instruction": "pick up the bottle"
    }
    
    with torch.no_grad():
        action = policy.select_action(obs)
        print(f"✓ Inference successful! Action shape: {action.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\nThis might be the normalization stats issue.")
    print("The model needs normalization statistics to work properly.")