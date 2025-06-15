#!/usr/bin/env python
"""Test if SmolVLA device issue is fixed"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from recyclobot.utils.smolvla_workaround import create_policy_with_workaround

print("Testing SmolVLA device fix...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

# Load policy
policy = create_policy_with_workaround()

# Check all model components are on same device
print("\nChecking device placement:")
devices = set()
for name, param in policy.named_parameters():
    devices.add(str(param.device))
    if "llama" in name.lower() or "vlm" in name.lower():
        print(f"  {name}: {param.device}")

print(f"\nUnique devices found: {devices}")

# Test inference
print("\nTesting inference...")
obs = {
    "observation.image": torch.randn(1, 3, 256, 256, device="cuda:0"),
    "observation.image2": torch.randn(1, 3, 256, 256, device="cuda:0"),
    "observation.image3": torch.randn(1, 3, 256, 256, device="cuda:0"),
    "observation.state": torch.randn(1, 6, device="cuda:0"),
    "task": "pick up the bottle"
}

try:
    with torch.no_grad():
        action = policy.select_action(obs)
    print(f"✅ Success! Action shape: {action.shape}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()