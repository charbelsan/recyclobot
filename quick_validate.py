#!/usr/bin/env python
"""Quick validation that SmolVLA loads correctly"""

import os
# Force single GPU to avoid multi-device issues
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from recyclobot.utils.smolvla_workaround import create_policy_with_workaround
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t0 = time.time()
policy = create_policy_with_workaround(device=device)
print(f"Loaded in {time.time() - t0:.2f}s → {device}")

# Check expected features
print("\nExpected features:")
if hasattr(policy.config, 'input_features'):
    for name in policy.config.input_features:
        print(f"  - {name}")

# Quick inference test
# Note: The model expects 6-dim state based on normalization buffers
# The model expects 'task' as a string for language conditioning
obs = {
    "observation.image": torch.randn(1, 3, 256, 256, device=device),
    "observation.image2": torch.randn(1, 3, 256, 256, device=device),
    "observation.image3": torch.randn(1, 3, 256, 256, device=device),
    "observation.state": torch.randn(1, 6, device=device),  # 6-dim to match normalization
    "task": "pick up the bottle"  # Language instruction as 'task'
    

}

print(f"\nTrying inference with keys: {list(obs.keys())}")

# First test without language instruction
with torch.no_grad():
    try:
        if hasattr(policy, 'select_action'):
            action = policy.select_action(obs)
            print(f"✅ Inference works! Action shape: {action.shape}")
        else:
            print("⚠️  No select_action method found")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        print("\nDebugging info:")
        if hasattr(policy.config, 'input_features'):
            print(f"Expected features: {list(policy.config.input_features.keys())}")
        print(f"Provided features: {list(obs.keys())}")

# The model might have a default task or expect task differently
print("\n--- Understanding the 'task' requirement ---")
print("Note: SmolVLA is a vision-language model, but language might be:")
print("1. Hardcoded in the pretrained model")
print("2. Set through a different API")
print("3. Required as part of model initialization")

# Check if the model has task-related attributes
if hasattr(policy, 'task'):
    print(f"\nFound policy.task: {policy.task}")
if hasattr(policy, 'default_task'):
    print(f"Found policy.default_task: {policy.default_task}")
    
# The error suggests the model internally expects a 'task' variable
# This might be a limitation of the pretrained model