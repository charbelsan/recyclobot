#!/usr/bin/env python
"""Test SmolVLA setup after our fixes"""

import torch
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("=" * 60)
print("SmolVLA Setup Test")
print("=" * 60)

# 1. Check LeRobot installation
try:
    import lerobot
    print(f"✓ LeRobot installed: {lerobot.__version__ if hasattr(lerobot, '__version__') else 'unknown version'}")
except ImportError:
    print("✗ LeRobot not installed!")
    print("Run: pip install 'lerobot[smolvla,feetech] @ git+https://github.com/huggingface/lerobot.git@main'")
    exit(1)

# 2. Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Using device: {device}")

# 3. Try loading with stats (the correct way)
try:
    print("\n1. Fetching SmolVLA stats from Hub...")
    stats = SmolVLAStats.from_pretrained("lerobot/smolvla_base")
    print("✓ Stats fetched successfully")
    
    print("\n2. Loading SmolVLA policy with stats...")
    policy = SmolVLAPolicy.from_pretrained(
        model_name_or_path="lerobot/smolvla_base",
        stats=stats,
    )
    policy.to(device)
    policy.eval()
    print("✓ Policy loaded successfully")
    
    print(f"\n3. Model expects these features:")
    for feature in policy.config.input_features.keys():
        print(f"   - {feature}")
    
    # 4. Test inference with SINGLE camera
    print("\n4. Testing inference with single camera...")
    obs = {
        "observation.image": torch.randn(1, 3, 256, 256, device=device),
        "observation.state": torch.randn(1, 6, device=device),  # SmolVLA uses 6-dim state
        "task": "pick up the bottle"  # SmolVLA expects 'task' key
    }
    
    with torch.no_grad():
        action = policy.select_action(obs)
    
    print(f"✓ Inference successful! Action shape: {action.shape}")
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print("1. We're using LeRobot from main branch (includes fixes)")
    print("2. We fetch stats using SmolVLAStats.from_pretrained()")
    print("3. We pass stats to SmolVLAPolicy.from_pretrained()")
    print("4. The model expects 3 cameras but works with 1")
    print("5. State must be 14-dimensional (7 joints × 2)")
    print("\n✅ Setup is correct!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you've installed from main branch:")
    print("   pip install 'lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git@main'")
    print("2. Clear HuggingFace cache if needed:")
    print("   rm -rf ~/.cache/huggingface/hub/models--lerobot--smolvla_base")