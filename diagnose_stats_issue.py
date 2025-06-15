#!/usr/bin/env python
"""Diagnose the normalization stats issue with SmolVLA"""

import torch
import json
import os
from pathlib import Path

print("Diagnosing SmolVLA normalization stats issue")
print("=" * 60)

# 1. Check what files are in the HuggingFace cache
print("\n1. Checking HuggingFace cache for SmolVLA files...")
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
smolvla_dirs = list(cache_dir.glob("models--lerobot--smolvla_base*"))

if smolvla_dirs:
    for d in smolvla_dirs:
        print(f"\nFound cache directory: {d}")
        snapshots = d / "snapshots"
        if snapshots.exists():
            for snapshot in snapshots.iterdir():
                print(f"\n  Snapshot: {snapshot.name}")
                # Look for stats-related files
                for f in snapshot.rglob("*"):
                    if f.is_file() and ("stat" in f.name.lower() or "norm" in f.name.lower() or "config" in f.name.lower()):
                        print(f"    - {f.name} ({f.stat().st_size} bytes)")
else:
    print("  No SmolVLA cache found!")

# 2. Try loading and check what's wrong
print("\n\n2. Loading SmolVLA and checking state...")
try:
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    
    # Check the state dict
    print("\n3. Checking state dict for normalization entries...")
    norm_entries = {}
    for name, tensor in policy.state_dict().items():
        if "normalize" in name:
            norm_entries[name] = {
                "shape": list(tensor.shape),
                "has_inf": torch.isinf(tensor).any().item(),
                "has_nan": torch.isnan(tensor).any().item(),
                "mean_value": float(tensor.mean()) if not torch.isinf(tensor).any() else "inf"
            }
    
    print(f"\nFound {len(norm_entries)} normalization entries:")
    for name, info in norm_entries.items():
        status = "❌" if info["has_inf"] or info["has_nan"] else "✅"
        print(f"  {status} {name}: shape={info['shape']}, inf={info['has_inf']}, nan={info['has_nan']}")
    
    # 4. Check if there's a manual way to load stats
    print("\n4. Looking for alternative stats loading methods...")
    
    # Check if there's a config.json with stats
    if hasattr(policy, 'config'):
        print("\nPolicy config attributes:")
        for attr in dir(policy.config):
            if not attr.startswith('_') and 'stat' in attr.lower():
                print(f"  - {attr}: {getattr(policy.config, attr, 'N/A')}")
    
    # 5. Try manual normalization setup
    print("\n5. Attempting manual normalization fix...")
    
    # Common stats for SmolVLA (these are typical values)
    default_stats = {
        "observation.image": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "observation.state": {"mean": [0.0] * 14, "std": [1.0] * 14},
        "action": {"mean": [0.0] * 7, "std": [1.0] * 7}
    }
    
    # Try to manually set stats if they're infinity
    if hasattr(policy, 'normalize_inputs'):
        for key, stats in default_stats.items():
            if key != "action":  # Only inputs
                buffer_key = key.replace(".", "_")
                mean_name = f"buffer_{buffer_key}_mean"
                std_name = f"buffer_{buffer_key}_std"
                
                if hasattr(policy.normalize_inputs, mean_name):
                    current_mean = getattr(policy.normalize_inputs, mean_name)
                    if torch.isinf(current_mean).any():
                        print(f"  Setting default stats for {key}")
                        policy.normalize_inputs.register_buffer(
                            mean_name, 
                            torch.tensor(stats["mean"], dtype=torch.float32)
                        )
                        policy.normalize_inputs.register_buffer(
                            std_name,
                            torch.tensor(stats["std"], dtype=torch.float32)
                        )
    
    print("\nDiagnosis complete!")
    
except Exception as e:
    print(f"\nError during diagnosis: {e}")
    import traceback
    traceback.print_exc()

# 6. Suggest solutions
print("\n" + "=" * 60)
print("POSSIBLE SOLUTIONS:")
print("=" * 60)
print("1. Clear the cache and re-download:")
print("   rm -rf ~/.cache/huggingface/hub/models--lerobot--smolvla_base")
print("   python -c 'from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy; SmolVLAPolicy.from_pretrained(\"lerobot/smolvla_base\")'")
print("\n2. The model might be missing normalization stats on HuggingFace")
print("   Check: https://huggingface.co/lerobot/smolvla_base/tree/main")
print("\n3. You might need to use a different model or train your own")