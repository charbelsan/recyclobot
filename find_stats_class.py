#!/usr/bin/env python
"""Find where SmolVLAStats is defined"""

# Try different import paths
attempts = [
    ("from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAStats", "configuration_smolvla.SmolVLAStats"),
    ("from lerobot.common.policies.smolvla import SmolVLAStats", "smolvla.SmolVLAStats"),
    ("from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAStats", "modeling_smolvla.SmolVLAStats"),
    ("from lerobot.common.policies.configuration import SmolVLAStats", "configuration.SmolVLAStats"),
]

for import_str, name in attempts:
    try:
        exec(import_str)
        print(f"✓ Found {name}")
        print(f"  Import: {import_str}")
        break
    except ImportError as e:
        print(f"✗ {name}: {e}")

# Also check what's in the HF model repo
print("\n" + "="*60)
print("Checking HuggingFace model contents...")
print("="*60)

try:
    from huggingface_hub import list_repo_files
    files = list_repo_files("lerobot/smolvla_base")
    print("Files in lerobot/smolvla_base:")
    for f in sorted(files):
        if "stat" in f.lower() or "norm" in f.lower() or "config" in f.lower():
            print(f"  - {f}")
except Exception as e:
    print(f"Error checking HF repo: {e}")

# Try the actual loading pattern from quick_validate.py
print("\n" + "="*60)
print("Trying pattern from quick_validate.py...")
print("="*60)

try:
    # Maybe stats are loaded differently
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    # Try loading with different methods
    print("1. Trying direct load:")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    print("✓ Loaded without explicit stats")
    
    # Check if stats are already in the model
    if hasattr(policy, 'normalize_inputs'):
        print("\n2. Checking normalization buffers:")
        for name, tensor in policy.state_dict().items():
            if 'normalize' in name and 'mean' in name:
                print(f"  - {name}: shape {tensor.shape}")
                
except Exception as e:
    print(f"Error: {e}")