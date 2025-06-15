#!/usr/bin/env python
"""Check what's available in SmolVLA module"""

import importlib
import inspect

print("Checking SmolVLA module contents...\n")

# Check configuration module
try:
    from lerobot.common.policies.smolvla import configuration_smolvla
    print("configuration_smolvla.py contents:")
    for name, obj in inspect.getmembers(configuration_smolvla):
        if not name.startswith('_'):
            print(f"  - {name}: {type(obj).__name__}")
except Exception as e:
    print(f"Error loading configuration_smolvla: {e}")

print("\n" + "="*50 + "\n")

# Check modeling module
try:
    from lerobot.common.policies.smolvla import modeling_smolvla
    print("modeling_smolvla.py contents:")
    for name, obj in inspect.getmembers(modeling_smolvla):
        if not name.startswith('_') and isinstance(obj, type):
            print(f"  - {name}: {type(obj).__name__}")
except Exception as e:
    print(f"Error loading modeling_smolvla: {e}")

print("\n" + "="*50 + "\n")

# Check the actual policy loading
try:
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print("Trying to load SmolVLA policy...")
    
    # Try different approaches
    print("\n1. Direct load without stats:")
    try:
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        print("✓ Direct load successful")
    except Exception as e:
        print(f"✗ Direct load failed: {e}")
    
    print("\n2. Check from_pretrained signature:")
    import inspect
    sig = inspect.signature(SmolVLAPolicy.from_pretrained)
    print(f"Parameters: {list(sig.parameters.keys())}")
    
except Exception as e:
    print(f"Error: {e}")