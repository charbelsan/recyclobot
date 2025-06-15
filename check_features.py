#!/usr/bin/env python
"""Check what feature names the SmolVLA checkpoint expects"""

from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("Loading SmolVLA to check expected features...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

print("\n1. Config input features:")
if hasattr(policy.config, 'input_features'):
    for name, info in policy.config.input_features.items():
        print(f"   {name}: {info}")
else:
    print("   No input_features in config")

print("\n2. Normalization buffers in state dict:")
norm_buffers = {}
for name, tensor in policy.state_dict().items():
    if name.startswith("normalize_inputs.buffer_") and name.endswith(".mean"):
        feature_name = name.split(".")[-2].replace("buffer_", "")
        norm_buffers[feature_name] = tensor.shape
        print(f"   {feature_name}: shape {tensor.shape}")

print("\n3. Expected vs Available:")
if hasattr(policy.config, 'input_features'):
    config_features = set(policy.config.input_features.keys())
    norm_features = set(norm_buffers.keys())
    print(f"   Config features: {config_features}")
    print(f"   Norm features: {norm_features}")
    print(f"   Missing from norm: {config_features - norm_features}")
    print(f"   Extra in norm: {norm_features - config_features}")

print("\n4. Image feature details:")
for name in ["observation.image", "observation_image", "observation.images.top", "observation_images_top"]:
    if name in norm_buffers:
        print(f"   ✓ {name} found in normalization")
    else:
        print(f"   ✗ {name} NOT found")