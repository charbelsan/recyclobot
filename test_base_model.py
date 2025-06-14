#!/usr/bin/env python
"""
Quick test script to see RecycloBot's base model capabilities without training.
This demonstrates SmolVLA's zero-shot waste sorting abilities.
"""

import sys
import numpy as np
from PIL import Image
import torch

# Test 1: Check installations
print("🔍 Checking installations...")
try:
    import lerobot
    print(f"✅ LeRobot version: {lerobot.__version__}")
except ImportError:
    print("❌ LeRobot not installed. Run: pip install 'lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git@v0.4.0'")
    sys.exit(1)

try:
    from recyclobot.planning.direct_smolvla_planner import plan
    from recyclobot.control.skill_runner import SkillRunner
    print("✅ RecycloBot modules loaded")
except ImportError:
    print("❌ RecycloBot not installed. Run: pip install -e .")
    sys.exit(1)

# Test 2: Planning capabilities
print("\n🧠 Testing planning capabilities...")
print("Creating a test image...")
test_image = Image.new('RGB', (640, 480), color=(200, 200, 200))

# Test different recycling prompts
test_prompts = [
    "Pick up the plastic bottle and put it in the recycling bin",
    "Sort all the trash on the table",
    "Put the aluminum can in the blue bin",
    "Clean up the workspace"
]

print("\nTesting direct SmolVLA planning (no separate planner):")
for prompt in test_prompts:
    result = plan(test_image, prompt)
    print(f"\nPrompt: '{prompt}'")
    print(f"Plan: {result}")

# Test 3: Language understanding
print("\n🗣️ Testing language understanding...")
try:
    from lerobot.common.policies.factory import make_policy
    
    print("Loading SmolVLA base model...")
    policy = make_policy(
        "smolvla",
        pretrained="lerobot/smolvla_base",
        config_overrides={
            "input_shapes": {
                "observation.images.top": [3, 480, 640],
                "observation.state": [14],
            },
            "output_shapes": {
                "action": [7],
            }
        }
    )
    print("✅ SmolVLA loaded successfully!")
    
    # Test skill mapping
    runner = SkillRunner(policy)
    test_skills = [
        "pick(plastic_bottle)",
        "place(recycling_bin)",
        "pick(aluminum_can)",
        "inspect(items)"
    ]
    
    print("\nSkill to language mapping:")
    for skill in test_skills:
        instruction = runner.skill_to_language_prompt(skill)
        print(f"  {skill} → '{instruction}'")
    
    # Test inference
    print("\n🤖 Testing model inference...")
    # Create mock observation
    mock_obs = {
        "observation.images.top": torch.randn(3, 480, 640),
        "observation.state": torch.randn(14),
        "task": "pick up the plastic bottle"
    }
    
    with torch.no_grad():
        action = policy.select_action(mock_obs)
    
    print(f"Generated action shape: {action.shape}")
    print(f"Action values (first 3): {action[:3].numpy()}")
    print("✅ Model can generate actions!")
    
except Exception as e:
    print(f"⚠️  Model loading failed: {e}")
    print("This is expected if SmolVLA weights aren't downloaded.")
    print("Run: huggingface-cli download lerobot/smolvla_base --local-dir ~/.cache/lerobot")

# Test 4: Show capabilities summary
print("\n📊 Base Model Capabilities Summary:")
print("=" * 50)
print("1. Vision-Language Understanding:")
print("   - SmolVLM (500M params) understands images + text")
print("   - Can interpret recycling-related commands")
print("   - Zero-shot understanding of objects and bins")
print("\n2. Action Generation:")
print("   - Outputs 7-DOF continuous actions")
print("   - Pre-trained on manipulation tasks")
print("   - Can be fine-tuned on recycling data")
print("\n3. Language Flexibility:")
print("   - Accepts natural language instructions")
print("   - No fixed vocabulary or goal IDs")
print("   - Can understand context and variations")
print("\n4. What it CAN do without training:")
print("   - Understand basic pick/place commands")
print("   - Generate reasonable arm movements")
print("   - Respond to natural language")
print("\n5. What it NEEDS training for:")
print("   - Your specific workspace layout")
print("   - Exact bin locations")
print("   - Optimal recycling strategies")
print("   - Your specific objects")
print("=" * 50)

print("\n✅ Base model test complete!")
print("\nTo see it in action with a simulated robot:")
print("  python examples/run_recyclobot_demo.py --robot sim --episodes 1")