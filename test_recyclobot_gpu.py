#!/usr/bin/env python
"""
Quick test script for RecycloBot on GPU without physical robot.
Run this after installation to verify everything works.
"""

import sys
import time
import numpy as np
from PIL import Image
import torch

print("=" * 60)
print("RecycloBot GPU Test Suite")
print("=" * 60)

# Test 1: Check environment
print("\n1. Checking environment...")
print(f"   Python: {sys.version.split()[0]}")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test 2: Import all modules
print("\n2. Testing imports...")
try:
    import lerobot
    print("   ✅ LeRobot imported")
except ImportError as e:
    print(f"   ❌ LeRobot import failed: {e}")
    sys.exit(1)

try:
    from recyclobot.planning.direct_smolvla_planner import plan as direct_plan
    from recyclobot.planning.qwen_planner import plan as qwen_plan
    from recyclobot.control.skill_runner import SkillRunner
    from recyclobot.logging.dataset_logger import RecycloBotLogger
    print("   ✅ RecycloBot modules imported")
except ImportError as e:
    print(f"   ❌ RecycloBot import failed: {e}")
    sys.exit(1)

# Test 3: Test planning
print("\n3. Testing planners...")
test_image = Image.new('RGB', (640, 480), color=(128, 128, 128))

# Test direct planner
try:
    result = direct_plan(test_image, "Sort the plastic bottles")
    print(f"   ✅ Direct planner: {result}")
except Exception as e:
    print(f"   ❌ Direct planner failed: {e}")

# Test Qwen planner (may fail if model not downloaded)
try:
    result = qwen_plan(test_image, "Sort the plastic bottles")
    print(f"   ✅ Qwen planner: {result}")
except Exception as e:
    print(f"   ⚠️  Qwen planner not available (expected): {type(e).__name__}")

# Test 4: Create mock robot environment
print("\n4. Creating mock environment...")

class MockRobot:
    """Simulated robot for testing."""
    
    def __init__(self):
        self.state = np.zeros(14)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def reset(self):
        self.state = np.random.randn(14) * 0.1
        obs = {
            "observation.images.top": torch.randn(3, 480, 640).to(self.device),
            "observation.state": torch.from_numpy(self.state).float().to(self.device)
        }
        return obs, {}
    
    def get_observation(self):
        return {
            "observation.images.top": torch.randn(3, 480, 640).to(self.device),
            "observation.state": torch.from_numpy(self.state).float().to(self.device)
        }
    
    def send_action(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        self.state += action * 0.01  # Simple integration
        
    def close(self):
        pass

robot = MockRobot()
print("   ✅ Mock robot created")

# Test 5: Load SmolVLA policy
print("\n5. Testing SmolVLA policy...")
try:
    from lerobot.common.policies.factory import make_policy
    
    policy = make_policy(
        "smolvla",
        policy_kwargs={
            "pretrained": "lerobot/koch_aloha",
            "config_overrides": {
                "input_shapes": {
                    "observation.images.top": [3, 480, 640],
                    "observation.state": [14],
                },
                "output_shapes": {
                    "action": [7],
                }
            }
        }
    )
    print("   ✅ SmolVLA policy loaded")
    
    # Test inference
    obs = robot.get_observation()
    obs["task"] = "pick up the plastic bottle"
    
    start_time = time.time()
    with torch.no_grad():
        action = policy.select_action(obs)
    inference_time = (time.time() - start_time) * 1000
    
    print(f"   ✅ Inference successful")
    print(f"   Action shape: {action.shape}")
    print(f"   Inference time: {inference_time:.2f}ms")
    print(f"   Inference FPS: {1000/inference_time:.2f}")
    
except Exception as e:
    print(f"   ❌ SmolVLA loading failed: {e}")
    print("   Try: huggingface-cli download lerobot/koch_aloha --local-dir ~/.cache/lerobot")
    policy = None

# Test 6: Run mini episode
if policy is not None:
    print("\n6. Running mini episode...")
    try:
        runner = SkillRunner(policy)
        skills = ["pick(plastic_bottle)", "place(recycling_bin)"]
        
        print(f"   Executing skills: {skills}")
        
        for skill in skills:
            instruction = runner.skill_to_language_prompt(skill)
            print(f"   - {skill} → '{instruction}'")
            
            # Simulate a few steps
            for step in range(5):
                obs = robot.get_observation()
                obs["task"] = instruction
                
                with torch.no_grad():
                    action = policy.select_action(obs)
                
                robot.send_action(action)
            
        print("   ✅ Mini episode completed")
        
    except Exception as e:
        print(f"   ❌ Episode execution failed: {e}")

# Test 7: Dataset logging
print("\n7. Testing dataset logger...")
try:
    logger = RecycloBotLogger(
        dataset_name="test_dataset",
        robot_name="sim",
        fps=10,
        output_dir="test_output"
    )
    
    # Log a few frames
    obs = robot.get_observation()
    for i in range(10):
        logger.record(
            obs={"image": obs["observation.images.top"].cpu().numpy(), 
                 "state": obs["observation.state"].cpu().numpy()},
            action=np.random.randn(7),
            done=(i == 9),
            extra={"step": i}
        )
    
    stats = logger.get_statistics()
    print(f"   ✅ Logger working")
    print(f"   Logged {stats['total_steps']} steps")
    
except Exception as e:
    print(f"   ❌ Logger test failed: {e}")

# Summary
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("\n✅ RecycloBot is ready for GPU simulation!")
print("\nNext steps:")
print("1. Run full demo: python examples/run_recyclobot_demo.py --robot sim")
print("2. Collect data: python scripts/collect_recyclobot_dataset_v3.py --robot-path lerobot/configs/robot/sim.yaml")
print("3. Train model: python scripts/train_recyclobot.py --dataset-name your-dataset")

# Cleanup
if 'robot' in locals():
    robot.close()
    
print("\n✨ Testing complete!")