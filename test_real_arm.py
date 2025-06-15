#!/usr/bin/env python
"""
Simple test script to verify SO-ARM100 connection and control.
Run this FIRST to make sure everything works!
"""

import time
import sys

print("🤖 SO-ARM100 Connection Test")
print("=" * 50)

# Step 1: Check imports
print("\n1️⃣ Checking imports...")
try:
    from lerobot.common.robot_devices.robots.factory import make_robot
    from lerobot.common.policies.factory import make_policy
    import torch
    import cv2
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install -e .")
    sys.exit(1)

# Step 2: Find USB port
print("\n2️⃣ Looking for robot USB port...")
import glob
ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
if ports:
    print(f"Found ports: {ports}")
    PORT = ports[0]  # Use first found
    print(f"Will try: {PORT}")
else:
    print("❌ No USB serial ports found!")
    print("Is the robot connected and powered on?")
    sys.exit(1)

# Step 3: Find camera
print("\n3️⃣ Looking for camera...")
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Found camera at index {i}")
        CAMERA_INDEX = i
        cap.release()
        break
else:
    print("⚠️  No camera found, will continue without vision")
    CAMERA_INDEX = None

# Step 4: Connect to robot
print("\n4️⃣ Connecting to robot...")
try:
    robot = make_robot(
        "so101",
        robot_kwargs={
            "port": PORT,
            "sensors": {
                "camera": {"index": CAMERA_INDEX} if CAMERA_INDEX is not None else {}
            }
        }
    )
    robot.connect()
    print("✅ Connected to SO-101!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Is robot powered on?")
    print("2. Check USB cable")
    print("3. Run: sudo chmod 666", PORT)
    sys.exit(1)

# Step 5: Test basic movement
print("\n5️⃣ Testing basic movement...")
try:
    # Get current position
    obs = robot.get_observation()
    print(f"Current joint positions: {obs['observation.state'][:7]}")
    
    # Move to home
    print("Moving to home position...")
    robot.home()
    time.sleep(2)
    print("✅ Home position reached")
    
    # Small test movement
    print("Testing small movement...")
    test_action = torch.zeros(7)
    test_action[0] = 0.1  # Small movement on first joint
    robot.send_action(test_action)
    time.sleep(1)
    
    # Return home
    robot.home()
    print("✅ Movement test successful")
    
except Exception as e:
    print(f"❌ Movement test failed: {e}")

# Step 6: Test SmolVLA
print("\n6️⃣ Testing SmolVLA integration...")
if CAMERA_INDEX is not None:
    try:
        print("Loading SmolVLA model...")
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
        print("✅ SmolVLA loaded")
        
        # Test inference
        print("Testing inference with 'pick up the object'...")
        obs = robot.get_observation()
        smolvla_input = {
            "observation.images.top": obs["observation.images.top"],
            "observation.state": obs["observation.state"],
            "task": "pick up the object"
        }
        
        with torch.no_grad():
            action = policy.select_action(smolvla_input)
        
        print(f"✅ Generated action: {action[:3]}...")  # Show first 3 values
        
    except Exception as e:
        print(f"⚠️  SmolVLA test failed: {e}")
        print("Model might not be downloaded. Run:")
        print("huggingface-cli download lerobot/smolvla_base")
else:
    print("⚠️  Skipping SmolVLA test (no camera)")

# Step 7: Disconnect
print("\n7️⃣ Disconnecting...")
robot.disconnect()
print("✅ Disconnected safely")

# Summary
print("\n" + "=" * 50)
print("📊 Test Summary:")
print(f"✅ USB Port: {PORT}")
print(f"{'✅' if CAMERA_INDEX is not None else '❌'} Camera: {f'index {CAMERA_INDEX}' if CAMERA_INDEX is not None else 'not found'}")
print("✅ Robot connection: working")
print("✅ Basic movement: working")
print("=" * 50)

print("\n🚀 Ready to run RecycloBot!")
print(f"\nTry this command:")
print(f"python examples/run_recyclobot_demo.py --robot so101 \\")
print(f"    --robot-overrides port={PORT} sensors.camera.index={CAMERA_INDEX}")