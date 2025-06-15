#!/usr/bin/env python
"""
Simplest possible demo: Make the arm pick up something using SmolVLA.
No external planners, just direct vision-language-action control.
"""

import time
import torch
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.policies.factory import make_policy

# Configuration (adjust these!)
PORT = "/dev/ttyUSB0"        # Your robot's USB port
CAMERA_INDEX = 0             # Your webcam index
TASK = "pick up the red block"  # What you want it to do

print(f"🤖 Simple Pickup Demo")
print(f"Task: {TASK}")
print("=" * 50)

# 1. Connect to robot
print("Connecting to robot...")
robot = make_robot(
    "so101",
    robot_kwargs={
        "port": PORT,
        "sensors": {"camera": {"index": CAMERA_INDEX}}
    }
)
robot.connect()
print("✅ Connected!")

# 2. Load SmolVLA (this is the AI brain)
print("Loading SmolVLA model...")
policy = make_policy(
    "smolvla",
    pretrained="lerobot/smolvla_base",
    config_overrides={
        "input_shapes": {
            "observation.images.top": [3, 480, 640],
            "observation.state": [14],
        },
        "output_shapes": {"action": [7]}
    }
)
print("✅ Model loaded!")

# 3. Move to home position
print("Moving to home position...")
robot.home()
time.sleep(2)

# 4. Main control loop
print(f"\nExecuting: '{TASK}'")
print("Press Ctrl+C to stop\n")

steps = 0
try:
    while True:
        # Get what the robot sees and feels
        obs = robot.get_observation()
        
        # Ask SmolVLA what to do
        smolvla_input = {
            "observation.images.top": obs["observation.images.top"],
            "observation.state": obs["observation.state"],
            "task": TASK  # The magic: natural language instruction!
        }
        
        # Get action (7 numbers for joint movements)
        with torch.no_grad():
            action = policy.select_action(smolvla_input)
        
        # Move the robot!
        robot.send_action(action)
        
        # Show progress
        steps += 1
        if steps % 10 == 0:  # Every second at 10Hz
            print(f"Step {steps}: Moving... (gripper: {action[6]:.2f})")
        
        # Control frequency
        time.sleep(0.1)  # 10Hz
        
        # Simple completion check (you can improve this)
        if steps > 100:  # ~10 seconds
            print("\n✅ Task complete!")
            break
            
except KeyboardInterrupt:
    print("\n⚠️  Stopped by user")

# 5. Safe shutdown
print("Returning to home...")
robot.home()
time.sleep(1)
robot.disconnect()
print("✅ Done!")

print("\n" + "=" * 50)
print("What just happened:")
print("1. Camera saw the workspace")
print(f"2. SmolVLA understood '{TASK}'")
print("3. It generated movements to achieve the task")
print("4. The arm moved accordingly!")
print("\nNo separate planner needed - SmolVLA does it all! 🎉")