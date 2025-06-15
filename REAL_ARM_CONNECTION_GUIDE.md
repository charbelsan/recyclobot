# 🦾 SO-ARM100 Real Robot Connection Guide

This guide shows exactly how to connect your SO-ARM100 robot and make it move using SmolVLA's direct mode.

## 📋 What You Need

### Hardware:
- **SO-ARM100** robot arm (6 DOF + gripper)
- **USB cable** (usually USB-A to USB-C or micro-USB)
- **Power supply** for the arm (12V typically)
- **Webcam** (USB webcam, positioned to see workspace)
- **Computer** with Ubuntu 20.04/22.04

### Software (already installed if you followed setup):
- RecycloBot package
- LeRobot with SmolVLA
- PyTorch with CUDA (if using GPU)

## 🔌 Step 1: Physical Connection

### 1.1 Connect the Robot
```bash
# 1. Plug in power supply to SO-ARM100
# 2. Connect USB cable from robot to computer
# 3. Power on the robot (switch/button on base)

# Check if robot is detected
ls -la /dev/ttyUSB* /dev/ttyACM*
# You should see something like: /dev/ttyUSB0 or /dev/ttyACM0
```

### 1.2 Set USB Permissions
```bash
# Give yourself permission to access the USB device
sudo usermod -a -G dialout $USER
# IMPORTANT: Log out and log back in for this to take effect!

# Or for immediate access (temporary):
sudo chmod 666 /dev/ttyUSB0  # Replace with your device
```

### 1.3 Connect Webcam
```bash
# Plug in USB webcam
# Check if detected
ls -la /dev/video*
# You should see: /dev/video0, /dev/video1, etc.

# Test webcam
cheese  # Or: vlc v4l2:///dev/video0
```

## 🧪 Step 2: Test Basic Connection

### 2.1 Simple Connection Test
```python
# test_arm_connection.py
import serial
import time

# Try to connect to the arm
try:
    # Replace with your actual device
    port = "/dev/ttyUSB0"  # or /dev/ttyACM0
    
    # SO-ARM100 typically uses 115200 baud
    ser = serial.Serial(port, 115200, timeout=1)
    print(f"✅ Connected to {port}")
    
    # Send a simple command (home position)
    ser.write(b"#000P1500T1000!")  # Servo 0 to center position
    time.sleep(1)
    
    ser.close()
    print("✅ Basic communication successful!")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("Check: 1) Power on? 2) USB connected? 3) Correct port?")
```

### 2.2 Test with LeRobot
```python
# test_lerobot_connection.py
from lerobot.common.robot_devices.robots.factory import make_robot

try:
    # Create robot instance
    robot = make_robot(
        "so101",
        robot_kwargs={
            "port": "/dev/ttyUSB0",  # Adjust to your port
            "sensors": {
                "camera": {"index": 0}  # Webcam index
            }
        }
    )
    
    # Connect
    robot.connect()
    print("✅ Connected to SO-101!")
    
    # Get current state
    state = robot.get_state()
    print(f"Joint positions: {state}")
    
    # Move to home position
    robot.home()
    print("✅ Moved to home position")
    
    robot.disconnect()
    
except Exception as e:
    print(f"❌ Error: {e}")
```

## 🚀 Step 3: Run RecycloBot with Real Arm

### 3.1 Simplest Test - Direct SmolVLA Control
```bash
# This uses SmolVLA directly (no external planner)
python examples/run_recyclobot_demo.py \
    --robot so101 \
    --prompt "Pick up the red block" \
    --episodes 1
```

### 3.2 What Happens Inside:
1. **Connects to robot** via USB serial
2. **Captures image** from webcam
3. **SmolVLA processes**: image + "Pick up the red block"
4. **Generates actions**: 7D joint commands
5. **Sends to robot**: Moves physical arm!

### 3.3 Step-by-Step Code
```python
# simple_arm_demo.py
import torch
from PIL import Image
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.policies.factory import make_policy

# 1. Connect to robot
robot = make_robot("so101", robot_kwargs={"port": "/dev/ttyUSB0"})
robot.connect()

# 2. Load SmolVLA
policy = make_policy(
    "smolvla",
    pretrained="lerobot/smolvla_base",
    config_overrides={
        "input_shapes": {
            "observation.images.top": [3, 480, 640],
            "observation.state": [14],  # 7 joints × 2 (pos + vel)
        },
        "output_shapes": {
            "action": [7],  # 6 joints + gripper
        }
    }
)

# 3. Main control loop
print("Starting control loop... Press Ctrl+C to stop")
task = "pick up the red block"

try:
    while True:
        # Get observation
        obs = robot.get_observation()
        
        # Format for SmolVLA
        smolvla_obs = {
            "observation.images.top": obs["observation.images.top"],
            "observation.state": obs["observation.state"],
            "task": task  # Natural language instruction
        }
        
        # Get action from SmolVLA
        with torch.no_grad():
            action = policy.select_action(smolvla_obs)
        
        # Send to robot
        robot.send_action(action)
        
        # Control frequency (10Hz)
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\nStopping...")
    robot.home()  # Return to safe position
    robot.disconnect()
```

## 🎯 Step 4: Test Different Tasks

### 4.1 Basic Pick and Place
```bash
# Place a plastic bottle in front of the robot
python examples/run_recyclobot_demo.py \
    --robot so101 \
    --prompt "Pick up the plastic bottle and place it in the blue bin" \
    --episodes 1
```

### 4.2 Sorting Task
```bash
# Place multiple objects
python examples/run_recyclobot_demo.py \
    --robot so101 \
    --prompt "Sort the items on the table into recycling and trash" \
    --episodes 1
```

### 4.3 Interactive Mode
```python
# interactive_control.py
from lerobot.common.robot_devices.robots.factory import make_robot
from recyclobot.control.skill_runner import SkillRunner
from lerobot.common.policies.factory import make_policy

# Setup
robot = make_robot("so101")
robot.connect()
policy = make_policy("smolvla", pretrained="lerobot/smolvla_base")
runner = SkillRunner(policy)

# Interactive loop
while True:
    command = input("\nEnter command (or 'quit'): ")
    if command.lower() == 'quit':
        break
    
    # Execute natural language command
    obs = robot.get_observation()
    
    # Direct execution with SmolVLA
    print(f"Executing: {command}")
    
    # Create simple skill from command
    if "pick" in command.lower():
        skills = ["pick(object)"]
    elif "place" in command.lower():
        skills = ["place(target)"]
    else:
        skills = ["inspect()"]
    
    runner.run(skills, robot)

robot.disconnect()
```

## 🔧 Step 5: Troubleshooting

### Robot Not Moving?
```bash
# 1. Check power
# LED on robot should be on

# 2. Check USB connection
dmesg | tail -20  # Look for USB device detection

# 3. Check permissions
groups  # Should include 'dialout'

# 4. Test with simple serial command
screen /dev/ttyUSB0 115200  # Ctrl+A, K to exit
```

### Camera Not Working?
```bash
# Find correct camera index
for i in 0 1 2; do
    echo "Testing /dev/video$i"
    python -c "import cv2; cap=cv2.VideoCapture($i); print(f'Camera {i}: {cap.isOpened()}'); cap.release()"
done

# Use the working index in your config
--robot-overrides sensors.camera.index=1
```

### Arm Moving Erratically?
```python
# Reduce speed and add safety limits
robot = make_robot(
    "so101",
    robot_kwargs={
        "max_velocity": 0.5,  # Slower movements
        "safety_limits": True
    }
)
```

## 📊 Step 6: Monitor Performance

### 6.1 Visual Feedback
```python
# visualize_control.py
import cv2
import matplotlib.pyplot as plt

# During control loop
while True:
    obs = robot.get_observation()
    
    # Show camera feed
    image = obs["observation.images.top"]
    cv2.imshow("Robot View", image)
    
    # Show joint states
    plt.clf()
    plt.bar(range(7), obs["observation.state"][:7])
    plt.ylim(-3.14, 3.14)
    plt.title("Joint Positions")
    plt.pause(0.01)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 6.2 Log Data
```bash
# Record your session
python examples/run_recyclobot_demo.py \
    --robot so101 \
    --prompt "Pick up the can" \
    --output my_robot_data \
    --episodes 5
```

## 🎓 Understanding the Flow

```
1. You say: "Pick up the red block"
                    ↓
2. Camera captures image of workspace
                    ↓
3. SmolVLA receives:
   - Image (what it sees)
   - Text: "Pick up the red block"
   - Current joint positions
                    ↓
4. SmolVLA outputs:
   - 7 numbers: [j0, j1, j2, j3, j4, j5, gripper]
   - Example: [0.1, -0.2, 0.3, 0.0, 0.1, 0.0, 1.0]
                    ↓
5. Robot moves joints to match these targets
                    ↓
6. Repeat at 10Hz until task complete
```

## 🚨 Safety First!

1. **Always** have emergency stop ready (power button)
2. **Start slow** - reduce max velocity initially
3. **Clear workspace** - no obstacles initially
4. **Test in simulation first** if unsure
5. **Keep hands clear** during operation

## 🎬 Quick Start Commands

```bash
# 1. Most basic test
python examples/run_recyclobot_demo.py --robot so101 --episodes 1

# 2. With specific USB port
python examples/run_recyclobot_demo.py \
    --robot so101 \
    --robot-overrides port=/dev/ttyACM0

# 3. With different camera
python examples/run_recyclobot_demo.py \
    --robot so101 \
    --robot-overrides sensors.camera.index=1

# 4. Slower and safer
python examples/run_recyclobot_demo.py \
    --robot so101 \
    --robot-overrides max_velocity=0.3
```

## 🎯 Next Steps

1. **Collect demonstrations** for your specific objects
2. **Fine-tune SmolVLA** on your data
3. **Add custom skills** for your recycling setup
4. **Optimize camera position** for best view

Remember: The base SmolVLA knows general manipulation, but fine-tuning teaches it YOUR specific recycling setup!

Need help? Check the robot LED status:
- 🟢 Green: Ready
- 🟡 Yellow: Moving
- 🔴 Red: Error/E-stop

Happy recycling! 🤖♻️