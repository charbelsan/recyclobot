#!/bin/bash
# RecycloBot Quick Start for GPU Station
# Copy and run these commands on your GPU workstation

echo "🚀 RecycloBot GPU Quick Start"
echo "============================"

# Step 1: Clone repository
echo -e "\n📦 Step 1: Cloning RecycloBot..."
git clone https://github.com/your-username/recyclobot.git
cd recyclobot

# Step 2: Create conda environment
echo -e "\n🐍 Step 2: Setting up Python environment..."
conda create -n recyclobot python=3.10 -y
conda activate recyclobot

# Step 3: Install PyTorch with CUDA
echo -e "\n🔥 Step 3: Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Step 4: Install LeRobot and RecycloBot
echo -e "\n🤖 Step 4: Installing LeRobot and RecycloBot..."
pip install "lerobot[smolvla,sim]>=0.5.0"
pip install -e .

# Step 5: Install additional dependencies
echo -e "\n📚 Step 5: Installing additional dependencies..."
pip install gymnasium opencv-python-headless imageio[ffmpeg] matplotlib

# Step 6: Download SmolVLA weights
echo -e "\n📥 Step 6: Downloading SmolVLA model weights..."
huggingface-cli download lerobot/koch_aloha --local-dir ~/.cache/lerobot

# Step 7: Run tests
echo -e "\n🧪 Step 7: Running tests..."
python test_recyclobot_gpu.py

# Step 8: Run simulation demo
echo -e "\n🎮 Step 8: Running simulation demo..."
python examples/run_recyclobot_demo.py --robot sim --episodes 1 --output test_demo

echo -e "\n✅ Setup complete! RecycloBot is ready on your GPU."
echo -e "\n📝 Next steps:"
echo "  1. Run more demos: python examples/run_recyclobot_demo.py --robot sim"
echo "  2. Collect dataset: python scripts/collect_recyclobot_dataset_v3.py --robot-path lerobot/configs/robot/sim.yaml"
echo "  3. Check GPU usage: nvidia-smi"