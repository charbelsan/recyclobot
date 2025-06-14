#!/bin/bash
# Quick setup for GPU station with Python 3.10 already installed

echo "🚀 Quick RecycloBot Setup"

# Install in current directory
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
pip install "lerobot[smolvla,sim,aloha]==0.4.0" && \
pip install -e . && \
pip install "transformers[vision]>=4.44.0" "accelerate>=0.26.0" && \
huggingface-cli download lerobot/koch_aloha --local-dir ~/.cache/lerobot && \
python verify_setup.py && \
echo "✅ Setup complete! Run: python examples/run_recyclobot_demo.py --robot sim"