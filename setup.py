"""RecycloBot setup configuration."""

from setuptools import setup, find_packages

setup(
    name="recyclobot",
    version="0.1.0",
    description="Vision-Language Planning for Robotic Waste Sorting with LeRobot",
    author="RecycloBot Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "lerobot[smolvla,feetech] @ git+https://github.com/huggingface/lerobot.git@v0.4.0",
        "google-generativeai>=0.3.0",
        "transformers[vision]>=4.44.0",
        "accelerate>=0.26.0",
        "pillow>=9.0",
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
        "qwen": [
            "bitsandbytes>=0.41.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "recyclobot-demo=examples.run_recyclobot_demo:main",
            "recyclobot-collect=scripts.collect_recyclobot_dataset:main",
            "recyclobot-train=scripts.train_recyclobot:main",
            "recyclobot-eval=scripts.evaluate_recyclobot:main",
        ],
    },
)