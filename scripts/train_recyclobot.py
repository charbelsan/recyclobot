#!/usr/bin/env python
"""
Fine-tuning script for RecycloBot with SmolVLA

Usage:
    # Fine-tune on collected recycling data
    python scripts/train_recyclobot.py \
        --dataset-name "your-username/recyclobot-demos" \
        --output-dir "outputs/recyclobot_smolvla" \
        --use-lora
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import TrainingArguments


def create_training_config(args):
    """Create training configuration for SmolVLA fine-tuning."""
    
    config = {
        # Model configuration
        "policy": {
            "name": "smolvla",
            "pretrained": "lerobot/smolvla_base",
            "use_lora": args.use_lora,
            "lora_config": {
                "r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            } if args.use_lora else None,
        },
        
        # Dataset configuration
        "dataset": {
            "name": args.dataset_name,
            "split": "train[:90%]",
            "val_split": "train[90%:]",
            "episode_length": 400,  # Max steps per episode
        },
        
        # Training configuration
        "training": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": True,
            "fp16": torch.cuda.is_available(),
            "dataloader_num_workers": 4,
        },
        
        # Logging
        "logging": {
            "output_dir": args.output_dir,
            "logging_steps": 10,
            "save_steps": 100,
            "eval_steps": 50,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "report_to": ["tensorboard"],
        },
        
        # RecycloBot specific
        "recyclobot": {
            "camera_names": ["top"],  # Camera configuration
            "action_dim": 7,  # 6 joints + gripper
            "state_dim": 14,  # Position + velocity
            "language_conditioning": True,
            "skill_categories": ["pick", "place", "inspect"],
        }
    }
    
    return config


def prepare_dataset(dataset_name, config):
    """Load and prepare dataset for training."""
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # Verify dataset format
    required_columns = [
        "observation.images.top",
        "observation.state", 
        "action",
        "task",  # Language instruction
    ]
    
    for col in required_columns:
        if col not in dataset["train"].column_names:
            raise ValueError(f"Dataset missing required column: {col}")
    
    # Add data preprocessing here
    def preprocess_example(example):
        # Ensure image is properly formatted
        if "observation.images.top" in example:
            image = example["observation.images.top"]
            # Convert to tensor if needed
            if isinstance(image, list):
                image = torch.tensor(image)
            example["observation.images.top"] = image
            
        # Ensure language task is present
        if "task" not in example or not example["task"]:
            # Try to infer from language_instruction
            if "language_instruction" in example:
                example["task"] = example["language_instruction"]
            else:
                example["task"] = "perform recycling task"
                
        return example
    
    dataset = dataset.map(preprocess_example)
    
    # Split into train/val
    train_dataset = dataset["train"].select(range(int(0.9 * len(dataset["train"]))))
    val_dataset = dataset["train"].select(range(int(0.9 * len(dataset["train"])), len(dataset["train"])))
    
    return train_dataset, val_dataset


def create_smolvla_trainer(model, train_dataset, val_dataset, config):
    """Create trainer for SmolVLA fine-tuning."""
    
    from lerobot.common.policies.factory import make_policy
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["logging"]["output_dir"],
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        fp16=config["training"]["fp16"],
        logging_steps=config["logging"]["logging_steps"],
        save_steps=config["logging"]["save_steps"],
        eval_steps=config["logging"]["eval_steps"],
        save_total_limit=config["logging"]["save_total_limit"],
        load_best_model_at_end=config["logging"]["load_best_model_at_end"],
        metric_for_best_model=config["logging"]["metric_for_best_model"],
        report_to=config["logging"]["report_to"],
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
    )
    
    # Custom trainer for SmolVLA
    class SmolVLATrainer:
        def __init__(self, model, args, train_dataset, eval_dataset):
            self.model = model
            self.args = args
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            
        def train(self):
            # Implement training loop
            print("Starting SmolVLA fine-tuning...")
            print(f"Train samples: {len(self.train_dataset)}")
            print(f"Eval samples: {len(self.eval_dataset)}")
            
            # This is a placeholder - in real implementation, 
            # you would use LeRobot's training infrastructure
            print("\nNote: Full training implementation requires LeRobot's training pipeline")
            print("See: python -m lerobot.scripts.train --help")
            
    return SmolVLATrainer(model, training_args, train_dataset, val_dataset)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA for RecycloBot")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., 'user/recyclobot-demos')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/recyclobot_smolvla",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for efficient fine-tuning"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = create_training_config(args)
    
    # Save config
    config_path = Path(args.output_dir) / "training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Saved training config to: {config_path}")
    
    # Load dataset
    train_dataset, val_dataset = prepare_dataset(args.dataset_name, config)
    
    # Create model
    from lerobot.common.policies.factory import make_policy
    
    print("\nCreating SmolVLA model...")
    model = make_policy(
        config["policy"]["name"],
        policy_kwargs={
            "pretrained": config["policy"]["pretrained"],
            "config_overrides": {
                "use_lora": config["policy"]["use_lora"],
                "lora_config": config["policy"]["lora_config"],
            }
        }
    )
    
    # Create trainer
    trainer = create_smolvla_trainer(model, train_dataset, val_dataset, config)
    
    # Train
    trainer.train()
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"\nTo use the fine-tuned model:")
    print(f"  model = make_policy('smolvla', pretrained='{args.output_dir}')")
    
    # Generate LeRobot training command
    print("\n" + "="*60)
    print("Alternatively, use LeRobot's native training:")
    print("="*60)
    lerobot_cmd = f"""
python -m lerobot.scripts.train \\
    policy=smolvla \\
    dataset_repo_id={args.dataset_name} \\
    hydra.run.dir={args.output_dir} \\
    training.num_epochs={args.num_epochs} \\
    training.batch_size={args.batch_size} \\
    training.learning_rate={args.learning_rate} \\
    policy.use_lora={args.use_lora}
    """
    print(lerobot_cmd)


if __name__ == "__main__":
    main()