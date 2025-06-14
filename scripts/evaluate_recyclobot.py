#!/usr/bin/env python
"""
Evaluate RecycloBot system performance

This script evaluates:
1. Planning accuracy (skill sequence quality)
2. Execution success rate
3. Object sorting accuracy
4. Error analysis

Usage:
    # Evaluate on test dataset
    python scripts/evaluate_recyclobot.py \
        --dataset your-username/recyclobot-test \
        --checkpoint outputs/recyclobot_smolvla
        
    # Live evaluation on robot
    python scripts/evaluate_recyclobot.py \
        --robot so101 \
        --live-eval \
        --num-trials 20
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


class RecycloBotEvaluator:
    """Comprehensive evaluation for RecycloBot system."""
    
    def __init__(self, planner, policy, robot=None):
        self.planner = planner
        self.policy = policy
        self.robot = robot
        self.results = defaultdict(list)
        
    def evaluate_planning_accuracy(self, test_data: List[Dict]) -> Dict:
        """Evaluate planning accuracy on test scenarios."""
        
        print("\n" + "="*60)
        print("Evaluating Planning Accuracy")
        print("="*60)
        
        correct_plans = 0
        total_plans = len(test_data)
        
        skill_confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        for scenario in tqdm(test_data, desc="Planning evaluation"):
            # Get ground truth plan
            gt_skills = scenario["ground_truth_skills"]
            
            # Generate predicted plan
            image = Image.open(scenario["image_path"])
            prompt = scenario["prompt"]
            
            try:
                pred_skills = self.planner(image, prompt)
            except Exception as e:
                print(f"Planning failed: {e}")
                pred_skills = []
            
            # Compare plans
            if pred_skills == gt_skills:
                correct_plans += 1
            
            # Detailed analysis
            for gt, pred in zip(gt_skills, pred_skills):
                gt_action = gt.split("(")[0]
                pred_action = pred.split("(")[0] if pred else "none"
                skill_confusion_matrix[gt_action][pred_action] += 1
            
            # Store for analysis
            self.results["planning_comparison"].append({
                "scenario": scenario["name"],
                "ground_truth": gt_skills,
                "predicted": pred_skills,
                "correct": pred_skills == gt_skills
            })
        
        accuracy = correct_plans / total_plans if total_plans > 0 else 0
        
        return {
            "planning_accuracy": accuracy,
            "correct_plans": correct_plans,
            "total_plans": total_plans,
            "skill_confusion_matrix": dict(skill_confusion_matrix)
        }
    
    def evaluate_execution_success(self, test_episodes: List[str]) -> Dict:
        """Evaluate execution success rate."""
        
        print("\n" + "="*60)
        print("Evaluating Execution Success")
        print("="*60)
        
        if not self.robot:
            print("No robot available - skipping execution evaluation")
            return {}
        
        success_count = 0
        partial_success_count = 0
        failure_count = 0
        
        error_types = defaultdict(int)
        
        for episode_path in tqdm(test_episodes, desc="Execution evaluation"):
            # Load episode data
            episode = load_dataset(episode_path)
            
            # Reset robot
            self.robot.reset()
            
            # Execute episode
            success, errors = self._execute_episode(episode)
            
            if success == "full":
                success_count += 1
            elif success == "partial":
                partial_success_count += 1
            else:
                failure_count += 1
            
            # Track error types
            for error in errors:
                error_types[error] += 1
        
        total = len(test_episodes)
        
        return {
            "execution_success_rate": success_count / total if total > 0 else 0,
            "partial_success_rate": partial_success_count / total if total > 0 else 0,
            "failure_rate": failure_count / total if total > 0 else 0,
            "error_types": dict(error_types),
            "total_episodes": total
        }
    
    def evaluate_sorting_accuracy(self, workspace_images: List[Dict]) -> Dict:
        """Evaluate object sorting accuracy using vision."""
        
        print("\n" + "="*60)
        print("Evaluating Sorting Accuracy")
        print("="*60)
        
        correct_sorts = 0
        total_objects = 0
        
        confusion_matrix = {
            "recycling": defaultdict(int),
            "compost": defaultdict(int),
            "trash": defaultdict(int)
        }
        
        for workspace in tqdm(workspace_images, desc="Sorting evaluation"):
            before_image = Image.open(workspace["before_image"])
            after_image = Image.open(workspace["after_image"])
            ground_truth = workspace["object_locations"]
            
            # Detect objects and their bins (simplified)
            detected_sorts = self._detect_sorting_result(before_image, after_image)
            
            # Compare with ground truth
            for obj_id, (obj_type, correct_bin) in ground_truth.items():
                if obj_id in detected_sorts:
                    detected_bin = detected_sorts[obj_id]
                    confusion_matrix[correct_bin][detected_bin] += 1
                    
                    if detected_bin == correct_bin:
                        correct_sorts += 1
                    total_objects += 1
        
        accuracy = correct_sorts / total_objects if total_objects > 0 else 0
        
        return {
            "sorting_accuracy": accuracy,
            "correct_sorts": correct_sorts,
            "total_objects": total_objects,
            "confusion_matrix": confusion_matrix
        }
    
    def evaluate_efficiency_metrics(self, episodes: List[Dict]) -> Dict:
        """Evaluate efficiency metrics like speed and trajectory quality."""
        
        print("\n" + "="*60)
        print("Evaluating Efficiency Metrics")
        print("="*60)
        
        execution_times = []
        trajectory_lengths = []
        gripper_changes = []
        
        for episode in tqdm(episodes, desc="Efficiency evaluation"):
            # Execution time
            exec_time = episode["end_time"] - episode["start_time"]
            execution_times.append(exec_time)
            
            # Trajectory length (sum of joint movements)
            states = np.array(episode["states"])
            trajectory_length = np.sum(np.abs(np.diff(states, axis=0)))
            trajectory_lengths.append(trajectory_length)
            
            # Gripper state changes
            gripper_states = states[:, -1]  # Assuming last dim is gripper
            changes = np.sum(np.abs(np.diff(gripper_states)) > 0.5)
            gripper_changes.append(changes)
        
        return {
            "mean_execution_time": np.mean(execution_times),
            "std_execution_time": np.std(execution_times),
            "mean_trajectory_length": np.mean(trajectory_lengths),
            "mean_gripper_changes": np.mean(gripper_changes),
            "efficiency_score": self._compute_efficiency_score(
                execution_times, trajectory_lengths
            )
        }
    
    def generate_error_analysis(self) -> Dict:
        """Analyze common failure modes."""
        
        print("\n" + "="*60)
        print("Error Analysis")
        print("="*60)
        
        # Planning errors
        planning_errors = defaultdict(int)
        for comp in self.results["planning_comparison"]:
            if not comp["correct"]:
                error_type = self._classify_planning_error(
                    comp["ground_truth"], 
                    comp["predicted"]
                )
                planning_errors[error_type] += 1
        
        # Execution errors
        execution_errors = defaultdict(list)
        for episode in self.results.get("failed_episodes", []):
            error_type = episode.get("error_type", "unknown")
            execution_errors[error_type].append(episode["scenario"])
        
        return {
            "planning_errors": dict(planning_errors),
            "execution_errors": dict(execution_errors),
            "recommendations": self._generate_recommendations(
                planning_errors, execution_errors
            )
        }
    
    def _execute_episode(self, episode) -> Tuple[str, List[str]]:
        """Execute episode and determine success."""
        # Simplified execution logic
        errors = []
        
        try:
            # Execute each action
            for step in episode:
                obs = step["observation"]
                action = self.policy.select_action(obs)
                self.robot.send_action(action)
                time.sleep(0.1)
            
            # Check success (simplified)
            success = "full"  # Would need actual success detection
            
        except Exception as e:
            errors.append(str(e))
            success = "failure"
        
        return success, errors
    
    def _detect_sorting_result(self, before: Image, after: Image) -> Dict:
        """Detect where objects were sorted (simplified)."""
        # In real implementation, would use computer vision
        # to detect object movements
        return {}
    
    def _compute_efficiency_score(self, times: List[float], 
                                  trajectories: List[float]) -> float:
        """Compute overall efficiency score."""
        # Normalize and combine metrics
        time_score = 1.0 / (1.0 + np.mean(times) / 60.0)  # Normalize by 1 minute
        traj_score = 1.0 / (1.0 + np.mean(trajectories) / 100.0)  # Normalize
        return (time_score + traj_score) / 2.0
    
    def _classify_planning_error(self, gt: List[str], pred: List[str]) -> str:
        """Classify type of planning error."""
        if len(pred) == 0:
            return "no_plan_generated"
        elif len(pred) != len(gt):
            return "incorrect_sequence_length"
        else:
            # Check for specific error patterns
            pick_errors = sum(1 for g, p in zip(gt, pred) 
                            if g.startswith("pick") and not p.startswith("pick"))
            place_errors = sum(1 for g, p in zip(gt, pred)
                             if g.startswith("place") and not p.startswith("place"))
            
            if pick_errors > place_errors:
                return "pick_action_errors"
            elif place_errors > pick_errors:
                return "place_action_errors"
            else:
                return "object_misidentification"
    
    def _generate_recommendations(self, planning_errors: Dict, 
                                  execution_errors: Dict) -> List[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []
        
        # Planning recommendations
        if planning_errors.get("object_misidentification", 0) > 5:
            recommendations.append(
                "Consider fine-tuning planner on more diverse object images"
            )
        
        if planning_errors.get("incorrect_sequence_length", 0) > 3:
            recommendations.append(
                "Add more examples with complex multi-step sequences"
            )
        
        # Execution recommendations
        if "gripper_failure" in execution_errors:
            recommendations.append(
                "Check gripper calibration and force settings"
            )
        
        if "collision" in execution_errors:
            recommendations.append(
                "Review workspace setup and obstacle avoidance"
            )
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description="Evaluate RecycloBot performance")
    
    # Evaluation mode
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["planning", "execution", "sorting", "efficiency", "full"],
        help="Evaluation mode"
    )
    
    # Data sources
    parser.add_argument(
        "--dataset",
        type=str,
        help="Test dataset for evaluation"
    )
    parser.add_argument(
        "--test-scenarios",
        type=str,
        help="JSON file with test scenarios"
    )
    
    # Model configuration
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="gemini",
        help="Planner to evaluate"
    )
    
    # Live evaluation
    parser.add_argument(
        "--live-eval",
        action="store_true",
        help="Run live evaluation on robot"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="so101",
        help="Robot type for live evaluation"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of trials for live evaluation"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load planner
    if args.planner == "gemini":
        from recyclobot.planning.gemini_planner import plan
        planner = plan
    elif args.planner == "qwen":
        from recyclobot.planning.qwen_planner import plan
        planner = plan
    else:
        from recyclobot.planning.openai_planner import plan
        planner = plan
    
    # Load policy
    from lerobot.common.policies.factory import make_policy
    
    if args.checkpoint:
        policy = make_policy("smolvla", pretrained=args.checkpoint)
    else:
        policy = make_policy("smolvla", pretrained="lerobot/smolvla_base")
    
    # Initialize robot if needed
    robot = None
    if args.live_eval:
        from lerobot.common.robot_devices.robots.factory import make_robot
        robot = make_robot(args.robot)
        robot.connect()
    
    # Create evaluator
    evaluator = RecycloBotEvaluator(planner, policy, robot)
    
    # Run evaluation
    results = {}
    
    if args.mode in ["planning", "full"]:
        # Load test scenarios
        if args.test_scenarios:
            with open(args.test_scenarios, 'r') as f:
                test_data = json.load(f)
        else:
            # Create simple test scenarios
            test_data = [
                {
                    "name": "simple_recycling",
                    "image_path": "test_images/scene1.jpg",
                    "prompt": "Put the plastic bottle in the recycling bin",
                    "ground_truth_skills": ["pick(plastic_bottle)", "place(recycling_bin)"]
                }
            ]
        
        planning_results = evaluator.evaluate_planning_accuracy(test_data)
        results["planning"] = planning_results
    
    if args.mode in ["execution", "full"] and args.live_eval:
        # Generate test episodes
        test_episodes = [f"episode_{i}" for i in range(args.num_trials)]
        execution_results = evaluator.evaluate_execution_success(test_episodes)
        results["execution"] = execution_results
    
    if args.mode in ["efficiency", "full"] and args.dataset:
        # Load episode data
        dataset = load_dataset(args.dataset)
        episodes = [ep for ep in dataset["test"]]
        efficiency_results = evaluator.evaluate_efficiency_metrics(episodes)
        results["efficiency"] = efficiency_results
    
    # Generate error analysis
    error_analysis = evaluator.generate_error_analysis()
    results["error_analysis"] = error_analysis
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    
    if "planning" in results:
        print(f"Planning Accuracy: {results['planning']['planning_accuracy']:.2%}")
    
    if "execution" in results:
        print(f"Execution Success Rate: {results['execution']['execution_success_rate']:.2%}")
    
    if "efficiency" in results:
        print(f"Efficiency Score: {results['efficiency']['efficiency_score']:.2f}")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Cleanup
    if robot:
        robot.disconnect()


if __name__ == "__main__":
    main()