"""
Test RecycloBot dataset logger roundtrip (save and reload)
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset


class TestLoggerRoundtrip(unittest.TestCase):
    """Test dataset logging and reloading."""
    
    def setUp(self):
        """Set up test directory."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test directory."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_logger_basic_recording(self):
        """Test basic recording functionality."""
        from recyclobot.logging.dataset_logger import RecycloBotLogger
        
        logger = RecycloBotLogger(self.test_dir, "test_dataset")
        
        # Start episode
        skills = ["pick(bottle)", "place(recycling_bin)"]
        logger.start_episode("gemini", skills)
        
        # Record some steps
        for i in range(10):
            obs = {
                "image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                "state": np.random.randn(7).astype(np.float32)
            }
            action = np.random.randn(7).astype(np.float32)
            
            logger.record(
                obs=obs,
                action=action,
                done=(i == 9),
                extra={
                    "current_skill": skills[i // 5],
                    "goal_id": i // 5,
                    "language_prompt": f"Step {i}"
                }
            )
        
        # Check files were created
        parquet_file = Path(self.test_dir) / "episode_0001.parquet"
        self.assertTrue(parquet_file.exists())
        
        metadata_file = Path(self.test_dir) / "metadata.json"
        self.assertTrue(metadata_file.exists())
        
    def test_logger_roundtrip_data_integrity(self):
        """Test data integrity through save/reload cycle."""
        from recyclobot.logging.dataset_logger import RecycloBotLogger, RECYCLOBOT_FEATURES
        
        logger = RecycloBotLogger(self.test_dir, "test_roundtrip")
        
        # Create test data
        test_skills = ["pick(can)", "place(recycling_bin)", "pick(banana)", "place(compost_bin)"]
        logger.start_episode("qwen", test_skills)
        
        # Store original data for comparison
        original_data = []
        
        for i in range(20):
            obs = {
                "image": np.ones((224, 224, 3), dtype=np.uint8) * i,  # Distinctive pattern
                "state": np.arange(7, dtype=np.float32) * i
            }
            action = np.arange(7, dtype=np.float32) * (i + 1)
            
            extra = {
                "current_skill": test_skills[i % len(test_skills)],
                "goal_id": i % 4,
                "language_prompt": f"Test prompt {i}",
                "planner_name": "qwen",
                "planner_log": str(test_skills)
            }
            
            original_data.append({
                "obs": obs.copy(),
                "action": action.copy(),
                "extra": extra.copy()
            })
            
            logger.record(obs, action, done=(i == 19), extra=extra)
        
        # Load saved data
        parquet_file = Path(self.test_dir) / "episode_0001.parquet"
        loaded_df = pq.read_table(str(parquet_file)).to_pandas()
        
        # Verify data integrity
        self.assertEqual(len(loaded_df), 20)
        
        for i, row in loaded_df.iterrows():
            # Check step metadata
            self.assertEqual(row["step_id"], i + 1)
            self.assertEqual(row["episode_id"], 1)
            
            # Check skill data
            expected_skill = test_skills[i % len(test_skills)]
            self.assertEqual(row["current_skill"], expected_skill)
            self.assertEqual(row["goal_id"], i % 4)
            self.assertEqual(row["language_prompt"], f"Test prompt {i}")
            
            # Check arrays (state and action)
            np.testing.assert_array_almost_equal(
                row["state"], 
                original_data[i]["obs"]["state"]
            )
            np.testing.assert_array_almost_equal(
                row["action"],
                original_data[i]["action"]
            )
            
    def test_logger_metadata_persistence(self):
        """Test metadata is correctly saved and loaded."""
        from recyclobot.logging.dataset_logger import RecycloBotLogger
        
        logger = RecycloBotLogger(self.test_dir, "test_metadata")
        
        # Record multiple episodes
        for ep in range(3):
            skills = [f"pick(item{ep})", f"place(bin{ep})"]
            logger.start_episode("gemini" if ep % 2 == 0 else "qwen", skills)
            
            for step in range(5):
                obs = {"image": np.zeros((224, 224, 3), dtype=np.uint8),
                       "state": np.zeros(7)}
                logger.record(obs, np.zeros(7), done=(step == 4), extra={})
        
        # Load metadata
        with open(Path(self.test_dir) / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Verify metadata
        self.assertEqual(metadata["dataset_name"], "test_metadata")
        self.assertEqual(len(metadata["recorded_episodes"]), 3)
        
        # Check episode details
        for i, ep_meta in enumerate(metadata["recorded_episodes"]):
            self.assertEqual(ep_meta["episode_id"], i + 1)
            self.assertEqual(ep_meta["planner_name"], "gemini" if i % 2 == 0 else "qwen")
            self.assertEqual(ep_meta["total_steps"], 5)
            self.assertIn("start_time", ep_meta)
            self.assertIn("end_time", ep_meta)
            
    def test_dataset_statistics(self):
        """Test dataset statistics calculation."""
        from recyclobot.logging.dataset_logger import RecycloBotLogger
        
        logger = RecycloBotLogger(self.test_dir, "test_stats")
        
        # Record data with known statistics
        all_skills = []
        for ep in range(2):
            skills = ["pick(bottle)", "place(recycling_bin)", "pick(can)", "place(recycling_bin)"]
            all_skills.extend(skills)
            logger.start_episode("gemini", skills)
            
            for _ in range(10):
                obs = {"image": np.zeros((224, 224, 3)), "state": np.zeros(7)}
                logger.record(obs, np.zeros(7), done=False, extra={})
            logger.record(obs, np.zeros(7), done=True, extra={})
        
        # Get statistics
        stats = logger.get_statistics()
        
        # Verify statistics
        self.assertEqual(stats["total_episodes"], 2)
        self.assertEqual(stats["total_steps"], 22)  # 11 steps per episode
        self.assertAlmostEqual(stats["average_episode_length"], 11.0)
        
        # Check skill usage counts
        self.assertEqual(stats["skill_usage"]["pick"], 4)
        self.assertEqual(stats["skill_usage"]["place"], 4)
        
    def test_dataset_card_generation(self):
        """Test dataset card is generated correctly."""
        from recyclobot.logging.dataset_logger import RecycloBotLogger
        
        logger = RecycloBotLogger(self.test_dir, "test_card")
        
        # Record minimal data
        logger.start_episode("gemini", ["sort()"])
        obs = {"image": np.zeros((224, 224, 3)), "state": np.zeros(7)}
        logger.record(obs, np.zeros(7), done=True, extra={})
        
        # Generate card
        logger.create_dataset_card()
        
        # Check card exists and contains expected content
        card_path = Path(self.test_dir) / "README.md"
        self.assertTrue(card_path.exists())
        
        with open(card_path, "r") as f:
            content = f.read()
            
        # Verify content
        self.assertIn("# test_card", content)
        self.assertIn("RecycloBot", content)
        self.assertIn("SO-ARM100", content)
        self.assertIn("waste sorting", content)
        self.assertIn("Episodes: 1", content)
        
        
if __name__ == "__main__":
    unittest.main()