"""
Test OpenAI-style planner with mocked API responses
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


class TestOpenAIPlanner(unittest.TestCase):
    """Test OpenAI-compatible planner."""
    
    def setUp(self):
        """Create test image and config."""
        self.test_image = Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        self.test_prompt = "Sort the trash"
        
        # Test config
        self.test_config = {
            "planners": {
                "openai": {
                    "api_key": "test-key",
                    "api_base": "https://api.openai.com/v1",
                    "model": "gpt-4-vision-preview"
                }
            }
        }
        
    @patch('recyclobot.planning.openai_planner.requests.post')
    def test_openai_planner_success(self, mock_post):
        """Test successful OpenAI API call."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '["pick(plastic_bottle)", "place(recycling_bin)"]'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        # Import and test
        from recyclobot.planning.openai_planner import OpenAIPlanner
        planner = OpenAIPlanner()
        planner.api_key = "test-key"  # Override for test
        
        skills = planner.plan(self.test_image, self.test_prompt)
        
        # Verify
        self.assertEqual(len(skills), 2)
        self.assertEqual(skills[0], "pick(plastic_bottle)")
        self.assertEqual(skills[1], "place(recycling_bin)")
        
        # Check API was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertIn("Authorization", call_args[1]["headers"])
        self.assertIn("messages", call_args[1]["json"])
        
    @patch('recyclobot.planning.openai_planner.requests.post')
    def test_openai_planner_with_extra_text(self, mock_post):
        """Test handling responses with extra text."""
        # Mock response with extra text
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '''Based on the image, here's the plan:
                    ["pick(aluminum_can)", "place(recycling_bin)", "pick(banana_peel)", "place(compost_bin)"]
                    This will sort the items correctly.'''
                }
            }]
        }
        mock_post.return_value = mock_response
        
        from recyclobot.planning.openai_planner import OpenAIPlanner
        planner = OpenAIPlanner()
        planner.api_key = "test-key"
        
        skills = planner.plan(self.test_image, self.test_prompt)
        
        # Should extract JSON correctly
        self.assertEqual(len(skills), 4)
        self.assertIn("pick(aluminum_can)", skills)
        self.assertIn("place(compost_bin)", skills)
        
    def test_config_loading(self):
        """Test loading configuration from file."""
        import tempfile
        import os
        
        # Create temp config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config, f)
            config_path = f.name
        
        try:
            from recyclobot.planning.openai_planner import OpenAIPlanner
            
            # Test with config file
            planner = OpenAIPlanner(config_path)
            self.assertEqual(planner.api_key, "test-key")
            self.assertEqual(planner.model, "gpt-4-vision-preview")
            self.assertEqual(planner.api_base, "https://api.openai.com/v1")
            
        finally:
            os.unlink(config_path)
            
    def test_different_providers(self):
        """Test configuration for different providers."""
        configs = {
            "anthropic": {
                "api_base": "https://api.anthropic.com/v1",
                "model": "claude-3-opus-20240229"
            },
            "ollama": {
                "api_base": "http://localhost:11434/v1",
                "model": "llava:13b"
            },
            "together": {
                "api_base": "https://api.together.xyz/v1",
                "model": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
            }
        }
        
        from recyclobot.planning.openai_planner import OpenAIPlanner
        
        for provider, config in configs.items():
            with self.subTest(provider=provider):
                planner = OpenAIPlanner()
                planner.api_base = config["api_base"]
                planner.model = config["model"]
                planner.api_key = "test-key"
                
                self.assertEqual(planner.api_base, config["api_base"])
                self.assertEqual(planner.model, config["model"])
                
    @patch('recyclobot.planning.openai_planner.requests.post')
    def test_api_error_handling(self, mock_post):
        """Test handling of API errors."""
        # Mock API error
        mock_post.side_effect = Exception("Connection error")
        
        from recyclobot.planning.openai_planner import OpenAIPlanner
        planner = OpenAIPlanner()
        planner.api_key = "test-key"
        
        # Should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            planner.plan(self.test_image, self.test_prompt)
            
        self.assertIn("Planning failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()