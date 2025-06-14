"""
Test RecycloBot planner JSON output validation
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


class TestPlannerJSON(unittest.TestCase):
    """Test planner JSON output validation."""
    
    def setUp(self):
        """Create test image."""
        self.test_image = Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        self.test_prompt = "Sort the trash"
        
    def test_gemini_planner_output_format(self):
        """Test Gemini planner returns valid JSON list."""
        # Mock the Gemini API
        with patch('google.generativeai.GenerativeModel') as mock_model:
            # Set up mock response
            mock_response = MagicMock()
            mock_response.text = '["pick(plastic_bottle)", "place(recycling_bin)"]'
            mock_model.return_value.generate_content.return_value = mock_response
            
            # Import after patching
            from recyclobot.planning.gemini_planner import plan
            
            # Test planning
            skills = plan(self.test_image, self.test_prompt)
            
            # Validate output
            self.assertIsInstance(skills, list)
            self.assertTrue(all(isinstance(s, str) for s in skills))
            self.assertEqual(len(skills), 2)
            self.assertEqual(skills[0], "pick(plastic_bottle)")
            self.assertEqual(skills[1], "place(recycling_bin)")
            
    def test_gemini_planner_handles_extra_text(self):
        """Test Gemini planner handles responses with extra text."""
        with patch('google.generativeai.GenerativeModel') as mock_model:
            # Response with extra text
            mock_response = MagicMock()
            mock_response.text = '''Here's the plan:
            ["pick(can)", "place(recycling_bin)", "pick(banana)", "place(compost_bin)"]
            This will sort the items correctly.'''
            mock_model.return_value.generate_content.return_value = mock_response
            
            from recyclobot.planning.gemini_planner import plan
            skills = plan(self.test_image, self.test_prompt)
            
            # Should extract JSON correctly
            self.assertEqual(len(skills), 4)
            self.assertIn("pick(can)", skills)
            self.assertIn("place(compost_bin)", skills)
            
    def test_qwen_planner_output_format(self):
        """Test Qwen planner returns valid JSON list."""
        # Mock the model loading
        with patch('transformers.AutoProcessor.from_pretrained') as mock_proc, \
             patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            
            # Set up mocks
            mock_processor = MagicMock()
            mock_proc.return_value = mock_processor
            
            mock_llm = MagicMock()
            mock_model.return_value = mock_llm
            
            # Mock tokenizer
            mock_processor.tokenizer.pad_token_id = 0
            mock_processor.tokenizer.eos_token_id = 1
            
            # Mock generate output
            mock_output = MagicMock()
            mock_llm.generate.return_value = mock_output
            
            # Mock decode
            mock_processor.decode.return_value = '["inspect(items)", "sort()"]'
            
            from recyclobot.planning.qwen_planner import plan
            skills = plan(self.test_image, self.test_prompt)
            
            # Validate
            self.assertIsInstance(skills, list)
            self.assertEqual(len(skills), 2)
            self.assertEqual(skills[0], "inspect(items)")
            self.assertEqual(skills[1], "sort()")
            
    def test_skill_format_validation(self):
        """Test that skills follow expected format."""
        valid_skills = [
            "pick(bottle)",
            "place(recycling_bin)",
            "inspect(unknown_item)",
            "sort()",
            "pick(aluminum_can)",
            "place(compost_bin)"
        ]
        
        skill_pattern = r'^(pick|place|inspect|sort)\([^)]*\)$'
        import re
        
        for skill in valid_skills:
            with self.subTest(skill=skill):
                self.assertIsNotNone(
                    re.match(skill_pattern, skill),
                    f"Skill '{skill}' doesn't match expected format"
                )
                
    def test_planner_error_handling(self):
        """Test planner handles errors gracefully."""
        with patch('google.generativeai.GenerativeModel') as mock_model:
            # Simulate API error
            mock_model.return_value.generate_content.side_effect = Exception("API Error")
            
            from recyclobot.planning.gemini_planner import plan
            
            # Should raise RuntimeError
            with self.assertRaises(RuntimeError):
                plan(self.test_image, self.test_prompt)
                
                
if __name__ == "__main__":
    unittest.main()