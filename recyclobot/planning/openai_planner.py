"""
RecycloBot OpenAI-style API Planner
Supports any OpenAI-compatible API (OpenAI, Anthropic, local models via vLLM/Ollama, etc.)
"""

import base64
import io
import json
import os
from typing import List, Optional

import requests
import yaml
from PIL import Image


class OpenAIPlanner:
    """Planner using OpenAI-compatible APIs for vision-language tasks."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize planner with configuration.
        
        Args:
            config_path: Path to config file. If None, uses env vars.
        """
        self.config = self._load_config(config_path)
        self.api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.api_base = self.config.get("api_base", "https://api.openai.com/v1")
        self.model = self.config.get("model", "gpt-4-vision-preview")
        
        if not self.api_key:
            raise ValueError("No API key found. Set OPENAI_API_KEY or use config file.")
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML or JSON file if provided."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        return {}
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def plan(self, image: Image.Image, user_prompt: str) -> List[str]:
        """
        Generate recycling skill sequence using OpenAI-compatible API.
        
        Args:
            image: PIL Image of workspace
            user_prompt: User instruction
            
        Returns:
            List of skill strings
        """
        # Prepare the prompt
        system_prompt = """You are RecycloBot, a high-level planner for a 6-DoF robot arm that sorts waste.

Available skills:
- pick(object): Pick up a specific item (e.g., "pick(plastic_bottle)", "pick(aluminum_can)")
- place(bin): Place item in a specific bin (e.g., "place(recycling_bin)", "place(compost_bin)", "place(trash_bin)")
- inspect(object): Examine an item more closely to determine its type
- sort(): General sorting command when multiple items need organizing

Common waste categories:
- Recycling: plastic bottles, aluminum cans, glass bottles, paper, cardboard
- Compost: food waste, organic materials
- Trash: non-recyclable plastics, mixed materials

Given an image and user request, respond ONLY with a JSON array of skills.
No explanations, no additional text."""

        # Prepare the messages
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"User request: {user_prompt}\n\nAnalyze the image and provide the skill sequence."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self._encode_image(image)}"
                        }
                    }
                ]
            }
        ]
        
        # Prepare the request
        request_data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.1
        }
        
        try:
            # Make API request
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=request_data,
                timeout=30
            )
            response.raise_for_status()
            
            # Extract response
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Try to parse JSON
            try:
                skills = json.loads(content)
                if isinstance(skills, list) and all(isinstance(s, str) for s in skills):
                    return skills
            except json.JSONDecodeError:
                # Try to extract JSON from response
                start = content.find('[')
                end = content.rfind(']')
                if start != -1 and end != -1:
                    skills = json.loads(content[start:end + 1])
                    if isinstance(skills, list):
                        return skills
            
            print(f"Warning: Could not parse response: {content}")
            return ["inspect(items)", "sort()"]
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            raise RuntimeError(f"OpenAI API request failed: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise RuntimeError(f"Planning failed: {e}")


# Convenience function for compatibility
def plan(image: Image.Image, user_prompt: str, config_path: Optional[str] = None) -> List[str]:
    """
    Plan using OpenAI-compatible API.
    
    Args:
        image: PIL Image
        user_prompt: User instruction
        config_path: Optional path to config file
        
    Returns:
        List of skills
    """
    planner = OpenAIPlanner(config_path)
    return planner.plan(image, user_prompt)