"""
SmolVLM-based planner for RecycloBot using HuggingFace's SmolVLM model.

This planner uses SmolVLM to analyze scenes and generate recycling plans,
potentially enabling end-to-end vision-language-action planning.
"""

import logging
import os
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

logger = logging.getLogger(__name__)

# SmolVLM model configuration
DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Recycling-specific prompt
RECYCLING_PROMPT = """You are a recycling robot assistant. Analyze the image and generate a sequence of pick and place actions to sort items into appropriate bins.

Available bins:
- recycling_bin (blue): for plastic bottles, aluminum cans, glass, paper
- compost_bin (green): for food waste, organic materials
- trash_bin (black): for non-recyclable items

Generate actions in this format:
pick(item_name)
place(bin_name)

User request: {user}

Actions:"""


class SmolVLMPlanner:
    """Planner using SmolVLM for vision-language understanding."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize SmolVLM planner.
        
        Args:
            model_name: HuggingFace model name for SmolVLM
        """
        self.model_name = model_name
        self._model = None
        self._processor = None
        
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading SmolVLM model: {self.model_name}")
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self._model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    device_map="auto" if DEVICE == "cuda" else None,
                    trust_remote_code=True
                )
                if DEVICE == "cpu":
                    self._model = self._model.to(DEVICE)
                logger.info(f"SmolVLM loaded on {DEVICE}")
            except Exception as e:
                logger.error(f"Failed to load SmolVLM: {e}")
                raise RuntimeError(f"Cannot load SmolVLM model: {e}")
    
    def plan(self, image: Image.Image, user_prompt: str) -> List[str]:
        """Generate recycling plan using SmolVLM.
        
        Args:
            image: PIL Image of the scene
            user_prompt: Natural language task description
            
        Returns:
            List of skill strings like ["pick(bottle)", "place(recycling_bin)"]
        """
        # Load model if needed
        self._load_model()
        
        # Prepare prompt
        full_prompt = RECYCLING_PROMPT.format(user=user_prompt)
        
        # Process inputs
        inputs = self._processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self._model.device) if hasattr(v, 'to') else v 
                  for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode response
        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part (after the prompt)
        if "Actions:" in response:
            response = response.split("Actions:")[-1].strip()
        
        # Parse skills from response
        skills = self._parse_skills(response)
        
        return skills
    
    def _parse_skills(self, response: str) -> List[str]:
        """Parse skill list from model response.
        
        Args:
            response: Model's text response
            
        Returns:
            List of skill strings
        """
        skills = []
        
        # Split by newlines and process each line
        for line in response.strip().split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
                
            # Look for pick/place patterns
            if line.startswith(('pick(', 'place(', 'inspect(')):
                # Clean up the skill string
                skill = line.split('.')[0]  # Remove trailing periods
                skill = skill.strip()
                
                # Validate parentheses
                if '(' in skill and ')' in skill:
                    skills.append(skill)
            
            # Stop at natural end markers
            if any(marker in line.lower() for marker in ['done', 'complete', 'finished']):
                break
        
        # Fallback if no skills parsed
        if not skills:
            logger.warning("No skills parsed from SmolVLM response, using default")
            skills = ["pick(object)", "place(recycling_bin)"]
        
        return skills


# Module-level planner instance
_planner = None


def plan(image: Image.Image, user_prompt: str) -> List[str]:
    """Generate recycling plan using SmolVLM.
    
    This is the main entry point matching the interface of other planners.
    
    Args:
        image: PIL Image of the scene
        user_prompt: Natural language task description
        
    Returns:
        List of skill strings
    """
    global _planner
    
    if _planner is None:
        _planner = SmolVLMPlanner()
    
    return _planner.plan(image, user_prompt)


# Optional: End-to-end planning that directly generates language instructions
def plan_end_to_end(image: Image.Image, user_prompt: str) -> List[str]:
    """Generate natural language instructions directly for SmolVLA.
    
    This bypasses the skill abstraction and generates instructions like:
    - "pick up the plastic bottle"
    - "place it in the recycling bin"
    
    Args:
        image: PIL Image of the scene
        user_prompt: Natural language task description
        
    Returns:
        List of natural language instructions
    """
    global _planner
    
    if _planner is None:
        _planner = SmolVLMPlanner()
    
    # Use a different prompt for end-to-end generation
    e2e_prompt = """You are controlling a robot arm. Generate step-by-step natural language instructions to complete the task.

Each instruction should be a simple imperative sentence like:
- "pick up the plastic bottle"
- "place it in the recycling bin"
- "move to the aluminum can"

Task: {user}

Instructions:"""
    
    # Similar processing but with different prompt
    full_prompt = e2e_prompt.format(user=user_prompt)
    
    # Load model if needed
    _planner._load_model()
    
    # Process inputs
    inputs = _planner._processor(
        text=full_prompt,
        images=image,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(_planner._model.device) if hasattr(v, 'to') else v 
              for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = _planner._model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode response
    response = _planner._processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract instructions
    if "Instructions:" in response:
        response = response.split("Instructions:")[-1].strip()
    
    # Parse natural language instructions
    instructions = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            # Clean up instruction
            line = line.lstrip('- ').rstrip('.')
            if len(line) > 5:  # Basic validation
                instructions.append(line)
    
    return instructions if instructions else ["pick up the item", "place it in the appropriate bin"]


if __name__ == "__main__":
    # Test the planner
    test_image = Image.new('RGB', (640, 480), color='white')
    
    print("Testing SmolVLM planner...")
    skills = plan(test_image, "Sort the recycling")
    print(f"Generated skills: {skills}")
    
    print("\nTesting end-to-end planning...")
    instructions = plan_end_to_end(test_image, "Sort the recycling")
    print(f"Generated instructions: {instructions}")