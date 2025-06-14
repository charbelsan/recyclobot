"""
RecycloBot Gemini Vision Planner
Analyzes images to generate recycling/sorting skill sequences.
Returns JSON array of skills like ["pick(plastic_bottle)", "place(recycling_bin)"]
"""

import base64
import io
import json
import os
from typing import List

from PIL import Image

# Check for API key
_API_KEY = os.getenv("GEMINI_API_KEY")
if _API_KEY is None:
    raise RuntimeError("GEMINI_API_KEY not set - will use Qwen fallback")

import google.generativeai as genai

genai.configure(api_key=_API_KEY)
_MODEL = genai.GenerativeModel("gemini-1.5-flash-latest")

# Recycling-focused prompt
_RECYCLING_PROMPT = """
You are RecycloBot, a high-level planner for a 6-DoF robot arm that sorts waste.

Available skills (pre-trained in SmolVLA):
- pick(object): Pick up a specific item (e.g., "pick(plastic_bottle)", "pick(aluminum_can)")
- place(bin): Place item in a specific bin (e.g., "place(recycling_bin)", "place(compost_bin)", "place(trash_bin)")
- highfive(): Gesture to signal completion

Note: SmolVLA uses visual context to identify objects. The object name in pick() helps planning but the actual targeting is done visually.
For inspection tasks, use a sequence like: ["pick(item)", "place(inspection_zone)", "pick(item)", "place(correct_bin)"]

Common waste categories:
- Recycling: plastic bottles, aluminum cans, glass bottles, paper, cardboard
- Compost: food waste, organic materials
- Trash: non-recyclable plastics, mixed materials

Given an image of a desk/table with waste items and a user request, respond ONLY with a JSON array of skills.
No explanations, no additional text.

### User request
{user}

### Expected output format
["pick(plastic_bottle)", "place(recycling_bin)", "pick(banana_peel)", "place(compost_bin)"]
"""


def _encode_image(img: Image.Image) -> str:
    """Encode PIL image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def plan(image: Image.Image, user_prompt: str) -> List[str]:
    """
    Generate a sequence of recycling skills from an image and prompt.
    
    Args:
        image: PIL Image of the workspace
        user_prompt: User's instruction (e.g., "Sort the trash")
    
    Returns:
        List of skill strings like ["pick(bottle)", "place(recycling_bin)"]
    """
    try:
        # Generate content with Gemini
        response = _MODEL.generate_content(
            [
                _RECYCLING_PROMPT.format(user=user_prompt),
                image
            ],
            generation_config={
                "temperature": 0.1,  # Low temperature for consistent outputs
                "max_output_tokens": 256,
            }
        )
        
        # Extract text from response
        text = response.text.strip()
        
        # Try to parse as JSON
        try:
            skills = json.loads(text)
            if isinstance(skills, list) and all(isinstance(s, str) for s in skills):
                return skills
        except json.JSONDecodeError:
            # Gemini sometimes adds extra text, try to extract JSON array
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx + 1]
                skills = json.loads(json_str)
                if isinstance(skills, list):
                    return skills
        
        # Fallback: return empty list if parsing fails
        print(f"Warning: Could not parse Gemini response: {text}")
        return []
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        raise RuntimeError(f"Gemini planning failed: {e}")