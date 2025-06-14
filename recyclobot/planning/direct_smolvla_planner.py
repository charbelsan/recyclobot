"""
Direct SmolVLA planning - skip the intermediate skill generation.

Instead of:
  Image → Planner → Skills → SmolVLA → Actions

We do:
  Image → SmolVLA (with task) → Actions

This leverages SmolVLA's built-in vision-language understanding.
"""

from typing import List
from PIL import Image


def plan(image: Image.Image, user_prompt: str) -> List[str]:
    """
    "Plan" by directly using the task description.
    
    Since SmolVLA already contains SmolVLM for vision-language understanding,
    we don't need a separate planning stage. We just return the task as-is.
    
    Args:
        image: PIL Image (not used here, but kept for API compatibility)
        user_prompt: Natural language task like "Sort all the recycling"
        
    Returns:
        List containing just the original task
    """
    # For direct execution, we return the high-level task
    # SmolVLA will handle the understanding and decomposition internally
    return [f"task:{user_prompt}"]


def plan_with_subtasks(image: Image.Image, user_prompt: str) -> List[str]:
    """
    Break down high-level task into subtasks that SmolVLA can execute sequentially.
    
    This is a simple heuristic-based approach that doesn't require another model.
    
    Args:
        image: PIL Image (could be analyzed with CV if needed)
        user_prompt: Natural language task
        
    Returns:
        List of subtask descriptions
    """
    prompt_lower = user_prompt.lower()
    
    # Heuristic breakdown based on common recycling tasks
    if "sort" in prompt_lower and ("all" in prompt_lower or "everything" in prompt_lower):
        # General sorting task
        return [
            "look for plastic bottles and pick them up",
            "place plastic bottles in the recycling bin",
            "look for aluminum cans and pick them up", 
            "place aluminum cans in the recycling bin",
            "look for organic waste and pick it up",
            "place organic waste in the compost bin"
        ]
    
    elif "plastic" in prompt_lower or "bottle" in prompt_lower:
        return [
            "pick up all plastic bottles",
            "place them in the recycling bin"
        ]
    
    elif "can" in prompt_lower or "aluminum" in prompt_lower:
        return [
            "pick up all aluminum cans",
            "place them in the recycling bin"
        ]
    
    elif "compost" in prompt_lower or "organic" in prompt_lower or "food" in prompt_lower:
        return [
            "pick up all organic waste",
            "place it in the compost bin"
        ]
    
    else:
        # Default: return the original task
        return [user_prompt]


# Alternative: Use skill notation but let SmolVLA interpret them as language
def plan_as_natural_skills(image: Image.Image, user_prompt: str) -> List[str]:
    """
    Generate natural language "skills" that read more like instructions.
    
    This bridges between structured skills and natural language.
    """
    prompt_lower = user_prompt.lower()
    skills = []
    
    # Detect what needs to be sorted
    if "plastic" in prompt_lower or "bottle" in prompt_lower:
        skills.extend([
            "pick(plastic_bottle)",
            "place(recycling_bin)"
        ])
    
    if "can" in prompt_lower or "aluminum" in prompt_lower:
        skills.extend([
            "pick(aluminum_can)", 
            "place(recycling_bin)"
        ])
    
    if "organic" in prompt_lower or "food" in prompt_lower or "compost" in prompt_lower:
        skills.extend([
            "pick(organic_waste)",
            "place(compost_bin)"
        ])
    
    # If no specific items mentioned, assume general sorting
    if not skills and ("sort" in prompt_lower or "clean" in prompt_lower):
        skills = [
            "pick(recyclable_items)",
            "place(recycling_bin)",
            "pick(organic_waste)",
            "place(compost_bin)"
        ]
    
    return skills if skills else ["pick(item)", "place(appropriate_bin)"]