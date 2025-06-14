"""
RecycloBot Qwen-VL Fallback Planner
Local vision-language model for recycling task planning.
Uses Qwen-VL-Chat quantized to 4-bit for efficiency.
"""

import json
import re
from typing import List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

# Model configuration
_MODEL_NAME = "Qwen/Qwen-VL-Chat"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy loading for model and processor
_processor = None
_model = None


def _load_model():
    """Lazy load the model and processor."""
    global _processor, _model
    
    if _processor is None:
        print("Loading Qwen-VL processor...")
        _processor = AutoProcessor.from_pretrained(_MODEL_NAME)
    
    if _model is None:
        print("Loading Qwen-VL model (4-bit quantized)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            device_map="auto",
            quantization_config=quantization_config if _DEVICE == "cuda" else None,
            torch_dtype=torch.float16 if _DEVICE == "cuda" else torch.float32
        )
        _model.eval()


# Recycling-focused prompt
_RECYCLING_PROMPT = """You are RecycloBot planner. Analyze the image and generate recycling skills.

Skills (SmolVLA pre-trained):
- pick(object): Pick item (uses visual context)
- place(bin): Place in bin (recycling_bin/compost_bin/trash_bin)
- highfive(): Completion gesture

For complex tasks, use sequences: inspect → ["pick(item)", "place(inspection_zone)", "pick(item)", "place(bin)"]

Categories:
- Recycling: bottles, cans, paper
- Compost: food, organic waste
- Trash: non-recyclables

User request: {user}

Output JSON array only:
["pick(item)", "place(bin)", ...]"""


def plan(image: Image.Image, user_prompt: str) -> List[str]:
    """
    Generate recycling skill sequence using Qwen-VL.
    
    Args:
        image: PIL Image of workspace
        user_prompt: User instruction
    
    Returns:
        List of skill strings
    """
    # Load model if needed
    _load_model()
    
    try:
        # Prepare inputs
        prompt = _RECYCLING_PROMPT.format(user=user_prompt)
        inputs = _processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Move to device
        if _DEVICE == "cuda":
            inputs = {k: v.to(_DEVICE) if hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False,
                pad_token_id=_processor.tokenizer.pad_token_id,
                eos_token_id=_processor.tokenizer.eos_token_id
            )
        
        # Decode output
        text = _processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        # Qwen often includes the prompt in output, so find the JSON part
        json_match = re.search(r'\[.*?\]', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                skills = json.loads(json_str)
                if isinstance(skills, list) and all(isinstance(s, str) for s in skills):
                    return skills
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to find skills manually
        skills = []
        skill_pattern = r'(pick|place|inspect|sort)\([^)]*\)'
        matches = re.findall(skill_pattern, text)
        for match in matches:
            skills.append(match)
        
        if skills:
            return skills
        
        # Last resort: return default sorting sequence
        print(f"Warning: Could not parse Qwen response: {text}")
        return ["inspect(items)", "sort()"]
        
    except Exception as e:
        print(f"Qwen inference error: {e}")
        # Return safe default
        return ["inspect(items)", "sort()"]