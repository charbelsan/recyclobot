# RecycloBot: Gemini/Qwen planner + dataset logger

## Summary

This PR adds RecycloBot, a vision-language planning system for robotic waste sorting to LeRobot. It enables robots to analyze cluttered workspaces and autonomously sort items into recycling, compost, and trash bins.

### Key Features

- **🤖 Vision-Language Planning**: Gemini (cloud) or Qwen-VL (local) analyze scenes and generate skill sequences
- **🗣️ Natural Language Control**: SmolVLA executes instructions like "pick up the plastic bottle"
- **📊 Enhanced Dataset Logging**: Records demonstrations with planning metadata
- **♻️ Recycling Focus**: Domain-specific vocabulary for waste sorting

### How SmolVLA Integration Works

SmolVLA is a Vision-Language-Action model that:
- Takes RGB images + natural language instructions
- Uses language to specify WHAT to manipulate ("the red block" vs "the blue block")
- Outputs continuous robot actions
- No goal IDs - everything is natural language!

### Implementation

**New Modules:**
- `recyclobot/planning/gemini_planner.py` - Gemini-1.5 vision planner
- `recyclobot/planning/qwen_planner.py` - Qwen-VL local fallback
- `recyclobot/control/skill_runner.py` - Skill to action mapping
- `recyclobot/logging/dataset_logger.py` - HuggingFace dataset logger
- `examples/run_recyclobot_demo.py` - CLI demo script

**Tests:**
- `tests/test_planner_json.py` - Validates planner JSON output
- `tests/test_logger_roundtrip.py` - Tests dataset save/load

**Documentation:**
- `README_RECYCLOBOT.md` - Quick start guide
- `docs/architecture.md` - System design details

## Usage

```bash
# Set up (optional for Gemini)
export GEMINI_API_KEY="your-key"

# Run demo
python examples/run_recyclobot_demo.py --prompt "Sort the trash"

# With real robot
python examples/run_recyclobot_demo.py --robot so101 --prompt "Clean workspace"
```

## Example Output

```
Planning with gemini...
Generated plan: ["pick(plastic_bottle)", "place(recycling_bin)", "pick(can)", "place(recycling_bin)"]

[1/4] pick(plastic_bottle)
Executing: pick(plastic_bottle) -> Goal ID: 0, Prompt: 'pick up the plastic bottle'
...
```

## Dataset Format

Extends LeRobot datasets with planning metadata:
- `planner_name`: Which VLM was used
- `planner_log`: Full skill sequence  
- `current_skill`: Currently executing skill
- `goal_id`: SmolVLA goal identifier
- `language_prompt`: Natural language instruction

## Testing

```bash
pytest tests/test_planner_json.py -v
pytest tests/test_logger_roundtrip.py -v
```

## Demo

🎥 [Video Demo](https://youtube.com/recyclobot-demo)
📊 [Example Dataset](https://huggingface.co/datasets/recyclobot/waste-sorting-demos)

## Future Work

- [ ] Closed-loop replanning based on execution
- [ ] Learning new skills from demonstrations
- [ ] Multi-robot coordination
- [ ] Active perception for better sorting

---

**RecycloBot** - Making robots environmentally conscious! 🌍♻️🤖