# RecycloBot Architecture

## System Overview

RecycloBot implements a hierarchical control system for robotic waste sorting, combining vision-language models for high-level planning with low-level goal-conditioned policies for execution.

```
┌─────────────────────────────────────────────────────────────┐
│                         User Input                           │
│                    "Sort the trash"                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Camera Observation                         │
│                    (RGB Image)                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              Vision-Language Planner                         │
│         ┌─────────────┐     ┌─────────────┐                │
│         │   Gemini    │     │   Qwen-VL   │                │
│         │   (Cloud)   │     │   (Local)   │                │
│         └──────┬──────┘     └──────┬──────┘                │
│                └──────────┬─────────┘                       │
│                           ▼                                  │
│                    Skill Sequence                           │
│         ["pick(bottle)", "place(recycling_bin)"]           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Skill Runner                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Skill Parser: "pick(bottle)" → action="pick"      │    │
│  │                                  param="bottle"     │    │
│  └────────────────────────┬───────────────────────────┘    │
│  ┌────────────────────────▼───────────────────────────┐    │
│  │  Goal Mapping: pick → goal_id=0                    │    │
│  │                place → goal_id=1                   │    │
│  └────────────────────────┬───────────────────────────┘    │
│  ┌────────────────────────▼───────────────────────────┐    │
│  │  Language Generation: "pick up the plastic bottle"  │    │
│  └────────────────────────┬───────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  SmolVLA Policy                              │
│            (Goal-Conditioned Control)                        │
│  Input: observation + goal_id + language                    │
│  Output: joint velocities/torques                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     Robot Hardware                           │
│                    (SO-ARM100)                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Dataset Logger                             │
│         Records: obs, action, skills, metadata              │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Vision-Language Planner

The planner analyzes visual scenes and generates skill sequences:

**Gemini Planner** (`gemini_planner.py`):
- Uses Gemini-1.5-Flash for fast inference
- Requires API key (cloud-based)
- Structured prompts ensure JSON output
- Handles recycling domain knowledge

**Qwen Planner** (`qwen_planner.py`):
- Local fallback using Qwen-VL-Chat
- 4-bit quantization for efficiency
- Runs on GPU if available
- Same output format as Gemini

**Planning Process**:
1. Receive image + user prompt
2. Apply recycling-specific system prompt
3. Generate JSON skill array
4. Validate and parse output

### 2. Skill Runner

Maps high-level skills to robot control:

**Skill Vocabulary**:
```python
RECYCLING_SKILLS = {
    "pick": 0,      # Pick up object
    "place": 1,     # Place in container
    "inspect": 2,   # Examine object
    "sort": 3,      # General sorting
}
```

**Execution Flow**:
1. Parse skill string (e.g., "pick(bottle)")
2. Extract action and parameters
3. Map to SmolVLA goal ID
4. Generate natural language prompt
5. Execute with timeout
6. Monitor completion

### 3. SmolVLA Integration

SmolVLA is a Vision-Language-Action model that processes both visual observations and natural language instructions:

**Input**:
- Visual observation (RGB images from cameras)
- Natural language instruction (e.g., "pick up the plastic bottle")
- Proprioceptive state (optional - joint positions)

**Output**:
- Continuous robot actions (joint velocities/torques)
- 7-DoF control (6 joints + gripper)

**Key Architecture Details** (from HuggingFace blog):
- **Vision-Language backbone**: Uses SmolVLM-500M-Instruct for understanding
- **Language specifies targets**: "pick up the RED block" vs "pick up the BLUE block"
- **Cross-modal understanding**: Fuses visual and language information
- **Action expert**: Dedicated module for outputting robot actions
- **Pre-trained on diverse tasks**: Generalizes to new objects/instructions

**How it works**:
1. Image encoder processes camera input
2. Language encoder processes instruction
3. Cross-attention fuses visual and language features
4. Action expert generates robot commands

**Important**: The language instruction is crucial! It tells SmolVLA WHAT to manipulate, not just HOW (pick vs place).

### 4. Dataset Logger

Records demonstrations with planning metadata:

**Schema Extensions**:
```python
# Standard LeRobot fields
"episode_id", "step_id", "timestamp"
"image", "state", "action"

# RecycloBot additions
"planner_name"      # Which planner was used
"planner_log"       # Full skill sequence
"current_skill"     # Currently executing skill
"goal_id"           # SmolVLA goal
"language_prompt"   # Natural language instruction
"detected_objects"  # Objects in scene
"target_bin"        # Destination for placement
```

**Storage Format**:
- Parquet files for efficient storage
- HuggingFace datasets compatibility
- Episode-based organization
- Metadata JSON sidecar

## Data Flow

1. **Perception**: Camera captures workspace image
2. **Planning**: VLM analyzes scene and generates skill plan
3. **Execution**: Skills executed sequentially with SmolVLA
4. **Logging**: All data recorded with planning context
5. **Upload**: Dataset pushed to HuggingFace Hub

## Extension Points

### Adding New Skills

1. Add to `RECYCLING_SKILLS` mapping
2. Implement in `skill_to_language_prompt()`
3. Update planner prompts
4. Add to documentation

### Adding New Planners

1. Implement planner interface:
   ```python
   def plan(image: PIL.Image, prompt: str) -> List[str]
   ```
2. Add to planner selection logic
3. Ensure consistent output format

### Custom Robots

1. Implement LeRobot robot interface
2. Configure observation/action spaces
3. Calibrate SmolVLA for new embodiment
4. Update skill execution parameters

## Performance Considerations

- **Planner Latency**: Gemini ~500ms, Qwen ~2s
- **Control Frequency**: 10Hz default (configurable)
- **Skill Timeout**: 5s per skill (adjustable)
- **Dataset Size**: ~1GB per hour of demos

## Future Enhancements

1. **Closed-Loop Planning**: Re-plan based on execution
2. **Skill Learning**: Learn new skills from demos
3. **Multi-Robot Coordination**: Collaborative sorting
4. **Active Perception**: Strategic camera movements
5. **Failure Recovery**: Detect and recover from errors