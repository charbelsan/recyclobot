# RecycloBot Project Structure

## Directory Layout

```
recyclobot/                    # Your RecycloBot project (this repo)
├── recyclobot/               # Core RecycloBot modules
│   ├── __init__.py
│   ├── planning/            # Vision-language planners
│   │   ├── gemini_planner.py
│   │   └── qwen_planner.py
│   ├── control/             # Skill execution
│   │   └── skill_runner.py
│   └── logging/             # Dataset recording
│       └── dataset_logger.py
├── examples/                # Demo scripts
│   └── run_recyclobot_demo.py
├── tests/                   # Unit tests
│   ├── test_planner_json.py
│   └── test_logger_roundtrip.py
├── docs/                    # Documentation
│   └── architecture.md
├── README.md               # This file
└── requirements.txt        # Python dependencies

lerobot/                    # LeRobot framework (separate repo)
├── lerobot/
│   ├── common/
│   │   ├── policies/      # Where SmolVLA lives
│   │   ├── robots/        # Robot interfaces
│   │   └── datasets/      # Dataset utilities
│   └── ...
└── ...
```

## How It Works

1. **RecycloBot is a standalone project** that uses LeRobot as a dependency
2. **LeRobot is installed via pip**: `pip install lerobot`
3. **SmolVLA is downloaded automatically** from HuggingFace when needed
4. **No need to modify LeRobot code** - RecycloBot extends it

## Key Design Decisions

### Why Standalone?

- **Clean separation**: RecycloBot features don't modify core LeRobot
- **Easy testing**: Can develop and test independently
- **Hackathon-friendly**: Judges can run without forking LeRobot
- **Future PR**: Can later be integrated into LeRobot if desired

### SmolVLA Integration

- Uses pre-trained weights only (no LoRA needed)
- Only 3 skills: pick (0), place (1), highfive (2)
- Object identification via visual features, not text
- Complex behaviors via skill sequences

## Running RecycloBot

```bash
# From recyclobot directory
python examples/run_recyclobot_demo.py --prompt "Sort the trash"
```

This will:
1. Import LeRobot (must be installed)
2. Download SmolVLA if needed (~2GB)
3. Run the recycling demo