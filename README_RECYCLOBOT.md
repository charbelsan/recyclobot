# RecycloBot 🤖♻️

**Vision-Language Planning for Robotic Waste Sorting**

RecycloBot adds intelligent waste sorting capabilities to [LeRobot](https://github.com/huggingface/lerobot) using vision-language models for high-level planning. It combines:

- **Vision-Language Planning**: Gemini/Qwen analyze scenes and generate sorting sequences
- **SmolVLA Execution**: A vision-language-action model that understands natural language instructions like "pick up the plastic bottle" and executes robot actions
- **Semantic Understanding**: The system knows WHAT to pick (via language) and HOW to pick it (via vision)

![RecycloBot Demo](docs/recyclobot_demo.gif)

## 🎯 Features

- **Vision-Language Planning**: Multiple planner options:
  - Google Gemini (cloud)
  - Qwen-VL (local)
  - OpenAI GPT-4V
  - Anthropic Claude
  - Any OpenAI-compatible API (Ollama, vLLM, Together.ai, etc.)
- **Skill-Based Control**: Maps high-level skills to low-level robot actions via SmolVLA
- **Dataset Logging**: Records demonstrations with planning metadata for training
- **Multi-Robot Support**: Works in simulation and on real SO-ARM100 hardware
- **Extensible Framework**: Easy to add new skills and planners

## 🚀 Mode d'emploi Express

*Brancher la caméra, configurer le bras SO-ARM100, installer SmolVLA et commencer à enregistrer des démonstrations*

### 1. Préparer l'environnement Python

```bash
# Nouveau venv ou conda
conda create -n recyclobot python=3.10 -y
conda activate recyclobot

# Installer LeRobot avec support SmolVLA et SO-101
pip install "lerobot[smolvla,feetech]==0.5.0"  # Version avec SmolVLA support

# Télécharger les poids SmolVLA (IMPORTANT!)
huggingface-cli download lerobot/smolvla_base --local-dir ~/.cache/lerobot
# ou pour un modèle pré-entraîné sur des tâches de manipulation:
huggingface-cli download lerobot/koch_aloha --local-dir ~/.cache/lerobot

# Si vous voulez Qwen planner local
pip install "transformers[vision]>=4.44.0" "bitsandbytes>=0.41.0" "accelerate>=0.26.0"

# Gemini (optionnel)
pip install google-generativeai

# Clone RecycloBot
git clone https://github.com/your-username/recyclobot.git
cd recyclobot

# Installer RecycloBot en mode dev
pip install -e .
```

### 2. Connecter et détecter la webcam

1. **Brancher** la webcam USB directement sur le PC (éviter les hub USB non alimentés)
2. **Repérer le device**:
   ```bash
   v4l2-ctl --list-devices          # liste /dev/video0, /dev/video1...
   ```
3. **Tester** que 640×480 @30 fps fonctionne:
   ```bash
   ffplay -f v4l2 -video_size 640x480 -i /dev/video0
   ```
4. **Option LeRobot** - enregistrer une image de test:
   ```bash
   python - <<'PY'
   import cv2, pathlib
   cam=cv2.VideoCapture(0); ok,fram=cam.read()
   pathlib.Path("test.jpg").write_bytes(cv2.imencode(".jpg",fram)[1])
   print("image écrite test.jpg")
   PY
   ```

### 3. Assembler et calibrer le SO-ARM100

1. **Impression/montage mécanique** selon le PDF Seeed: vérifier que les 12 servos sont câblés en daisy-chain (UART)
2. **Alimentation**: toujours brancher le 5V 3A *avant* l'USB-C
3. **Calibrage zéro**:
   ```bash
   python -m lerobot.scripts.control_robot \
      --robot.type so101 \
      --control.type calibrate
   ```
   Suivre l'assistant (mettre chaque articulation sur la marque, puis `Enter`)
4. **Sauvegarder** le fichier `~/.lerobot/so101_calib.json`

### 4. Tester le télé-op clavier

```bash
python -m lerobot.scripts.control_robot \
   --robot.type so101 \
   --control.type teleoperate \
   --robot.camera_indices "[0]"
```

* Flèches: déplacer l'end-effector
* `PgUp/PgDn`: ouvrir/fermer la pince
* `Esc`: quitter

### 5. Enregistrer un jeu de démonstrations

```bash
huggingface-cli login                 # si vous voulez pousser ensuite
python -m lerobot.scripts.control_robot \
   --robot.type so101 \
   --control.type record \
   --control.single_task "Trier plastique et papier" \
   --control.num_episodes 5 \
   --control.fps 25 \
   --robot.camera_indices "[0]" \
   --control.output_dir data_run1 \
   --control.display_data true
```

* Chaque episode dure jusqu'à `Esc`
* Les fichiers générés:
  ```
  data_run1/
     info.json        # schéma + splits
     chunk-000.parquet
     videos/episode_000.mp4
  ```

### 6. Utiliser SmolVLA + planner (boucle autonome)

```bash
# Option 1: Gemini (Google)
export GEMINI_API_KEY="votre_clé"
python examples/run_recyclobot_demo.py --robot so101 --prompt "Trie les déchets"

# Option 2: OpenAI GPT-4V
export OPENAI_API_KEY="votre_clé" 
python examples/run_recyclobot_demo.py --robot so101 --planner openai --prompt "Trie les déchets"

# Option 3: Local avec Ollama
python examples/run_recyclobot_demo.py --robot so101 --planner ollama --config recyclobot/config.yaml

# Option 4: Anthropic Claude
python examples/run_recyclobot_demo.py --robot so101 --planner anthropic --config recyclobot/config.yaml
```

Le script:
* Capture une image
* Appelle `gemini_planner.plan()` ou `qwen_planner.plan()`
* Mappe chaque skill vers l'ID attendu par SmolVLA
* Passe l'image + l'ID au policy
* Enregistre tout dans `recyclobot_data/`

### 7. Publier le dataset

```bash
huggingface-cli repo create recyclobot-desk --type dataset -y
huggingface-cli upload --repo-type dataset \
    recyclobot-desk recyclobot_data/* -y
```

### 8. Collecte de Dataset au Format LeRobot (IMPORTANT!)

Pour fine-tuner SmolVLA sur vos propres tâches de recyclage:

```bash
# Collecte téléopérée avec annotations de planning
python scripts/collect_recyclobot_dataset.py \
    --robot-type so101 \
    --repo-id your-username/recyclobot-demos \
    --num-episodes 50 \
    --tasks-file recycling_tasks.json

# Collecte autonome avec planner
python scripts/collect_recyclobot_dataset.py \
    --robot-type so101 \
    --repo-id your-username/recyclobot-demos \
    --autonomous \
    --planner gemini \
    --num-episodes 20
```

### 9. Fine-tuning SmolVLA pour RecycloBot

```bash
# Option 1: Script RecycloBot avec LoRA (efficace)
python scripts/train_recyclobot.py \
    --dataset-name your-username/recyclobot-demos \
    --output-dir outputs/recyclobot_smolvla \
    --use-lora \
    --num-epochs 10

# Option 2: Utiliser LeRobot natif
python -m lerobot.scripts.train \
    policy=smolvla \
    dataset_repo_id=your-username/recyclobot-demos \
    hydra.run.dir=outputs/train/recyclobot_smolvla \
    training.num_epochs=20 \
    training.batch_size=8 \
    policy.use_lora=true
```

### 10. Évaluation du Système

```bash
# Évaluer la précision du planning
python scripts/evaluate_recyclobot.py \
    --dataset your-username/recyclobot-test \
    --checkpoint outputs/recyclobot_smolvla \
    --mode planning

# Test sur robot réel
python scripts/evaluate_recyclobot.py \
    --robot so101 \
    --live-eval \
    --num-trials 20 \
    --mode full
```

## 📋 Commandes clés résumé

```bash
# Calibration première fois
python -m lerobot.scripts.control_robot --robot.type so101 --control.type calibrate

# Télé-op
python -m lerobot.scripts.control_robot --robot.type so101 --control.type teleoperate --robot.camera_indices "[0]"

# Record 5 épisodes
python -m lerobot.scripts.control_robot --robot.type so101 --control.type record --control.num_episodes 5 --control.output_dir data

# Démo autonome (fallback Qwen si pas de clé Gemini)
python examples/run_recyclobot_demo.py --robot so101 --prompt "Trie les déchets"
```

## 📁 Project Structure

RecycloBot is designed as a **standalone extension** to LeRobot:

```
your-workspace/
├── lerobot/          # LeRobot framework (pip installed)
└── recyclobot/       # This project
    ├── recyclobot/   # Core modules
    ├── examples/     # Demo scripts
    ├── tests/        # Unit tests
    └── docs/         # Documentation
```

### Pourquoi pas de submodule/fork?

- **`pip install lerobot==0.4.*`** assure la même version pour tout le monde
- **Pas de `git submodule`** → clonage simplifié pour le jury
- **Les wrappers** (`gemini_planner.py`, `qwen_planner.py`) sont ajoutés à votre dépôt
- **Après le hackathon**, un PR officiel vers LeRobot est trivial

## ⚙️ Configuration

### Using Config File

Edit `recyclobot/config.yaml` to configure planners:

```yaml
# RecycloBot Configuration
planners:
  openai:
    api_key: "sk-..."  # or use OPENAI_API_KEY env var
    api_base: "https://api.openai.com/v1"
    model: "gpt-4-vision-preview"
    
  ollama:
    api_key: "not-needed"  # Local models don't need keys
    api_base: "http://localhost:11434/v1" 
    model: "llava:13b"
    
  anthropic:
    api_key: "your-key"
    api_base: "https://api.anthropic.com/v1"
    model: "claude-3-opus-20240229"

default_planner: "openai"
```

### Supported Providers

| Provider | API Base | Models | Notes |
|----------|----------|---------|-------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4-vision-preview` | Best quality |
| Anthropic | `https://api.anthropic.com/v1` | `claude-3-opus-20240229` | Good reasoning |
| Ollama | `http://localhost:11434/v1` | `llava:13b`, `bakllava` | Local, free |
| vLLM | `http://localhost:8000/v1` | Any vision model | High performance |
| Together | `https://api.together.xyz/v1` | Llama-3.2-Vision | Fast inference |

## 📊 Example Output

```
RecycloBot Demo - Waste Sorting with Vision-Language Planning
============================================================
Robot: sim
Task: Sort all the trash into bins
Episodes: 1
Output: recyclobot_data
============================================================
Planning with gemini...
Generated plan: ["pick(plastic_bottle)", "place(recycling_bin)", "pick(aluminum_can)", "place(recycling_bin)", "pick(banana_peel)", "place(compost_bin)"]

[1/6] pick(plastic_bottle)
Executing: pick(plastic_bottle) -> Goal ID: 0, Prompt: 'pick up the plastic bottle'

[2/6] place(recycling_bin)
Executing: place(recycling_bin) -> Goal ID: 1, Prompt: 'place the object in the recycling bin (blue)'

...

Demo Complete!
============================================================
Total episodes: 1
Total steps: 247
Average episode length: 247.0 steps
Skills used: {'pick': 3, 'place': 3}

Dataset saved to: recyclobot_data
```

## 🏗️ Architecture

```
RecycloBot System
├── Vision-Language Planner
│   ├── Gemini-1.5 (Google Cloud)
│   ├── GPT-4V (OpenAI)
│   ├── Claude-3 (Anthropic)
│   ├── Qwen-VL (local)
│   └── Any OpenAI-compatible API
├── Skill Mapping Layer
│   └── Skills → Natural Language Instructions
├── Control Layer
│   └── SmolVLA Policy (vision-language-action)
└── Dataset Logger
    └── HuggingFace Datasets + Planning Metadata
```

### How It Works

1. **Planner** sees image → generates skills: `["pick(bottle)", "place(recycling_bin)"]`
2. **Skill Runner** converts to natural language: `"pick up the plastic bottle"`  
3. **SmolVLA** processes image + instruction → outputs robot actions
4. **Robot** executes continuous joint commands

### Natural Language Instructions

SmolVLA understands instructions like:
- "pick up the plastic bottle"
- "place the object in the recycling bin"
- "pick up the aluminum can"
- "give a high five"

The language specifies WHAT to manipulate, while vision shows WHERE it is!

## 📁 Dataset Format

RecycloBot extends LeRobot's dataset format with planning metadata:

```python
{
    # Standard robot data
    "episode_id": 1,
    "step_id": 42,
    "image": np.array(...),      # Camera observation
    "state": [0.1, -0.2, ...],   # Joint positions
    "action": [0.05, 0.0, ...],  # Joint commands
    
    # RecycloBot additions
    "planner_name": "gemini",
    "planner_log": '["pick(bottle)", "place(recycling_bin)"]',
    "current_skill": "pick(bottle)",
    "language_instruction": "pick up the plastic bottle"  # Natural language for SmolVLA
}
```

## 🚨 Troubleshooting

### Common Issues and Solutions

**1. SmolVLA Model Loading Fails**
```bash
# Error: "Could not load pretrained model"
# Solution: Download weights explicitly
huggingface-cli download lerobot/koch_aloha --local-dir ~/.cache/lerobot
```

**2. Camera Not Found**
```bash
# Error: "No image found in observation"
# Solution: Check camera index
python -m lerobot.find_cameras  # Lists available cameras
# Update robot config with correct index
```

**3. State Dimension Mismatch**
```bash
# Error: "Expected state dim 14 but got 7"
# Solution: SO-101 has 14 state dimensions (7 joints x 2 for pos+vel)
# Update your robot config if using different robot
```

**4. Planning Fails**
```bash
# Check API keys
echo $GEMINI_API_KEY
echo $OPENAI_API_KEY

# Test planner directly
python -c "from recyclobot.planning.gemini_planner import plan; print(plan(Image.new('RGB', (640,480)), 'test'))"
```

**5. Dataset Format Issues**
```bash
# Use LeRobot's native format
# Wrong: {"image": ..., "instruction": ...}
# Right: {"observation.images.top": ..., "task": ...}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Specific tests
pytest tests/test_planner_json.py -v
pytest tests/test_logger_roundtrip.py -v
```

## 📤 Uploading Datasets

```bash
# After recording demonstrations
cd recyclobot_data
huggingface-cli upload your-username/recyclobot-demos .
```

## 🎥 Demo Video

Watch RecycloBot in action: [YouTube](https://youtube.com/recyclobot-demo) | [HuggingFace](https://huggingface.co/datasets/recyclobot/demos)

## 🛠️ Development

### Code Style

```bash
# Format code
black -l 88 recyclobot/
isort recyclobot/

# Type checking (optional)
mypy recyclobot/
```

### Adding New Skills

1. Add skill to `RECYCLING_SKILLS` in `skill_runner.py`
2. Update `skill_to_language_prompt()` method
3. Add to planner prompts
4. Test with demo script

### Adding New Planners

1. Create new file in `recyclobot/planning/`
2. Implement `plan(image: PIL.Image, prompt: str) -> List[str]`
3. Update planner selection in demo script

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Training Custom Policies](docs/training.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🔧 Troubleshooting

### Camera not detected
```bash
# Check USB devices
lsusb | grep -i camera
# Try different index
python examples/run_recyclobot_demo.py --robot so101 --camera-index 1
```

### Robot not responding
```bash
# Check USB connection
ls /dev/ttyUSB* /dev/ttyACM*
# Verify power (5V before USB-C!)
# Re-run calibration
python -m lerobot.scripts.control_robot --robot.type so101 --control.type calibrate
```

### SmolVLA download fails
```bash
# Manual download
huggingface-cli download lerobot/smolvla_base --local-dir ~/.cache/huggingface
```

### Gemini API errors
```bash
# Check API key
echo $GEMINI_API_KEY
# Falls back to Qwen automatically
```

## 🙏 Acknowledgments

- Built on [LeRobot](https://github.com/huggingface/lerobot) by HuggingFace
- Uses [SmolVLA](https://huggingface.co/lerobot/smolvla_base) for control
- Powered by Google Gemini and Qwen-VL
- Hardware: [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) by Seeed Studio

## 📧 Contact

For questions or collaboration: recyclobot@example.com

---

**RecycloBot** - Making robots environmentally conscious, one sort at a time! 🌍