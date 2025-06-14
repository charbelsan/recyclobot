#!/bin/bash
# Quick one-liner tests for RecycloBot base model

echo "🤖 RecycloBot Quick Tests"
echo "========================"

echo -e "\n1️⃣ Test planning (no GPU needed):"
python -c "
from recyclobot.planning.direct_smolvla_planner import plan
from PIL import Image
img = Image.new('RGB', (640, 480))
print('Input: Sort the recycling')
print('Output:', plan(img, 'Sort the recycling'))
"

echo -e "\n2️⃣ Test skill mapping:"
python -c "
from recyclobot.control.skill_runner import SkillRunner
runner = SkillRunner(None)
print('pick(plastic_bottle) →', runner.skill_to_language_prompt('pick(plastic_bottle)'))
print('place(recycling_bin) →', runner.skill_to_language_prompt('place(recycling_bin)'))
"

echo -e "\n3️⃣ Test model loading (requires GPU + weights):"
python -c "
try:
    from lerobot.common.policies.factory import make_policy
    policy = make_policy('smolvla', pretrained='lerobot/smolvla_base')
    print('✅ SmolVLA loaded successfully!')
except Exception as e:
    print('⚠️  Model loading failed (expected without weights):', str(e)[:50], '...')
    print('   Run: huggingface-cli download lerobot/smolvla_base')
"

echo -e "\n✅ Quick tests complete! For full test run: python test_base_model.py"