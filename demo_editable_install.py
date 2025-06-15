#!/usr/bin/env python
"""Demonstrate how editable install works"""

print("1. Before pip install -e .")
try:
    import recyclobot
    print(f"   ✓ recyclobot is already installed at: {recyclobot.__file__}")
except ImportError:
    print("   ✗ Cannot import recyclobot (not installed)")

print("\n2. What pip install -e . does:")
print("   - Reads setup.py to understand your package")
print("   - Creates a .egg-link file pointing to your code")
print("   - Adds your package to Python's path")
print("   - Installs dependencies from install_requires")

print("\n3. After pip install -e .")
print("   Now you can do:")
print("   >>> from recyclobot.planning.gemini_planner import plan")
print("   >>> from recyclobot.control.skill_runner import SkillRunner")
print("   From ANYWHERE on your system!")

print("\n4. The magic of -e (editable):")
print("   - Edit any file in recyclobot/")
print("   - Changes take effect immediately")
print("   - No need to reinstall")
print("   - Perfect for development!")

print("\n5. Where Python looks:")
import sys
for path in sys.path:
    if 'recyclobot' in path:
        print(f"   {path}")