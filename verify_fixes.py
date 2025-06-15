#!/usr/bin/env python
"""Verify all critical fixes are properly implemented."""

import sys
import os
from pathlib import Path

print("🔍 RecycloBot Fix Verification")
print("=" * 60)

# Track results
results = []

# 1. Check vendored lerobot is removed
print("\n1. Checking vendored LeRobot removal...")
lerobot_dir = Path("lerobot/")
if lerobot_dir.exists() and lerobot_dir.is_dir():
    results.append(("❌", "Vendored lerobot/ directory still exists!"))
else:
    results.append(("✅", "Vendored lerobot/ directory removed"))

# 2. Check packaging files exist
print("\n2. Checking packaging files...")
for file in ["MANIFEST.in", "pyproject.toml", "LICENSE"]:
    if Path(file).exists():
        results.append(("✅", f"{file} exists"))
    else:
        results.append(("❌", f"{file} missing!"))

# 3. Check state adapter module
print("\n3. Checking state adapter...")
try:
    from recyclobot.control.adapters import pad_state, adapt_observation_for_policy
    # Test basic functionality
    import numpy as np
    state = pad_state(np.zeros(6))
    if state.shape == (14,):
        results.append(("✅", "State adapter works (6→14 dimensions)"))
    else:
        results.append(("❌", f"State adapter wrong shape: {state.shape}"))
except Exception as e:
    results.append(("❌", f"State adapter error: {e}"))

# 4. Check mock environment dimensions
print("\n4. Checking mock environment...")
try:
    with open("examples/run_recyclobot_demo.py", "r") as f:
        content = f.read()
        if "np.random.randn(14)" in content:
            results.append(("✅", "Mock environment uses 14 dimensions"))
        else:
            results.append(("⚠️", "Mock environment may not use 14 dimensions"))
except Exception as e:
    results.append(("❌", f"Could not check mock environment: {e}"))

# 5. Check CI configuration
print("\n5. Checking CI configuration...")
ci_path = Path(".github/workflows/test.yml")
if ci_path.exists():
    with open(ci_path, "r") as f:
        content = f.read()
        if "No CPU tests were collected!" in content:
            results.append(("✅", "CI fails on zero tests"))
        else:
            results.append(("⚠️", "CI may not fail on zero tests"))
else:
    results.append(("❌", "CI configuration missing"))

# 6. Check pre-commit hooks
print("\n6. Checking pre-commit configuration...")
if Path(".pre-commit-config.yaml").exists():
    results.append(("✅", "Pre-commit hooks configured"))
else:
    results.append(("⚠️", "Pre-commit hooks not configured"))

# 7. Check LeRobot version in requirements
print("\n7. Checking LeRobot version...")
for file in ["requirements.txt", "setup.py", "pyproject.toml"]:
    if Path(file).exists():
        with open(file, "r") as f:
            content = f.read()
            if "@main" in content and "lerobot" in content.lower():
                results.append(("✅", f"{file} uses LeRobot@main"))
            elif "@v0.4.0" in content:
                results.append(("❌", f"{file} still uses v0.4.0!"))

# 8. Check .gitignore updates
print("\n8. Checking .gitignore...")
if Path(".gitignore").exists():
    with open(".gitignore", "r") as f:
        content = f.read()
        if "*.rec" in content and "*.parquet" in content:
            results.append(("✅", ".gitignore includes robot recordings"))
        else:
            results.append(("⚠️", ".gitignore missing robot recording patterns"))

# Summary
print("\n" + "=" * 60)
print("📊 VERIFICATION SUMMARY")
print("=" * 60)

passed = sum(1 for status, _ in results if status == "✅")
warnings = sum(1 for status, _ in results if status == "⚠️")
failed = sum(1 for status, _ in results if status == "❌")

for status, message in results:
    print(f"{status} {message}")

print("\n" + "-" * 60)
print(f"Total: {len(results)} checks")
print(f"✅ Passed: {passed}")
print(f"⚠️  Warnings: {warnings}")
print(f"❌ Failed: {failed}")

if failed == 0:
    print("\n🎉 All critical fixes verified successfully!")
    sys.exit(0)
else:
    print(f"\n⚠️  {failed} critical issues need attention!")
    sys.exit(1)