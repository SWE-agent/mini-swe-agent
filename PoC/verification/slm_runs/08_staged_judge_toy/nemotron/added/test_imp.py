import sys
from pathlib import Path

# Add observatory to path
_OBSERVATORY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_OBSERVATORY_DIR))

# Step 1: Import observatory_log
print("After import observatory_log:")
print("  'observatory_log' in dir():", "observatory_log" in dir())
print("  'observatory_log' in sys.modules:", "observatory_log" in sys.modules)

# Step 2: Import the names
print("\nAfter from observatory_log import ...")
print("  'observatory_log' in dir():", "observatory_log" in dir())

# Step 3: Try importing agent
try:
    print("\nAgent imported successfully")
except Exception as e:
    print("\nAgent import failed:", type(e).__name__, e)
    print("  This is expected because agent.py imports anthropic")

# Step 4: Check namespace after agent import
print("\nAfter agent import attempt:")
print("  'observatory_log' in dir():", "observatory_log" in dir())
