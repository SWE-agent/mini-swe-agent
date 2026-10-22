"""Debug import namespace"""
import sys
from pathlib import Path

# Add observatory to path
_OBSERVATORY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_OBSERVATORY_DIR))

# Import observability_log module FIRST - this ensures the name 'observability_log'
# is in sys.modules and will be available when llm.py does "from observability_log import ..."
# AND when it references "observability_log.llm_query(...)"
import observatory_log

print("After import observatory_log:")
print("  'observatory_log' in dir():", 'observatory_log' in dir())
print("  'observatory_log' in sys.modules:", 'observatory_log' in sys.modules)

# Now stub anthropic
_mock_anthropic = type(sys)("anthropic")
_mock_anthropic.Anthropic = type("AnthropicClient", (), {})
sys.modules["anthropic"] = _mock_anthropic

# Now import agent (which imports llm.py)
print("\nAbout to import agent...")
from agent import ToyAgent
print("Agent imported successfully")

# Check namespace
print("\nAfter agent import:")
print("  'observatory_log' in dir():", 'observatory_log' in dir())
print("  'observatory_log' in sys.modules:", 'observatory_log' in sys.modules)

# Try calling observability_log.llm_query
try:
    result = observatory_log.llm_query(1, model="m", messages=[])
    print("\nSUCCESS: observatory_log.llm_query works!", result)
except NameError as e:
    print("\nNameError:", e)
except Exception as e:
    print("\nOther error:", type(e).__name__, e)
