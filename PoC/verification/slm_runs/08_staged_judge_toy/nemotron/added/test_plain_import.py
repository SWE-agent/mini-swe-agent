import sys

# Step 1: Pre-register observatory_log in sys.modules BEFORE other imports
_log_mod = type(sys)("observatory_log")
_log_mod.llm_query = lambda step, model=None, messages=None: "llm_query called from stub"
_log_mod.llm_response = lambda step, actions=None: "llm_response called"
sys.modules["observability_log"] = _log_mod

print("Step 1 - Pre-registered:")
print("  'observability_log' in sys.modules:", 'observability_log' in sys.modules)

# Step 2: plain import observatory_log (NOT as Y)
import observatory_log
print("\nStep 2 - After 'import observatory_log':")
print("  'observatory_log' in dir():", 'observatory_log' in dir())
print("  'observatory_log' in sys.modules:", 'observatory_log' in sys.modules)

# Step 2b: also check the original module
print("  _log_mod is observatory_log:", _log_mod is observatory_log)

# Step 3: from observatory_log import llm_query
from observatory_log import llm_query
print("\nStep 3 - After 'from observatory_log import llm_query':")
print("  'observatory_log' in dir():", 'observatory_log' in dir())

# Step 4: Can we access observatory_log.llm_query?
try:
    result = observatory_log.llm_query(1, model="m", messages=[])
    print("\nStep 4 - SUCCESS: observatory_log.llm_query works:", result)
except NameError as e:
    print("\nStep 4 - NameError:", e)
