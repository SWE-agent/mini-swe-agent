import sys

# Step 1: Put a mock observatory_log in sys.modules FIRST
_log_mod = type(sys)("observatory_log")
_log_mod.llm_query = lambda step, model=None, messages=None: "llm_query called"
_log_mod.llm_response = lambda step, actions=None: "llm_response called"
sys.modules["observatory_log"] = _log_mod

print("Step 1 - After putting observatory_log in sys.modules:")
print("  'observatory_log' in sys.modules:", 'observatory_log' in sys.modules)

# Step 2: Now do from observatory_log import llm_query (simulating what llm.py does)
from observatory_log import llm_query
print("\nStep 2 - After from observatory_log import llm_query:")
print("  llm_query:", llm_query)
print("  'observatory_log' in dir():", 'observatory_log' in dir())
print("  'observatory_log' in sys.modules:", 'observatory_log' in sys.modules)

# Step 3: Can we access observatory_log.llm_query?
try:
    result = sys.modules["observatory_log"].llm_query(1, model="m", messages=[])
    print("\nStep 3 - observatory_log.llm_query works:", result)
except NameError as e:
    print("\nStep 3 - NameError:", e)
except Exception as e:
    print("\nStep 3 - Other error:", type(e).__name__, e)
