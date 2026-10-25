import sys

# Step 1: Import observatory_log first
import observatory_log

print("After import observatory_log:")
print("  observatory_log in sys.modules:", "observatory_log" in sys.modules)
print("  'observatory_log' in dir():", "observatory_log" in dir())

# Step 2: Now import something that does "from observatory_log import ..."
print("\nAfter from observatory_log import ...")
print("  'observatory_log' in dir():", "observatory_log" in dir())
print("  'observatory_log' in sys.modules:", "observatory_log" in sys.modules)

# Step 3: Now simulate what llm.py does - it references observatory_log.llm_query
try:
    result = observatory_log.llm_query(1, model="m", messages=[])
    print("\nobservatory_log.llm_query works from our namespace")
except Exception as e:
    print("\nobservatory_log.llm_query failed:", e)
