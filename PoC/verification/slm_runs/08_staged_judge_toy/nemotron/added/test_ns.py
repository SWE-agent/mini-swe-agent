namespace = {}

import observatory_log

namespace["observatory_log"] = observatory_log
print("After import:", "observability_log" in namespace)

print("After from import:", "observability_log" in namespace)

try:
    result = namespace["observatory_log"].llm_query(1, model="m", messages=[])
    print("works:", result)
except NameError as e:
    print("NameError:", e)
