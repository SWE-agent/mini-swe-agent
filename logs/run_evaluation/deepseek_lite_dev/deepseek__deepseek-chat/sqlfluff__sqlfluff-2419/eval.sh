#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff f1dba0e1dd764ae72d67c3d5e1471cf14d3db030
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout f1dba0e1dd764ae72d67c3d5e1471cf14d3db030 
git apply -v - <<'EOF_114329324912'
diff --git a/test/rules/std_L060_test.py b/test/rules/std_L060_test.py
new file mode 100644
--- /dev/null
+++ b/test/rules/std_L060_test.py
@@ -0,0 +1,12 @@
+"""Tests the python routines within L060."""
+import sqlfluff
+
+
+def test__rules__std_L060_raised() -> None:
+    """L060 is raised for use of ``IFNULL`` or ``NVL``."""
+    sql = "SELECT\n\tIFNULL(NULL, 100),\n\tNVL(NULL,100);"
+    result = sqlfluff.lint(sql, rules=["L060"])
+
+    assert len(result) == 2
+    assert result[0]["description"] == "Use 'COALESCE' instead of 'IFNULL'."
+    assert result[1]["description"] == "Use 'COALESCE' instead of 'NVL'."

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA test/rules/std_L060_test.py
: '>>>>> End Test Output'
git checkout f1dba0e1dd764ae72d67c3d5e1471cf14d3db030 
