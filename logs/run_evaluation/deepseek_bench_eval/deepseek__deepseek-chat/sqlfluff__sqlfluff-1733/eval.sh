#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff a1579a16b1d8913d9d7c7d12add374a290bcc78c
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout a1579a16b1d8913d9d7c7d12add374a290bcc78c test/rules/std_L016_L36_combo.py
git apply -v - <<'EOF_114329324912'
diff --git a/test/rules/std_L003_L036_L039_combo_test.py b/test/rules/std_L003_L036_L039_combo_test.py
new file mode 100644
--- /dev/null
+++ b/test/rules/std_L003_L036_L039_combo_test.py
@@ -0,0 +1,36 @@
+"""Tests issue #1373 doesn't reoccur.
+
+The combination of L003 (incorrect indentation), L036 (select targets),
+and L039 (unnecessary white space) can result in incorrect indentation.
+"""
+
+import sqlfluff
+
+
+def test__rules__std_L003_L036_L039():
+    """Verify that double indents don't flag L039."""
+    sql = """
+    WITH example AS (
+        SELECT my_id,
+            other_thing,
+            one_more
+        FROM
+            my_table
+    )
+
+    SELECT *
+    FROM example\n"""
+    fixed_sql = """
+    WITH example AS (
+        SELECT
+            my_id,
+            other_thing,
+            one_more
+        FROM
+            my_table
+    )
+
+    SELECT *
+    FROM example\n"""
+    result = sqlfluff.fix(sql)
+    assert result == fixed_sql
diff --git a/test/rules/std_L016_L36_combo.py b/test/rules/std_L016_L36_combo_test.py
similarity index 100%
rename from test/rules/std_L016_L36_combo.py
rename to test/rules/std_L016_L36_combo_test.py

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA test/rules/std_L003_L036_L039_combo_test.py test/rules/std_L016_L36_combo_test.py
: '>>>>> End Test Output'
git checkout a1579a16b1d8913d9d7c7d12add374a290bcc78c test/rules/std_L016_L36_combo.py
