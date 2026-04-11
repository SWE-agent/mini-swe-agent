#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 6cf238d089cf4b6753c94cfc089b4a47487711e5
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 6cf238d089cf4b6753c94cfc089b4a47487711e5 tests/unittest_brain_builtin.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/unittest_brain_builtin.py b/tests/unittest_brain_builtin.py
--- a/tests/unittest_brain_builtin.py
+++ b/tests/unittest_brain_builtin.py
@@ -103,6 +103,12 @@ def test_string_format(self, format_string: str) -> None:
             """
             "My name is {fname}, I'm {age}".format(fsname = "Daniel", age = 12)
             """,
+            """
+            "My unicode character is {:c}".format(None)
+            """,
+            """
+            "My hex format is {:4x}".format('1')
+            """,
         ],
     )
     def test_string_format_uninferable(self, format_string: str) -> None:

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA tests/unittest_brain_builtin.py
: '>>>>> End Test Output'
git checkout 6cf238d089cf4b6753c94cfc089b4a47487711e5 tests/unittest_brain_builtin.py
