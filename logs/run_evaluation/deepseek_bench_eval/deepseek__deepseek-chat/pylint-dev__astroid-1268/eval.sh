#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff ce5cbce5ba11cdc2f8139ade66feea1e181a7944
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout ce5cbce5ba11cdc2f8139ade66feea1e181a7944 tests/unittest_nodes.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/unittest_nodes.py b/tests/unittest_nodes.py
--- a/tests/unittest_nodes.py
+++ b/tests/unittest_nodes.py
@@ -306,6 +306,11 @@ def test_f_strings(self):
         ast = abuilder.string_build(code)
         self.assertEqual(ast.as_string().strip(), code.strip())
 
+    @staticmethod
+    def test_as_string_unknown() -> None:
+        assert nodes.Unknown().as_string() == "Unknown.Unknown()"
+        assert nodes.Unknown(lineno=1, col_offset=0).as_string() == "Unknown.Unknown()"
+
 
 class _NodeTest(unittest.TestCase):
     """test transformation of If Node"""

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA tests/unittest_nodes.py
: '>>>>> End Test Output'
git checkout ce5cbce5ba11cdc2f8139ade66feea1e181a7944 tests/unittest_nodes.py
