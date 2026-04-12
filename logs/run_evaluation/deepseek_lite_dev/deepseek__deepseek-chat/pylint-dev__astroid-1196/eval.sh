#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 39c2a9805970ca57093d32bbaf0e6a63e05041d8
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 39c2a9805970ca57093d32bbaf0e6a63e05041d8 tests/unittest_python3.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/unittest_python3.py b/tests/unittest_python3.py
--- a/tests/unittest_python3.py
+++ b/tests/unittest_python3.py
@@ -5,7 +5,9 @@
 import unittest
 from textwrap import dedent
 
-from astroid import nodes
+import pytest
+
+from astroid import exceptions, nodes
 from astroid.builder import AstroidBuilder, extract_node
 from astroid.test_utils import require_version
 
@@ -285,6 +287,33 @@ def test_unpacking_in_dict_getitem(self) -> None:
             self.assertIsInstance(value, nodes.Const)
             self.assertEqual(value.value, expected)
 
+    @staticmethod
+    def test_unpacking_in_dict_getitem_with_ref() -> None:
+        node = extract_node(
+            """
+        a = {1: 2}
+        {**a, 2: 3}  #@
+        """
+        )
+        assert isinstance(node, nodes.Dict)
+
+        for key, expected in ((1, 2), (2, 3)):
+            value = node.getitem(nodes.Const(key))
+            assert isinstance(value, nodes.Const)
+            assert value.value == expected
+
+    @staticmethod
+    def test_unpacking_in_dict_getitem_uninferable() -> None:
+        node = extract_node("{**a, 2: 3}")
+        assert isinstance(node, nodes.Dict)
+
+        with pytest.raises(exceptions.AstroidIndexError):
+            node.getitem(nodes.Const(1))
+
+        value = node.getitem(nodes.Const(2))
+        assert isinstance(value, nodes.Const)
+        assert value.value == 3
+
     def test_format_string(self) -> None:
         code = "f'{greetings} {person}'"
         node = extract_node(code)

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA tests/unittest_python3.py
: '>>>>> End Test Output'
git checkout 39c2a9805970ca57093d32bbaf0e6a63e05041d8 tests/unittest_python3.py
