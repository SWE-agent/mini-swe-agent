#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 0c9ab0fe56703fa83c73e514a1020d398d23fa7f
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 0c9ab0fe56703fa83c73e514a1020d398d23fa7f tests/unittest_raw_building.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/unittest_raw_building.py b/tests/unittest_raw_building.py
--- a/tests/unittest_raw_building.py
+++ b/tests/unittest_raw_building.py
@@ -8,8 +8,15 @@
 # For details: https://github.com/PyCQA/astroid/blob/main/LICENSE
 # Copyright (c) https://github.com/PyCQA/astroid/blob/main/CONTRIBUTORS.txt
 
+from __future__ import annotations
+
+import logging
+import os
+import sys
 import types
 import unittest
+from typing import Any
+from unittest import mock
 
 import _io
 import pytest
@@ -117,5 +124,45 @@ def test_module_object_with_broken_getattr(self) -> None:
         AstroidBuilder().inspect_build(fm_getattr, "test")
 
 
+@pytest.mark.skipif(
+    "posix" not in sys.builtin_module_names, reason="Platform doesn't support posix"
+)
+def test_build_module_getattr_catch_output(
+    capsys: pytest.CaptureFixture[str],
+    caplog: pytest.LogCaptureFixture,
+) -> None:
+    """Catch stdout and stderr in module __getattr__ calls when building a module.
+
+    Usually raised by DeprecationWarning or FutureWarning.
+    """
+    caplog.set_level(logging.INFO)
+    original_sys = sys.modules
+    original_module = sys.modules["posix"]
+    expected_out = "INFO (TEST): Welcome to posix!"
+    expected_err = "WARNING (TEST): Monkey-patched version of posix - module getattr"
+
+    class CustomGetattr:
+        def __getattr__(self, name: str) -> Any:
+            print(f"{expected_out}")
+            print(expected_err, file=sys.stderr)
+            return getattr(original_module, name)
+
+    def mocked_sys_modules_getitem(name: str) -> types.ModuleType | CustomGetattr:
+        if name != "posix":
+            return original_sys[name]
+        return CustomGetattr()
+
+    with mock.patch("astroid.raw_building.sys.modules") as sys_mock:
+        sys_mock.__getitem__.side_effect = mocked_sys_modules_getitem
+        builder = AstroidBuilder()
+        builder.inspect_build(os)
+
+    out, err = capsys.readouterr()
+    assert expected_out in caplog.text
+    assert expected_err in caplog.text
+    assert not out
+    assert not err
+
+
 if __name__ == "__main__":
     unittest.main()

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA tests/unittest_raw_building.py
: '>>>>> End Test Output'
git checkout 0c9ab0fe56703fa83c73e514a1020d398d23fa7f tests/unittest_raw_building.py
