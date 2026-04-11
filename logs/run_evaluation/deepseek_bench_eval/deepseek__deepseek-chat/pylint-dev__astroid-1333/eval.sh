#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff d2a5b3c7b1e203fec3c7ca73c30eb1785d3d4d0a
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout d2a5b3c7b1e203fec3c7ca73c30eb1785d3d4d0a tests/unittest_modutils.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/unittest_modutils.py b/tests/unittest_modutils.py
--- a/tests/unittest_modutils.py
+++ b/tests/unittest_modutils.py
@@ -30,6 +30,7 @@
 import tempfile
 import unittest
 import xml
+from pathlib import Path
 from xml import etree
 from xml.etree import ElementTree
 
@@ -189,6 +190,30 @@ def test_load_from_module_symlink_on_symlinked_paths_in_syspath(self) -> None:
         # this should be equivalent to: import secret
         self.assertEqual(modutils.modpath_from_file(symlink_secret_path), ["secret"])
 
+    def test_load_packages_without_init(self) -> None:
+        """Test that we correctly find packages with an __init__.py file.
+
+        Regression test for issue reported in:
+        https://github.com/PyCQA/astroid/issues/1327
+        """
+        tmp_dir = Path(tempfile.gettempdir())
+        self.addCleanup(os.chdir, os.curdir)
+        os.chdir(tmp_dir)
+
+        self.addCleanup(shutil.rmtree, tmp_dir / "src")
+        os.mkdir(tmp_dir / "src")
+        os.mkdir(tmp_dir / "src" / "package")
+        with open(tmp_dir / "src" / "__init__.py", "w", encoding="utf-8"):
+            pass
+        with open(tmp_dir / "src" / "package" / "file.py", "w", encoding="utf-8"):
+            pass
+
+        # this should be equivalent to: import secret
+        self.assertEqual(
+            modutils.modpath_from_file(str(Path("src") / "package"), ["."]),
+            ["src", "package"],
+        )
+
 
 class LoadModuleFromPathTest(resources.SysPathSetup, unittest.TestCase):
     def test_do_not_load_twice(self) -> None:

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA tests/unittest_modutils.py
: '>>>>> End Test Output'
git checkout d2a5b3c7b1e203fec3c7ca73c30eb1785d3d4d0a tests/unittest_modutils.py
