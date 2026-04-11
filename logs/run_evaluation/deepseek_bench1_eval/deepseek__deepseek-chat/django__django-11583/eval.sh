#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en
export LC_ALL=en_US.UTF-8
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 60dc957a825232fdda9138e2f8878b2ca407a7c9
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 60dc957a825232fdda9138e2f8878b2ca407a7c9 tests/utils_tests/test_autoreload.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/utils_tests/test_autoreload.py b/tests/utils_tests/test_autoreload.py
--- a/tests/utils_tests/test_autoreload.py
+++ b/tests/utils_tests/test_autoreload.py
@@ -140,6 +140,17 @@ def test_main_module_without_file_is_not_resolved(self):
         fake_main = types.ModuleType('__main__')
         self.assertEqual(autoreload.iter_modules_and_files((fake_main,), frozenset()), frozenset())
 
+    def test_path_with_embedded_null_bytes(self):
+        for path in (
+            'embedded_null_byte\x00.py',
+            'di\x00rectory/embedded_null_byte.py',
+        ):
+            with self.subTest(path=path):
+                self.assertEqual(
+                    autoreload.iter_modules_and_files((), frozenset([path])),
+                    frozenset(),
+                )
+
 
 class TestCommonRoots(SimpleTestCase):
     def test_common_roots(self):

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 utils_tests.test_autoreload
: '>>>>> End Test Output'
git checkout 60dc957a825232fdda9138e2f8878b2ca407a7c9 tests/utils_tests/test_autoreload.py
