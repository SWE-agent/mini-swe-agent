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
git -c core.fileMode=false diff df46b329e0900e9e4dc1d60816c1dce6dfc1094e
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout df46b329e0900e9e4dc1d60816c1dce6dfc1094e tests/utils_tests/test_autoreload.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/utils_tests/test_autoreload.py b/tests/utils_tests/test_autoreload.py
--- a/tests/utils_tests/test_autoreload.py
+++ b/tests/utils_tests/test_autoreload.py
@@ -132,6 +132,10 @@ def test_module_without_spec(self):
         del module.__spec__
         self.assertEqual(autoreload.iter_modules_and_files((module,), frozenset()), frozenset())
 
+    def test_main_module_is_resolved(self):
+        main_module = sys.modules['__main__']
+        self.assertFileFound(Path(main_module.__file__))
+
 
 class TestCommonRoots(SimpleTestCase):
     def test_common_roots(self):

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 utils_tests.test_autoreload
: '>>>>> End Test Output'
git checkout df46b329e0900e9e4dc1d60816c1dce6dfc1094e tests/utils_tests/test_autoreload.py
