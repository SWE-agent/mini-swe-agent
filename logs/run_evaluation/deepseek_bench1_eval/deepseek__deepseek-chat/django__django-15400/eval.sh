#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 4c76ffc2d6c77c850b4bef8d9acc197d11c47937
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 4c76ffc2d6c77c850b4bef8d9acc197d11c47937 tests/utils_tests/test_lazyobject.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/utils_tests/test_lazyobject.py b/tests/utils_tests/test_lazyobject.py
--- a/tests/utils_tests/test_lazyobject.py
+++ b/tests/utils_tests/test_lazyobject.py
@@ -317,6 +317,17 @@ def test_repr(self):
         self.assertIsInstance(obj._wrapped, int)
         self.assertEqual(repr(obj), "<SimpleLazyObject: 42>")
 
+    def test_add(self):
+        obj1 = self.lazy_wrap(1)
+        self.assertEqual(obj1 + 1, 2)
+        obj2 = self.lazy_wrap(2)
+        self.assertEqual(obj2 + obj1, 3)
+        self.assertEqual(obj1 + obj2, 3)
+
+    def test_radd(self):
+        obj1 = self.lazy_wrap(1)
+        self.assertEqual(1 + obj1, 2)
+
     def test_trace(self):
         # See ticket #19456
         old_trace_func = sys.gettrace()

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 utils_tests.test_lazyobject
: '>>>>> End Test Output'
git checkout 4c76ffc2d6c77c850b4bef8d9acc197d11c47937 tests/utils_tests/test_lazyobject.py
