#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 6a970a8b4600eb91be25f38caed0a52269d6303d
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 6a970a8b4600eb91be25f38caed0a52269d6303d tests/defer/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/defer/tests.py b/tests/defer/tests.py
--- a/tests/defer/tests.py
+++ b/tests/defer/tests.py
@@ -49,8 +49,16 @@ def test_defer_only_chaining(self):
         qs = Primary.objects.all()
         self.assert_delayed(qs.only("name", "value").defer("name")[0], 2)
         self.assert_delayed(qs.defer("name").only("value", "name")[0], 2)
+        self.assert_delayed(qs.defer('name').only('name').only('value')[0], 2)
         self.assert_delayed(qs.defer("name").only("value")[0], 2)
         self.assert_delayed(qs.only("name").defer("value")[0], 2)
+        self.assert_delayed(qs.only('name').defer('name').defer('value')[0], 1)
+        self.assert_delayed(qs.only('name').defer('name', 'value')[0], 1)
+
+    def test_defer_only_clear(self):
+        qs = Primary.objects.all()
+        self.assert_delayed(qs.only('name').defer('name')[0], 0)
+        self.assert_delayed(qs.defer('name').only('name')[0], 0)
 
     def test_defer_on_an_already_deferred_field(self):
         qs = Primary.objects.all()

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 defer.tests
: '>>>>> End Test Output'
git checkout 6a970a8b4600eb91be25f38caed0a52269d6303d tests/defer/tests.py
