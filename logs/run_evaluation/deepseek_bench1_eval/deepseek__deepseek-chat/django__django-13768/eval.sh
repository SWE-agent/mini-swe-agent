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
git -c core.fileMode=false diff 965d2d95c630939b53eb60d9c169f5dfc77ee0c6
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 965d2d95c630939b53eb60d9c169f5dfc77ee0c6 tests/dispatch/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/dispatch/tests.py b/tests/dispatch/tests.py
--- a/tests/dispatch/tests.py
+++ b/tests/dispatch/tests.py
@@ -165,13 +165,28 @@ def test_send_robust_fail(self):
         def fails(val, **kwargs):
             raise ValueError('this')
         a_signal.connect(fails)
-        result = a_signal.send_robust(sender=self, val="test")
-        err = result[0][1]
-        self.assertIsInstance(err, ValueError)
-        self.assertEqual(err.args, ('this',))
-        self.assertTrue(hasattr(err, '__traceback__'))
-        self.assertIsInstance(err.__traceback__, TracebackType)
-        a_signal.disconnect(fails)
+        try:
+            with self.assertLogs('django.dispatch', 'ERROR') as cm:
+                result = a_signal.send_robust(sender=self, val='test')
+            err = result[0][1]
+            self.assertIsInstance(err, ValueError)
+            self.assertEqual(err.args, ('this',))
+            self.assertIs(hasattr(err, '__traceback__'), True)
+            self.assertIsInstance(err.__traceback__, TracebackType)
+
+            log_record = cm.records[0]
+            self.assertEqual(
+                log_record.getMessage(),
+                'Error calling '
+                'DispatcherTests.test_send_robust_fail.<locals>.fails in '
+                'Signal.send_robust() (this)',
+            )
+            self.assertIsNotNone(log_record.exc_info)
+            _, exc_value, _ = log_record.exc_info
+            self.assertIsInstance(exc_value, ValueError)
+            self.assertEqual(str(exc_value), 'this')
+        finally:
+            a_signal.disconnect(fails)
         self.assertTestIsClean(a_signal)
 
     def test_disconnection(self):

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 dispatch.tests
: '>>>>> End Test Output'
git checkout 965d2d95c630939b53eb60d9c169f5dfc77ee0c6 tests/dispatch/tests.py
