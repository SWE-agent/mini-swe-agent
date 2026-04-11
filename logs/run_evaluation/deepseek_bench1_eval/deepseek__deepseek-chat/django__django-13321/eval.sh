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
git -c core.fileMode=false diff 35b03788b0607c1f8d2b64e4fa9e1669b0907ea4
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 35b03788b0607c1f8d2b64e4fa9e1669b0907ea4 tests/sessions_tests/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/sessions_tests/tests.py b/tests/sessions_tests/tests.py
--- a/tests/sessions_tests/tests.py
+++ b/tests/sessions_tests/tests.py
@@ -333,11 +333,16 @@ def test_default_hashing_algorith_legacy_decode(self):
             self.assertEqual(self.session._legacy_decode(encoded), data)
 
     def test_decode_failure_logged_to_security(self):
-        bad_encode = base64.b64encode(b'flaskdj:alkdjf').decode('ascii')
-        with self.assertLogs('django.security.SuspiciousSession', 'WARNING') as cm:
-            self.assertEqual({}, self.session.decode(bad_encode))
-        # The failed decode is logged.
-        self.assertIn('corrupted', cm.output[0])
+        tests = [
+            base64.b64encode(b'flaskdj:alkdjf').decode('ascii'),
+            'bad:encoded:value',
+        ]
+        for encoded in tests:
+            with self.subTest(encoded=encoded):
+                with self.assertLogs('django.security.SuspiciousSession', 'WARNING') as cm:
+                    self.assertEqual(self.session.decode(encoded), {})
+                # The failed decode is logged.
+                self.assertIn('Session data corrupted', cm.output[0])
 
     def test_actual_expiry(self):
         # this doesn't work with JSONSerializer (serializing timedelta)

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 sessions_tests.tests
: '>>>>> End Test Output'
git checkout 35b03788b0607c1f8d2b64e4fa9e1669b0907ea4 tests/sessions_tests/tests.py
