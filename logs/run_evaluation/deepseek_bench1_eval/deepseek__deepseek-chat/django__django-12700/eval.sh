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
git -c core.fileMode=false diff d51c50d836c5cf8db5566da17963f871be554615
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout d51c50d836c5cf8db5566da17963f871be554615 tests/view_tests/tests/test_debug.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/view_tests/tests/test_debug.py b/tests/view_tests/tests/test_debug.py
--- a/tests/view_tests/tests/test_debug.py
+++ b/tests/view_tests/tests/test_debug.py
@@ -1249,6 +1249,41 @@ def test_cleanse_setting_recurses_in_dictionary(self):
             {'login': 'cooper', 'password': reporter_filter.cleansed_substitute},
         )
 
+    def test_cleanse_setting_recurses_in_list_tuples(self):
+        reporter_filter = SafeExceptionReporterFilter()
+        initial = [
+            {
+                'login': 'cooper',
+                'password': 'secret',
+                'apps': (
+                    {'name': 'app1', 'api_key': 'a06b-c462cffae87a'},
+                    {'name': 'app2', 'api_key': 'a9f4-f152e97ad808'},
+                ),
+                'tokens': ['98b37c57-ec62-4e39', '8690ef7d-8004-4916'],
+            },
+            {'SECRET_KEY': 'c4d77c62-6196-4f17-a06b-c462cffae87a'},
+        ]
+        cleansed = [
+            {
+                'login': 'cooper',
+                'password': reporter_filter.cleansed_substitute,
+                'apps': (
+                    {'name': 'app1', 'api_key': reporter_filter.cleansed_substitute},
+                    {'name': 'app2', 'api_key': reporter_filter.cleansed_substitute},
+                ),
+                'tokens': reporter_filter.cleansed_substitute,
+            },
+            {'SECRET_KEY': reporter_filter.cleansed_substitute},
+        ]
+        self.assertEqual(
+            reporter_filter.cleanse_setting('SETTING_NAME', initial),
+            cleansed,
+        )
+        self.assertEqual(
+            reporter_filter.cleanse_setting('SETTING_NAME', tuple(initial)),
+            tuple(cleansed),
+        )
+
     def test_request_meta_filtering(self):
         request = self.rf.get('/', HTTP_SECRET_HEADER='super_secret')
         reporter_filter = SafeExceptionReporterFilter()

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 view_tests.tests.test_debug
: '>>>>> End Test Output'
git checkout d51c50d836c5cf8db5566da17963f871be554615 tests/view_tests/tests/test_debug.py
