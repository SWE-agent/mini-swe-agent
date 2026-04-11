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
git -c core.fileMode=false diff 580e644f24f1c5ae5b94784fb73a9953a178fd26
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 580e644f24f1c5ae5b94784fb73a9953a178fd26 tests/file_storage/tests.py tests/settings_tests/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/file_storage/tests.py b/tests/file_storage/tests.py
--- a/tests/file_storage/tests.py
+++ b/tests/file_storage/tests.py
@@ -521,7 +521,7 @@ def test_setting_changed(self):
         defaults_storage = self.storage_class()
         settings = {
             'MEDIA_ROOT': 'overridden_media_root',
-            'MEDIA_URL': 'overridden_media_url/',
+            'MEDIA_URL': '/overridden_media_url/',
             'FILE_UPLOAD_PERMISSIONS': 0o333,
             'FILE_UPLOAD_DIRECTORY_PERMISSIONS': 0o333,
         }
diff --git a/tests/settings_tests/tests.py b/tests/settings_tests/tests.py
--- a/tests/settings_tests/tests.py
+++ b/tests/settings_tests/tests.py
@@ -12,6 +12,7 @@
     override_settings, signals,
 )
 from django.test.utils import requires_tz_support
+from django.urls import clear_script_prefix, set_script_prefix
 
 
 @modify_settings(ITEMS={
@@ -567,3 +568,51 @@ def decorated_function():
         signals.setting_changed.disconnect(self.receiver)
         # This call shouldn't raise any errors.
         decorated_function()
+
+
+class MediaURLStaticURLPrefixTest(SimpleTestCase):
+    def set_script_name(self, val):
+        clear_script_prefix()
+        if val is not None:
+            set_script_prefix(val)
+
+    def test_not_prefixed(self):
+        # Don't add SCRIPT_NAME prefix to valid URLs, absolute paths or None.
+        tests = (
+            '/path/',
+            'http://myhost.com/path/',
+            None,
+        )
+        for setting in ('MEDIA_URL', 'STATIC_URL'):
+            for path in tests:
+                new_settings = {setting: path}
+                with self.settings(**new_settings):
+                    for script_name in ['/somesubpath', '/somesubpath/', '/', '', None]:
+                        with self.subTest(script_name=script_name, **new_settings):
+                            try:
+                                self.set_script_name(script_name)
+                                self.assertEqual(getattr(settings, setting), path)
+                            finally:
+                                clear_script_prefix()
+
+    def test_add_script_name_prefix(self):
+        tests = (
+            # Relative paths.
+            ('/somesubpath', 'path/', '/somesubpath/path/'),
+            ('/somesubpath/', 'path/', '/somesubpath/path/'),
+            ('/', 'path/', '/path/'),
+            # Invalid URLs.
+            ('/somesubpath/', 'htp://myhost.com/path/', '/somesubpath/htp://myhost.com/path/'),
+            # Blank settings.
+            ('/somesubpath/', '', '/somesubpath/'),
+        )
+        for setting in ('MEDIA_URL', 'STATIC_URL'):
+            for script_name, path, expected_path in tests:
+                new_settings = {setting: path}
+                with self.settings(**new_settings):
+                    with self.subTest(script_name=script_name, **new_settings):
+                        try:
+                            self.set_script_name(script_name)
+                            self.assertEqual(getattr(settings, setting), expected_path)
+                        finally:
+                            clear_script_prefix()

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 file_storage.tests settings_tests.tests
: '>>>>> End Test Output'
git checkout 580e644f24f1c5ae5b94784fb73a9953a178fd26 tests/file_storage/tests.py tests/settings_tests/tests.py
