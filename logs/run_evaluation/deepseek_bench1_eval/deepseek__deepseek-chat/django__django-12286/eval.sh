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
git -c core.fileMode=false diff 979f61abd322507aafced9627702362e541ec34e
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 979f61abd322507aafced9627702362e541ec34e tests/check_framework/test_translation.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/check_framework/test_translation.py b/tests/check_framework/test_translation.py
--- a/tests/check_framework/test_translation.py
+++ b/tests/check_framework/test_translation.py
@@ -3,7 +3,7 @@
     check_language_settings_consistent, check_setting_language_code,
     check_setting_languages, check_setting_languages_bidi,
 )
-from django.test import SimpleTestCase
+from django.test import SimpleTestCase, override_settings
 
 
 class TranslationCheckTests(SimpleTestCase):
@@ -75,12 +75,36 @@ def test_invalid_languages_bidi(self):
                     Error(msg % tag, id='translation.E003'),
                 ])
 
+    @override_settings(USE_I18N=True, LANGUAGES=[('en', 'English')])
     def test_inconsistent_language_settings(self):
         msg = (
             'You have provided a value for the LANGUAGE_CODE setting that is '
             'not in the LANGUAGES setting.'
         )
-        with self.settings(LANGUAGE_CODE='fr', LANGUAGES=[('en', 'English')]):
-            self.assertEqual(check_language_settings_consistent(None), [
-                Error(msg, id='translation.E004'),
-            ])
+        for tag in ['fr', 'fr-CA', 'fr-357']:
+            with self.subTest(tag), self.settings(LANGUAGE_CODE=tag):
+                self.assertEqual(check_language_settings_consistent(None), [
+                    Error(msg, id='translation.E004'),
+                ])
+
+    @override_settings(
+        USE_I18N=True,
+        LANGUAGES=[
+            ('de', 'German'),
+            ('es', 'Spanish'),
+            ('fr', 'French'),
+            ('ca', 'Catalan'),
+        ],
+    )
+    def test_valid_variant_consistent_language_settings(self):
+        tests = [
+            # language + region.
+            'fr-CA',
+            'es-419',
+            'de-at',
+            # language + region + variant.
+            'ca-ES-valencia',
+        ]
+        for tag in tests:
+            with self.subTest(tag), self.settings(LANGUAGE_CODE=tag):
+                self.assertEqual(check_language_settings_consistent(None), [])

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 check_framework.test_translation
: '>>>>> End Test Output'
git checkout 979f61abd322507aafced9627702362e541ec34e tests/check_framework/test_translation.py
