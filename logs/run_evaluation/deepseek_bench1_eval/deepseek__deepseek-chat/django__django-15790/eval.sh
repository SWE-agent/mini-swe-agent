#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff c627226d05dd52aef59447dcfb29cec2c2b11b8a
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout c627226d05dd52aef59447dcfb29cec2c2b11b8a tests/check_framework/test_templates.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/check_framework/test_templates.py b/tests/check_framework/test_templates.py
--- a/tests/check_framework/test_templates.py
+++ b/tests/check_framework/test_templates.py
@@ -158,6 +158,19 @@ def test_template_tags_with_same_library_name(self):
                 [self.error_same_tags],
             )
 
+    @override_settings(
+        INSTALLED_APPS=["check_framework.template_test_apps.same_tags_app_1"]
+    )
+    def test_template_tags_same_library_in_installed_apps_libraries(self):
+        with self.settings(
+            TEMPLATES=[
+                self.get_settings(
+                    "same_tags", "same_tags_app_1.templatetags.same_tags"
+                ),
+            ]
+        ):
+            self.assertEqual(check_for_template_tags_with_the_same_name(None), [])
+
     @override_settings(
         INSTALLED_APPS=["check_framework.template_test_apps.same_tags_app_1"]
     )

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 check_framework.test_templates
: '>>>>> End Test Output'
git checkout c627226d05dd52aef59447dcfb29cec2c2b11b8a tests/check_framework/test_templates.py
