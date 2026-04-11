#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff c5cd8783825b5f6384417dac5f3889b4210b7d08
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout c5cd8783825b5f6384417dac5f3889b4210b7d08 tests/template_tests/test_autoreloader.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/template_tests/test_autoreloader.py b/tests/template_tests/test_autoreloader.py
--- a/tests/template_tests/test_autoreloader.py
+++ b/tests/template_tests/test_autoreloader.py
@@ -39,6 +39,19 @@ def test_non_template_changed(self, mock_reset):
         self.assertIsNone(autoreload.template_changed(None, Path(__file__)))
         mock_reset.assert_not_called()
 
+    @override_settings(
+        TEMPLATES=[
+            {
+                'DIRS': [ROOT],
+                'BACKEND': 'django.template.backends.django.DjangoTemplates',
+            }
+        ]
+    )
+    @mock.patch('django.template.autoreload.reset_loaders')
+    def test_non_template_changed_in_template_directory(self, mock_reset):
+        self.assertIsNone(autoreload.template_changed(None, Path(__file__)))
+        mock_reset.assert_not_called()
+
     def test_watch_for_template_changes(self):
         mock_reloader = mock.MagicMock()
         autoreload.watch_for_template_changes(mock_reloader)

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 template_tests.test_autoreloader
: '>>>>> End Test Output'
git checkout c5cd8783825b5f6384417dac5f3889b4210b7d08 tests/template_tests/test_autoreloader.py
