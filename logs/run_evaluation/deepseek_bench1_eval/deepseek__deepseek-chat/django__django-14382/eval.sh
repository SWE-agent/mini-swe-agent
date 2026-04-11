#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 29345aecf6e8d53ccb3577a3762bb0c263f7558d
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 29345aecf6e8d53ccb3577a3762bb0c263f7558d tests/admin_scripts/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/admin_scripts/tests.py b/tests/admin_scripts/tests.py
--- a/tests/admin_scripts/tests.py
+++ b/tests/admin_scripts/tests.py
@@ -2206,6 +2206,13 @@ def test_importable_target_name(self):
             "another directory."
         )
 
+    def test_trailing_slash_in_target_app_directory_name(self):
+        app_dir = os.path.join(self.test_dir, 'apps', 'app1')
+        os.makedirs(app_dir)
+        _, err = self.run_django_admin(['startapp', 'app', os.path.join('apps', 'app1', '')])
+        self.assertNoOutput(err)
+        self.assertIs(os.path.exists(os.path.join(app_dir, 'apps.py')), True)
+
     def test_overlaying_app(self):
         # Use a subdirectory so it is outside the PYTHONPATH.
         os.makedirs(os.path.join(self.test_dir, 'apps/app1'))

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 admin_scripts.tests
: '>>>>> End Test Output'
git checkout 29345aecf6e8d53ccb3577a3762bb0c263f7558d tests/admin_scripts/tests.py
