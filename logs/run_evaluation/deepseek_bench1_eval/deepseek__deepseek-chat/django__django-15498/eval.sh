#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff d90e34c61b27fba2527834806639eebbcfab9631
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout d90e34c61b27fba2527834806639eebbcfab9631 tests/view_tests/tests/test_static.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/view_tests/tests/test_static.py b/tests/view_tests/tests/test_static.py
--- a/tests/view_tests/tests/test_static.py
+++ b/tests/view_tests/tests/test_static.py
@@ -191,3 +191,6 @@ def test_was_modified_since_fp(self):
         mtime = 1343416141.107817
         header = http_date(mtime)
         self.assertFalse(was_modified_since(header, mtime))
+
+    def test_was_modified_since_empty_string(self):
+        self.assertTrue(was_modified_since(header="", mtime=1))

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 view_tests.tests.test_static
: '>>>>> End Test Output'
git checkout d90e34c61b27fba2527834806639eebbcfab9631 tests/view_tests/tests/test_static.py
