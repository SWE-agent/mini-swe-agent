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
git -c core.fileMode=false diff 62254c5202e80a68f4fe6572a2be46a3d953de1a
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 62254c5202e80a68f4fe6572a2be46a3d953de1a 
git apply -v - <<'EOF_114329324912'
diff --git a/tests/backends/sqlite/test_creation.py b/tests/backends/sqlite/test_creation.py
new file mode 100644
--- /dev/null
+++ b/tests/backends/sqlite/test_creation.py
@@ -0,0 +1,18 @@
+import copy
+import unittest
+
+from django.db import connection
+from django.test import SimpleTestCase
+
+
+@unittest.skipUnless(connection.vendor == 'sqlite', 'SQLite tests')
+class TestDbSignatureTests(SimpleTestCase):
+    def test_custom_test_name(self):
+        saved_settings = copy.deepcopy(connection.settings_dict)
+        try:
+            connection.settings_dict['NAME'] = None
+            connection.settings_dict['TEST']['NAME'] = 'custom.sqlite.db'
+            signature = connection.creation.test_db_signature()
+            self.assertEqual(signature, (None, 'custom.sqlite.db'))
+        finally:
+            connection.settings_dict = saved_settings

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 backends.sqlite.test_creation
: '>>>>> End Test Output'
git checkout 62254c5202e80a68f4fe6572a2be46a3d953de1a 
