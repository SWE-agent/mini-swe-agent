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
git -c core.fileMode=false diff bceadd2788dc2dad53eba0caae172bd8522fd483
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout bceadd2788dc2dad53eba0caae172bd8522fd483 
git apply -v - <<'EOF_114329324912'
diff --git a/tests/model_fields/test_filepathfield.py b/tests/model_fields/test_filepathfield.py
new file mode 100644
--- /dev/null
+++ b/tests/model_fields/test_filepathfield.py
@@ -0,0 +1,22 @@
+import os
+
+from django.db.models import FilePathField
+from django.test import SimpleTestCase
+
+
+class FilePathFieldTests(SimpleTestCase):
+    def test_path(self):
+        path = os.path.dirname(__file__)
+        field = FilePathField(path=path)
+        self.assertEqual(field.path, path)
+        self.assertEqual(field.formfield().path, path)
+
+    def test_callable_path(self):
+        path = os.path.dirname(__file__)
+
+        def generate_path():
+            return path
+
+        field = FilePathField(path=generate_path)
+        self.assertEqual(field.path(), path)
+        self.assertEqual(field.formfield().path, path)

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 model_fields.test_filepathfield
: '>>>>> End Test Output'
git checkout bceadd2788dc2dad53eba0caae172bd8522fd483 
