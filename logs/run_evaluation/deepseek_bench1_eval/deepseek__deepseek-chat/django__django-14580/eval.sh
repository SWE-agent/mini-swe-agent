#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 36fa071d6ebd18a61c4d7f1b5c9d17106134bd44
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 36fa071d6ebd18a61c4d7f1b5c9d17106134bd44 tests/migrations/test_writer.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/migrations/test_writer.py b/tests/migrations/test_writer.py
--- a/tests/migrations/test_writer.py
+++ b/tests/migrations/test_writer.py
@@ -658,6 +658,13 @@ def test_serialize_functools_partialmethod(self):
     def test_serialize_type_none(self):
         self.assertSerializedEqual(type(None))
 
+    def test_serialize_type_model(self):
+        self.assertSerializedEqual(models.Model)
+        self.assertSerializedResultEqual(
+            MigrationWriter.serialize(models.Model),
+            ("('models.Model', {'from django.db import models'})", set()),
+        )
+
     def test_simple_migration(self):
         """
         Tests serializing a simple migration.

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 migrations.test_writer
: '>>>>> End Test Output'
git checkout 36fa071d6ebd18a61c4d7f1b5c9d17106134bd44 tests/migrations/test_writer.py
