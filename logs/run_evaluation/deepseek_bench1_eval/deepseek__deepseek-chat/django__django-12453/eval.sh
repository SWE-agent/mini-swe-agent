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
git -c core.fileMode=false diff b330b918e979ea39a21d47b61172d112caf432c3
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout b330b918e979ea39a21d47b61172d112caf432c3 tests/backends/base/test_creation.py tests/backends/models.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/backends/base/test_creation.py b/tests/backends/base/test_creation.py
--- a/tests/backends/base/test_creation.py
+++ b/tests/backends/base/test_creation.py
@@ -7,6 +7,8 @@
 )
 from django.test import SimpleTestCase
 
+from ..models import Object, ObjectReference
+
 
 def get_connection_copy():
     # Get a copy of the default connection. (Can't use django.db.connection
@@ -73,3 +75,29 @@ def test_migrate_test_setting_true(self, mocked_migrate, mocked_ensure_connectio
         finally:
             with mock.patch.object(creation, '_destroy_test_db'):
                 creation.destroy_test_db(old_database_name, verbosity=0)
+
+
+class TestDeserializeDbFromString(SimpleTestCase):
+    databases = {'default'}
+
+    def test_circular_reference(self):
+        # deserialize_db_from_string() handles circular references.
+        data = """
+        [
+            {
+                "model": "backends.object",
+                "pk": 1,
+                "fields": {"obj_ref": 1, "related_objects": []}
+            },
+            {
+                "model": "backends.objectreference",
+                "pk": 1,
+                "fields": {"obj": 1}
+            }
+        ]
+        """
+        connection.creation.deserialize_db_from_string(data)
+        obj = Object.objects.get()
+        obj_ref = ObjectReference.objects.get()
+        self.assertEqual(obj.obj_ref, obj_ref)
+        self.assertEqual(obj_ref.obj, obj)
diff --git a/tests/backends/models.py b/tests/backends/models.py
--- a/tests/backends/models.py
+++ b/tests/backends/models.py
@@ -89,6 +89,7 @@ def __str__(self):
 
 class Object(models.Model):
     related_objects = models.ManyToManyField("self", db_constraint=False, symmetrical=False)
+    obj_ref = models.ForeignKey('ObjectReference', models.CASCADE, null=True)
 
     def __str__(self):
         return str(self.id)

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 backends.base.test_creation backends.models
: '>>>>> End Test Output'
git checkout b330b918e979ea39a21d47b61172d112caf432c3 tests/backends/base/test_creation.py tests/backends/models.py
