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
git -c core.fileMode=false diff d232fd76a85870daf345fd8f8d617fe7802ae194
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout d232fd76a85870daf345fd8f8d617fe7802ae194 tests/migrations/test_autodetector.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/migrations/test_autodetector.py b/tests/migrations/test_autodetector.py
--- a/tests/migrations/test_autodetector.py
+++ b/tests/migrations/test_autodetector.py
@@ -932,6 +932,30 @@ def test_rename_foreign_object_fields(self):
             changes, 'app', 0, 1, model_name='bar', old_name='second', new_name='second_renamed',
         )
 
+    def test_rename_referenced_primary_key(self):
+        before = [
+            ModelState('app', 'Foo', [
+                ('id', models.CharField(primary_key=True, serialize=False)),
+            ]),
+            ModelState('app', 'Bar', [
+                ('id', models.AutoField(primary_key=True)),
+                ('foo', models.ForeignKey('app.Foo', models.CASCADE)),
+            ]),
+        ]
+        after = [
+            ModelState('app', 'Foo', [
+                ('renamed_id', models.CharField(primary_key=True, serialize=False))
+            ]),
+            ModelState('app', 'Bar', [
+                ('id', models.AutoField(primary_key=True)),
+                ('foo', models.ForeignKey('app.Foo', models.CASCADE)),
+            ]),
+        ]
+        changes = self.get_changes(before, after, MigrationQuestioner({'ask_rename': True}))
+        self.assertNumberMigrations(changes, 'app', 1)
+        self.assertOperationTypes(changes, 'app', 0, ['RenameField'])
+        self.assertOperationAttributes(changes, 'app', 0, 0, old_name='id', new_name='renamed_id')
+
     def test_rename_field_preserved_db_column(self):
         """
         RenameField is used if a field is renamed and db_column equal to the

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 migrations.test_autodetector
: '>>>>> End Test Output'
git checkout d232fd76a85870daf345fd8f8d617fe7802ae194 tests/migrations/test_autodetector.py
