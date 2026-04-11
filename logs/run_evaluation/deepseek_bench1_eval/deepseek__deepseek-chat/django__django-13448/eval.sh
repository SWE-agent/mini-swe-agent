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
git -c core.fileMode=false diff 7b9596b974fb0ad1868b10c8c2174e10b72be403
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 7b9596b974fb0ad1868b10c8c2174e10b72be403 tests/backends/base/test_creation.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/backends/base/app_unmigrated/__init__.py b/tests/backends/base/app_unmigrated/__init__.py
new file mode 100644
diff --git a/tests/backends/base/app_unmigrated/migrations/0001_initial.py b/tests/backends/base/app_unmigrated/migrations/0001_initial.py
new file mode 100644
--- /dev/null
+++ b/tests/backends/base/app_unmigrated/migrations/0001_initial.py
@@ -0,0 +1,17 @@
+from django.db import migrations, models
+
+
+class Migration(migrations.Migration):
+    initial = True
+
+    dependencies = []
+
+    operations = [
+        migrations.CreateModel(
+            name='Foo',
+            fields=[
+                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
+                ('name', models.CharField(max_length=255)),
+            ],
+        ),
+    ]
diff --git a/tests/backends/base/app_unmigrated/migrations/__init__.py b/tests/backends/base/app_unmigrated/migrations/__init__.py
new file mode 100644
diff --git a/tests/backends/base/app_unmigrated/models.py b/tests/backends/base/app_unmigrated/models.py
new file mode 100644
--- /dev/null
+++ b/tests/backends/base/app_unmigrated/models.py
@@ -0,0 +1,8 @@
+from django.db import models
+
+
+class Foo(models.Model):
+    name = models.CharField(max_length=255)
+
+    class Meta:
+        app_label = 'app_unmigrated'
diff --git a/tests/backends/base/test_creation.py b/tests/backends/base/test_creation.py
--- a/tests/backends/base/test_creation.py
+++ b/tests/backends/base/test_creation.py
@@ -6,6 +6,7 @@
     TEST_DATABASE_PREFIX, BaseDatabaseCreation,
 )
 from django.test import SimpleTestCase, TransactionTestCase
+from django.test.utils import override_settings
 
 from ..models import (
     CircularA, CircularB, Object, ObjectReference, ObjectSelfReference,
@@ -49,31 +50,57 @@ def test_custom_test_name_with_test_prefix(self):
         self.assertEqual(signature[3], test_name)
 
 
+@override_settings(INSTALLED_APPS=['backends.base.app_unmigrated'])
 @mock.patch.object(connection, 'ensure_connection')
-@mock.patch('django.core.management.commands.migrate.Command.handle', return_value=None)
+@mock.patch.object(connection, 'prepare_database')
+@mock.patch('django.db.migrations.recorder.MigrationRecorder.has_table', return_value=False)
+@mock.patch('django.db.migrations.executor.MigrationExecutor.migrate')
+@mock.patch('django.core.management.commands.migrate.Command.sync_apps')
 class TestDbCreationTests(SimpleTestCase):
-    def test_migrate_test_setting_false(self, mocked_migrate, mocked_ensure_connection):
+    available_apps = ['backends.base.app_unmigrated']
+
+    def test_migrate_test_setting_false(self, mocked_sync_apps, mocked_migrate, *mocked_objects):
         test_connection = get_connection_copy()
         test_connection.settings_dict['TEST']['MIGRATE'] = False
         creation = test_connection.creation_class(test_connection)
+        if connection.vendor == 'oracle':
+            # Don't close connection on Oracle.
+            creation.connection.close = mock.Mock()
         old_database_name = test_connection.settings_dict['NAME']
         try:
             with mock.patch.object(creation, '_create_test_db'):
                 creation.create_test_db(verbosity=0, autoclobber=True, serialize=False)
-            mocked_migrate.assert_not_called()
+            # Migrations don't run.
+            mocked_migrate.assert_called()
+            args, kwargs = mocked_migrate.call_args
+            self.assertEqual(args, ([],))
+            self.assertEqual(kwargs['plan'], [])
+            # App is synced.
+            mocked_sync_apps.assert_called()
+            mocked_args, _ = mocked_sync_apps.call_args
+            self.assertEqual(mocked_args[1], {'app_unmigrated'})
         finally:
             with mock.patch.object(creation, '_destroy_test_db'):
                 creation.destroy_test_db(old_database_name, verbosity=0)
 
-    def test_migrate_test_setting_true(self, mocked_migrate, mocked_ensure_connection):
+    def test_migrate_test_setting_true(self, mocked_sync_apps, mocked_migrate, *mocked_objects):
         test_connection = get_connection_copy()
         test_connection.settings_dict['TEST']['MIGRATE'] = True
         creation = test_connection.creation_class(test_connection)
+        if connection.vendor == 'oracle':
+            # Don't close connection on Oracle.
+            creation.connection.close = mock.Mock()
         old_database_name = test_connection.settings_dict['NAME']
         try:
             with mock.patch.object(creation, '_create_test_db'):
                 creation.create_test_db(verbosity=0, autoclobber=True, serialize=False)
-            mocked_migrate.assert_called_once()
+            # Migrations run.
+            mocked_migrate.assert_called()
+            args, kwargs = mocked_migrate.call_args
+            self.assertEqual(args, ([('app_unmigrated', '0001_initial')],))
+            self.assertEqual(len(kwargs['plan']), 1)
+            # App is not synced.
+            mocked_sync_apps.assert_not_called()
         finally:
             with mock.patch.object(creation, '_destroy_test_db'):
                 creation.destroy_test_db(old_database_name, verbosity=0)

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 backends.base.app_unmigrated.__init__ backends.base.app_unmigrated.migrations.0001_initial backends.base.app_unmigrated.migrations.__init__ backends.base.app_unmigrated.models backends.base.test_creation
: '>>>>> End Test Output'
git checkout 7b9596b974fb0ad1868b10c8c2174e10b72be403 tests/backends/base/test_creation.py
