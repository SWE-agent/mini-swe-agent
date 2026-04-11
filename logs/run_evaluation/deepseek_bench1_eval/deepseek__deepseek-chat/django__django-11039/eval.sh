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
git -c core.fileMode=false diff d5276398046ce4a102776a1e67dcac2884d80dfe
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout d5276398046ce4a102776a1e67dcac2884d80dfe tests/migrations/test_commands.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/migrations/test_commands.py b/tests/migrations/test_commands.py
--- a/tests/migrations/test_commands.py
+++ b/tests/migrations/test_commands.py
@@ -536,7 +536,13 @@ def test_sqlmigrate_forwards(self):
         index_op_desc_unique_together = output.find('-- alter unique_together')
         index_tx_end = output.find(connection.ops.end_transaction_sql().lower())
 
-        self.assertGreater(index_tx_start, -1, "Transaction start not found")
+        if connection.features.can_rollback_ddl:
+            self.assertGreater(index_tx_start, -1, "Transaction start not found")
+            self.assertGreater(
+                index_tx_end, index_op_desc_unique_together,
+                "Transaction end not found or found before operation description (unique_together)"
+            )
+
         self.assertGreater(
             index_op_desc_author, index_tx_start,
             "Operation description (author) not found or found before transaction start"
@@ -553,10 +559,6 @@ def test_sqlmigrate_forwards(self):
             index_op_desc_unique_together, index_op_desc_tribble,
             "Operation description (unique_together) not found or found before operation description (tribble)"
         )
-        self.assertGreater(
-            index_tx_end, index_op_desc_unique_together,
-            "Transaction end not found or found before operation description (unique_together)"
-        )
 
     @override_settings(MIGRATION_MODULES={"migrations": "migrations.test_migrations"})
     def test_sqlmigrate_backwards(self):
@@ -577,7 +579,12 @@ def test_sqlmigrate_backwards(self):
         index_drop_table = output.rfind('drop table')
         index_tx_end = output.find(connection.ops.end_transaction_sql().lower())
 
-        self.assertGreater(index_tx_start, -1, "Transaction start not found")
+        if connection.features.can_rollback_ddl:
+            self.assertGreater(index_tx_start, -1, "Transaction start not found")
+            self.assertGreater(
+                index_tx_end, index_op_desc_unique_together,
+                "Transaction end not found or found before DROP TABLE"
+            )
         self.assertGreater(
             index_op_desc_unique_together, index_tx_start,
             "Operation description (unique_together) not found or found before transaction start"
@@ -595,10 +602,6 @@ def test_sqlmigrate_backwards(self):
             index_drop_table, index_op_desc_author,
             "DROP TABLE not found or found before operation description (author)"
         )
-        self.assertGreater(
-            index_tx_end, index_op_desc_unique_together,
-            "Transaction end not found or found before DROP TABLE"
-        )
 
         # Cleanup by unmigrating everything
         call_command("migrate", "migrations", "zero", verbosity=0)
@@ -616,6 +619,22 @@ def test_sqlmigrate_for_non_atomic_migration(self):
             self.assertNotIn(connection.ops.start_transaction_sql().lower(), queries)
         self.assertNotIn(connection.ops.end_transaction_sql().lower(), queries)
 
+    @override_settings(MIGRATION_MODULES={'migrations': 'migrations.test_migrations'})
+    def test_sqlmigrate_for_non_transactional_databases(self):
+        """
+        Transaction wrappers aren't shown for databases that don't support
+        transactional DDL.
+        """
+        out = io.StringIO()
+        with mock.patch.object(connection.features, 'can_rollback_ddl', False):
+            call_command('sqlmigrate', 'migrations', '0001', stdout=out)
+        output = out.getvalue().lower()
+        queries = [q.strip() for q in output.splitlines()]
+        start_transaction_sql = connection.ops.start_transaction_sql()
+        if start_transaction_sql:
+            self.assertNotIn(start_transaction_sql.lower(), queries)
+        self.assertNotIn(connection.ops.end_transaction_sql().lower(), queries)
+
     @override_settings(
         INSTALLED_APPS=[
             "migrations.migrations_test_apps.migrated_app",

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 migrations.test_commands
: '>>>>> End Test Output'
git checkout d5276398046ce4a102776a1e67dcac2884d80dfe tests/migrations/test_commands.py
