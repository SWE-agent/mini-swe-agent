#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 0d4e575c96d408e0efb4dfd0cbfc864219776950
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 0d4e575c96d408e0efb4dfd0cbfc864219776950 tests/backends/test_ddl_references.py tests/migrations/test_operations.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/backends/test_ddl_references.py b/tests/backends/test_ddl_references.py
--- a/tests/backends/test_ddl_references.py
+++ b/tests/backends/test_ddl_references.py
@@ -5,6 +5,7 @@
 from django.db.models import ExpressionList, F
 from django.db.models.functions import Upper
 from django.db.models.indexes import IndexExpression
+from django.db.models.sql import Query
 from django.test import SimpleTestCase, TransactionTestCase
 
 from .models import Person
@@ -229,6 +230,27 @@ def test_rename_table_references(self):
             str(self.expressions),
         )
 
+    def test_rename_table_references_without_alias(self):
+        compiler = Query(Person, alias_cols=False).get_compiler(connection=connection)
+        table = Person._meta.db_table
+        expressions = Expressions(
+            table=table,
+            expressions=ExpressionList(
+                IndexExpression(Upper('last_name')),
+                IndexExpression(F('first_name')),
+            ).resolve_expression(compiler.query),
+            compiler=compiler,
+            quote_value=self.editor.quote_value,
+        )
+        expressions.rename_table_references(table, 'other')
+        self.assertIs(expressions.references_table(table), False)
+        self.assertIs(expressions.references_table('other'), True)
+        expected_str = '(UPPER(%s)), %s' % (
+            self.editor.quote_name('last_name'),
+            self.editor.quote_name('first_name'),
+        )
+        self.assertEqual(str(expressions), expected_str)
+
     def test_rename_column_references(self):
         table = Person._meta.db_table
         self.expressions.rename_column_references(table, 'first_name', 'other')
diff --git a/tests/migrations/test_operations.py b/tests/migrations/test_operations.py
--- a/tests/migrations/test_operations.py
+++ b/tests/migrations/test_operations.py
@@ -2106,6 +2106,25 @@ def test_remove_func_index(self):
         self.assertEqual(definition[1], [])
         self.assertEqual(definition[2], {'model_name': 'Pony', 'name': index_name})
 
+    @skipUnlessDBFeature('supports_expression_indexes')
+    def test_alter_field_with_func_index(self):
+        app_label = 'test_alfuncin'
+        index_name = f'{app_label}_pony_idx'
+        table_name = f'{app_label}_pony'
+        project_state = self.set_up_test_model(
+            app_label,
+            indexes=[models.Index(Abs('pink'), name=index_name)],
+        )
+        operation = migrations.AlterField('Pony', 'pink', models.IntegerField(null=True))
+        new_state = project_state.clone()
+        operation.state_forwards(app_label, new_state)
+        with connection.schema_editor() as editor:
+            operation.database_forwards(app_label, editor, project_state, new_state)
+        self.assertIndexNameExists(table_name, index_name)
+        with connection.schema_editor() as editor:
+            operation.database_backwards(app_label, editor, new_state, project_state)
+        self.assertIndexNameExists(table_name, index_name)
+
     def test_alter_field_with_index(self):
         """
         Test AlterField operation with an index to ensure indexes created via
@@ -2664,6 +2683,26 @@ def test_remove_covering_unique_constraint(self):
             'name': 'covering_pink_constraint_rm',
         })
 
+    def test_alter_field_with_func_unique_constraint(self):
+        app_label = 'test_alfuncuc'
+        constraint_name = f'{app_label}_pony_uq'
+        table_name = f'{app_label}_pony'
+        project_state = self.set_up_test_model(
+            app_label,
+            constraints=[models.UniqueConstraint('pink', 'weight', name=constraint_name)]
+        )
+        operation = migrations.AlterField('Pony', 'pink', models.IntegerField(null=True))
+        new_state = project_state.clone()
+        operation.state_forwards(app_label, new_state)
+        with connection.schema_editor() as editor:
+            operation.database_forwards(app_label, editor, project_state, new_state)
+        if connection.features.supports_expression_indexes:
+            self.assertIndexNameExists(table_name, constraint_name)
+        with connection.schema_editor() as editor:
+            operation.database_backwards(app_label, editor, new_state, project_state)
+        if connection.features.supports_expression_indexes:
+            self.assertIndexNameExists(table_name, constraint_name)
+
     def test_add_func_unique_constraint(self):
         app_label = 'test_adfuncuc'
         constraint_name = f'{app_label}_pony_abs_uq'

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 backends.test_ddl_references migrations.test_operations
: '>>>>> End Test Output'
git checkout 0d4e575c96d408e0efb4dfd0cbfc864219776950 tests/backends/test_ddl_references.py tests/migrations/test_operations.py
