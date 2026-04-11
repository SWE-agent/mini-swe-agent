#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff b55ebe32417e0884b6b8b3e1bc0379033aa221af
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout b55ebe32417e0884b6b8b3e1bc0379033aa221af tests/expressions/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/expressions/tests.py b/tests/expressions/tests.py
--- a/tests/expressions/tests.py
+++ b/tests/expressions/tests.py
@@ -537,6 +537,15 @@ def test_subquery_eq(self):
             qs.query.annotations['small_company'],
         )
 
+    def test_subquery_sql(self):
+        employees = Employee.objects.all()
+        employees_subquery = Subquery(employees)
+        self.assertIs(employees_subquery.query.subquery, True)
+        self.assertIs(employees.query.subquery, False)
+        compiler = employees_subquery.query.get_compiler(connection=connection)
+        sql, _ = employees_subquery.as_sql(compiler, connection)
+        self.assertIn('(SELECT ', sql)
+
     def test_in_subquery(self):
         # This is a contrived test (and you really wouldn't write this query),
         # but it is a succinct way to test the __in=Subquery() construct.

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 expressions.tests
: '>>>>> End Test Output'
git checkout b55ebe32417e0884b6b8b3e1bc0379033aa221af tests/expressions/tests.py
