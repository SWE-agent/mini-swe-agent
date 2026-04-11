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
git -c core.fileMode=false diff ef082ebb84f00e38af4e8880d04e8365c2766d34
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout ef082ebb84f00e38af4e8880d04e8365c2766d34 tests/expressions/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/expressions/tests.py b/tests/expressions/tests.py
--- a/tests/expressions/tests.py
+++ b/tests/expressions/tests.py
@@ -384,6 +384,29 @@ def test_order_by_exists(self):
         )
         self.assertSequenceEqual(mustermanns_by_seniority, [self.max, mary])
 
+    def test_order_by_multiline_sql(self):
+        raw_order_by = (
+            RawSQL('''
+                CASE WHEN num_employees > 1000
+                     THEN num_chairs
+                     ELSE 0 END
+            ''', []).desc(),
+            RawSQL('''
+                CASE WHEN num_chairs > 1
+                     THEN 1
+                     ELSE 0 END
+            ''', []).asc()
+        )
+        for qs in (
+            Company.objects.all(),
+            Company.objects.distinct(),
+        ):
+            with self.subTest(qs=qs):
+                self.assertSequenceEqual(
+                    qs.order_by(*raw_order_by),
+                    [self.example_inc, self.gmbh, self.foobar_ltd],
+                )
+
     def test_outerref(self):
         inner = Company.objects.filter(point_of_contact=OuterRef('pk'))
         msg = (

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 expressions.tests
: '>>>>> End Test Output'
git checkout ef082ebb84f00e38af4e8880d04e8365c2766d34 tests/expressions/tests.py
