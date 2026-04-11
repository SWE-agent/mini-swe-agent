#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 1710cdbe79c90665046034fe1700933d038d90ad
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 1710cdbe79c90665046034fe1700933d038d90ad tests/queries/test_q.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/queries/test_q.py b/tests/queries/test_q.py
--- a/tests/queries/test_q.py
+++ b/tests/queries/test_q.py
@@ -8,6 +8,10 @@ def test_combine_and_empty(self):
         self.assertEqual(q & Q(), q)
         self.assertEqual(Q() & q, q)
 
+        q = Q(x__in={}.keys())
+        self.assertEqual(q & Q(), q)
+        self.assertEqual(Q() & q, q)
+
     def test_combine_and_both_empty(self):
         self.assertEqual(Q() & Q(), Q())
 
@@ -16,6 +20,10 @@ def test_combine_or_empty(self):
         self.assertEqual(q | Q(), q)
         self.assertEqual(Q() | q, q)
 
+        q = Q(x__in={}.keys())
+        self.assertEqual(q | Q(), q)
+        self.assertEqual(Q() | q, q)
+
     def test_combine_or_both_empty(self):
         self.assertEqual(Q() | Q(), Q())
 

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 queries.test_q
: '>>>>> End Test Output'
git checkout 1710cdbe79c90665046034fe1700933d038d90ad tests/queries/test_q.py
