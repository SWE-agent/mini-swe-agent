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
git -c core.fileMode=false diff 3346b78a8a872286a245d1e77ef4718fc5e6be1a
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 3346b78a8a872286a245d1e77ef4718fc5e6be1a tests/lookup/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/lookup/tests.py b/tests/lookup/tests.py
--- a/tests/lookup/tests.py
+++ b/tests/lookup/tests.py
@@ -5,6 +5,7 @@
 
 from django.core.exceptions import FieldError
 from django.db import connection
+from django.db.models import Max
 from django.db.models.expressions import Exists, OuterRef
 from django.db.models.functions import Substr
 from django.test import TestCase, skipUnlessDBFeature
@@ -956,3 +957,15 @@ def test_nested_outerref_lhs(self):
             ),
         )
         self.assertEqual(qs.get(has_author_alias_match=True), tag)
+
+    def test_exact_query_rhs_with_selected_columns(self):
+        newest_author = Author.objects.create(name='Author 2')
+        authors_max_ids = Author.objects.filter(
+            name='Author 2',
+        ).values(
+            'name',
+        ).annotate(
+            max_id=Max('id'),
+        ).values('max_id')
+        authors = Author.objects.filter(id=authors_max_ids[:1])
+        self.assertEqual(authors.get(), newest_author)

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 lookup.tests
: '>>>>> End Test Output'
git checkout 3346b78a8a872286a245d1e77ef4718fc5e6be1a tests/lookup/tests.py
