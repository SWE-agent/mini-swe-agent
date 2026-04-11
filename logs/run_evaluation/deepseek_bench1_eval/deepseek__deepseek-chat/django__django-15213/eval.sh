#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 03cadb912c78b769d6bf4a943a2a35fc1d952960
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 03cadb912c78b769d6bf4a943a2a35fc1d952960 tests/annotations/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/annotations/tests.py b/tests/annotations/tests.py
--- a/tests/annotations/tests.py
+++ b/tests/annotations/tests.py
@@ -210,6 +210,26 @@ def test_empty_expression_annotation(self):
         self.assertEqual(len(books), Book.objects.count())
         self.assertTrue(all(not book.selected for book in books))
 
+    def test_full_expression_annotation(self):
+        books = Book.objects.annotate(
+            selected=ExpressionWrapper(~Q(pk__in=[]), output_field=BooleanField()),
+        )
+        self.assertEqual(len(books), Book.objects.count())
+        self.assertTrue(all(book.selected for book in books))
+
+    def test_full_expression_annotation_with_aggregation(self):
+        qs = Book.objects.filter(isbn='159059725').annotate(
+            selected=ExpressionWrapper(~Q(pk__in=[]), output_field=BooleanField()),
+            rating_count=Count('rating'),
+        )
+        self.assertEqual([book.rating_count for book in qs], [1])
+
+    def test_aggregate_over_full_expression_annotation(self):
+        qs = Book.objects.annotate(
+            selected=ExpressionWrapper(~Q(pk__in=[]), output_field=BooleanField()),
+        ).aggregate(Sum('selected'))
+        self.assertEqual(qs['selected__sum'], Book.objects.count())
+
     def test_empty_queryset_annotation(self):
         qs = Author.objects.annotate(
             empty=Subquery(Author.objects.values('id').none())

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 annotations.tests
: '>>>>> End Test Output'
git checkout 03cadb912c78b769d6bf4a943a2a35fc1d952960 tests/annotations/tests.py
