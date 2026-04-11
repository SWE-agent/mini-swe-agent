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
git -c core.fileMode=false diff 895f28f9cbed817c00ab68770433170d83132d90
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 895f28f9cbed817c00ab68770433170d83132d90 tests/aggregation/models.py tests/aggregation/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/aggregation/models.py b/tests/aggregation/models.py
--- a/tests/aggregation/models.py
+++ b/tests/aggregation/models.py
@@ -5,6 +5,7 @@ class Author(models.Model):
     name = models.CharField(max_length=100)
     age = models.IntegerField()
     friends = models.ManyToManyField('self', blank=True)
+    rating = models.FloatField(null=True)
 
     def __str__(self):
         return self.name
diff --git a/tests/aggregation/tests.py b/tests/aggregation/tests.py
--- a/tests/aggregation/tests.py
+++ b/tests/aggregation/tests.py
@@ -1191,6 +1191,22 @@ def test_aggregation_subquery_annotation_values(self):
             },
         ])
 
+    def test_aggregation_subquery_annotation_values_collision(self):
+        books_rating_qs = Book.objects.filter(
+            publisher=OuterRef('pk'),
+            price=Decimal('29.69'),
+        ).values('rating')
+        publisher_qs = Publisher.objects.filter(
+            book__contact__age__gt=20,
+            name=self.p1.name,
+        ).annotate(
+            rating=Subquery(books_rating_qs),
+            contacts_count=Count('book__contact'),
+        ).values('rating').annotate(total_count=Count('rating'))
+        self.assertEqual(list(publisher_qs), [
+            {'rating': 4.0, 'total_count': 2},
+        ])
+
     @skipUnlessDBFeature('supports_subqueries_in_group_by')
     @skipIf(
         connection.vendor == 'mysql' and 'ONLY_FULL_GROUP_BY' in connection.sql_mode,

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 aggregation.models aggregation.tests
: '>>>>> End Test Output'
git checkout 895f28f9cbed817c00ab68770433170d83132d90 tests/aggregation/models.py tests/aggregation/tests.py
