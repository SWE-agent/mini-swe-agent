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
git -c core.fileMode=false diff 2f72480fbd27896c986c45193e1603e35c0b19a7
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 2f72480fbd27896c986c45193e1603e35c0b19a7 tests/lookup/models.py tests/lookup/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/lookup/models.py b/tests/lookup/models.py
--- a/tests/lookup/models.py
+++ b/tests/lookup/models.py
@@ -96,3 +96,15 @@ class Product(models.Model):
 class Stock(models.Model):
     product = models.ForeignKey(Product, models.CASCADE)
     qty_available = models.DecimalField(max_digits=6, decimal_places=2)
+
+
+class Freebie(models.Model):
+    gift_product = models.ForeignKey(Product, models.CASCADE)
+    stock_id = models.IntegerField(blank=True, null=True)
+
+    stock = models.ForeignObject(
+        Stock,
+        from_fields=['stock_id', 'gift_product'],
+        to_fields=['id', 'product'],
+        on_delete=models.CASCADE,
+    )
diff --git a/tests/lookup/tests.py b/tests/lookup/tests.py
--- a/tests/lookup/tests.py
+++ b/tests/lookup/tests.py
@@ -9,9 +9,10 @@
 from django.db.models.expressions import Exists, OuterRef
 from django.db.models.functions import Substr
 from django.test import TestCase, skipUnlessDBFeature
+from django.utils.deprecation import RemovedInDjango40Warning
 
 from .models import (
-    Article, Author, Game, IsNullWithNoneAsRHS, Player, Season, Tag,
+    Article, Author, Freebie, Game, IsNullWithNoneAsRHS, Player, Season, Tag,
 )
 
 
@@ -969,3 +970,24 @@ def test_exact_query_rhs_with_selected_columns(self):
         ).values('max_id')
         authors = Author.objects.filter(id=authors_max_ids[:1])
         self.assertEqual(authors.get(), newest_author)
+
+    def test_isnull_non_boolean_value(self):
+        # These tests will catch ValueError in Django 4.0 when using
+        # non-boolean values for an isnull lookup becomes forbidden.
+        # msg = (
+        #     'The QuerySet value for an isnull lookup must be True or False.'
+        # )
+        msg = (
+            'Using a non-boolean value for an isnull lookup is deprecated, '
+            'use True or False instead.'
+        )
+        tests = [
+            Author.objects.filter(alias__isnull=1),
+            Article.objects.filter(author__isnull=1),
+            Season.objects.filter(games__isnull=1),
+            Freebie.objects.filter(stock__isnull=1),
+        ]
+        for qs in tests:
+            with self.subTest(qs=qs):
+                with self.assertWarnsMessage(RemovedInDjango40Warning, msg):
+                    qs.exists()

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 lookup.models lookup.tests
: '>>>>> End Test Output'
git checkout 2f72480fbd27896c986c45193e1603e35c0b19a7 tests/lookup/models.py tests/lookup/tests.py
