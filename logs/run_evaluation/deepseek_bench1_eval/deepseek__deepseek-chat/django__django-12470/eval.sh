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
git -c core.fileMode=false diff 142ab6846ac09d6d401e26fc8b6b988a583ac0f5
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 142ab6846ac09d6d401e26fc8b6b988a583ac0f5 tests/model_inheritance/models.py tests/model_inheritance/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/model_inheritance/models.py b/tests/model_inheritance/models.py
--- a/tests/model_inheritance/models.py
+++ b/tests/model_inheritance/models.py
@@ -181,6 +181,8 @@ class GrandParent(models.Model):
     place = models.ForeignKey(Place, models.CASCADE, null=True, related_name='+')
 
     class Meta:
+        # Ordering used by test_inherited_ordering_pk_desc.
+        ordering = ['-pk']
         unique_together = ('first_name', 'last_name')
 
 
diff --git a/tests/model_inheritance/tests.py b/tests/model_inheritance/tests.py
--- a/tests/model_inheritance/tests.py
+++ b/tests/model_inheritance/tests.py
@@ -7,7 +7,7 @@
 
 from .models import (
     Base, Chef, CommonInfo, GrandChild, GrandParent, ItalianRestaurant,
-    MixinModel, ParkingLot, Place, Post, Restaurant, Student, SubBase,
+    MixinModel, Parent, ParkingLot, Place, Post, Restaurant, Student, SubBase,
     Supplier, Title, Worker,
 )
 
@@ -204,6 +204,19 @@ class A(models.Model):
 
         self.assertEqual(A.attr.called, (A, 'attr'))
 
+    def test_inherited_ordering_pk_desc(self):
+        p1 = Parent.objects.create(first_name='Joe', email='joe@email.com')
+        p2 = Parent.objects.create(first_name='Jon', email='jon@email.com')
+        expected_order_by_sql = 'ORDER BY %s.%s DESC' % (
+            connection.ops.quote_name(Parent._meta.db_table),
+            connection.ops.quote_name(
+                Parent._meta.get_field('grandparent_ptr').column
+            ),
+        )
+        qs = Parent.objects.all()
+        self.assertSequenceEqual(qs, [p2, p1])
+        self.assertIn(expected_order_by_sql, str(qs.query))
+
 
 class ModelInheritanceDataTests(TestCase):
     @classmethod

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 model_inheritance.models model_inheritance.tests
: '>>>>> End Test Output'
git checkout 142ab6846ac09d6d401e26fc8b6b988a583ac0f5 tests/model_inheritance/models.py tests/model_inheritance/tests.py
