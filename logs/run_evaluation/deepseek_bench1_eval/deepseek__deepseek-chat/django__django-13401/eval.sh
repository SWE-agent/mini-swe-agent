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
git -c core.fileMode=false diff 453967477e3ddae704cd739eac2449c0e13d464c
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 453967477e3ddae704cd739eac2449c0e13d464c tests/model_fields/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/model_fields/tests.py b/tests/model_fields/tests.py
--- a/tests/model_fields/tests.py
+++ b/tests/model_fields/tests.py
@@ -102,6 +102,36 @@ def test_deconstruct_nested_field(self):
         name, path, args, kwargs = Nested.Field().deconstruct()
         self.assertEqual(path, 'model_fields.tests.Nested.Field')
 
+    def test_abstract_inherited_fields(self):
+        """Field instances from abstract models are not equal."""
+        class AbstractModel(models.Model):
+            field = models.IntegerField()
+
+            class Meta:
+                abstract = True
+
+        class InheritAbstractModel1(AbstractModel):
+            pass
+
+        class InheritAbstractModel2(AbstractModel):
+            pass
+
+        abstract_model_field = AbstractModel._meta.get_field('field')
+        inherit1_model_field = InheritAbstractModel1._meta.get_field('field')
+        inherit2_model_field = InheritAbstractModel2._meta.get_field('field')
+
+        self.assertNotEqual(abstract_model_field, inherit1_model_field)
+        self.assertNotEqual(abstract_model_field, inherit2_model_field)
+        self.assertNotEqual(inherit1_model_field, inherit2_model_field)
+
+        self.assertLess(abstract_model_field, inherit1_model_field)
+        self.assertLess(abstract_model_field, inherit2_model_field)
+        self.assertLess(inherit1_model_field, inherit2_model_field)
+
+        self.assertNotEqual(hash(abstract_model_field), hash(inherit1_model_field))
+        self.assertNotEqual(hash(abstract_model_field), hash(inherit2_model_field))
+        self.assertNotEqual(hash(inherit1_model_field), hash(inherit2_model_field))
+
 
 class ChoicesTests(SimpleTestCase):
 

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 model_fields.tests
: '>>>>> End Test Output'
git checkout 453967477e3ddae704cd739eac2449c0e13d464c tests/model_fields/tests.py
