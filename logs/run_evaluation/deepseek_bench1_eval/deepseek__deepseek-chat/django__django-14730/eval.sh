#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 4fe3774c729f3fd5105b3001fe69a70bdca95ac3
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 4fe3774c729f3fd5105b3001fe69a70bdca95ac3 tests/field_deconstruction/tests.py tests/invalid_models_tests/test_relative_fields.py tests/model_meta/models.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/field_deconstruction/tests.py b/tests/field_deconstruction/tests.py
--- a/tests/field_deconstruction/tests.py
+++ b/tests/field_deconstruction/tests.py
@@ -438,7 +438,6 @@ class MyModel(models.Model):
             m2m = models.ManyToManyField('self')
             m2m_related_name = models.ManyToManyField(
                 'self',
-                related_name='custom_name',
                 related_query_name='custom_query_name',
                 limit_choices_to={'flag': True},
             )
@@ -455,7 +454,6 @@ class MyModel(models.Model):
         self.assertEqual(args, [])
         self.assertEqual(kwargs, {
             'to': 'field_deconstruction.MyModel',
-            'related_name': 'custom_name',
             'related_query_name': 'custom_query_name',
             'limit_choices_to': {'flag': True},
         })
diff --git a/tests/invalid_models_tests/test_relative_fields.py b/tests/invalid_models_tests/test_relative_fields.py
--- a/tests/invalid_models_tests/test_relative_fields.py
+++ b/tests/invalid_models_tests/test_relative_fields.py
@@ -128,6 +128,20 @@ class ThroughModel(models.Model):
             ),
         ])
 
+    def test_many_to_many_with_useless_related_name(self):
+        class ModelM2M(models.Model):
+            m2m = models.ManyToManyField('self', related_name='children')
+
+        field = ModelM2M._meta.get_field('m2m')
+        self.assertEqual(ModelM2M.check(), [
+            DjangoWarning(
+                'related_name has no effect on ManyToManyField with '
+                'a symmetrical relationship, e.g. to "self".',
+                obj=field,
+                id='fields.W345',
+            ),
+        ])
+
     def test_ambiguous_relationship_model_from(self):
         class Person(models.Model):
             pass
diff --git a/tests/model_meta/models.py b/tests/model_meta/models.py
--- a/tests/model_meta/models.py
+++ b/tests/model_meta/models.py
@@ -23,7 +23,7 @@ class AbstractPerson(models.Model):
 
     # M2M fields
     m2m_abstract = models.ManyToManyField(Relation, related_name='m2m_abstract_rel')
-    friends_abstract = models.ManyToManyField('self', related_name='friends_abstract', symmetrical=True)
+    friends_abstract = models.ManyToManyField('self', symmetrical=True)
     following_abstract = models.ManyToManyField('self', related_name='followers_abstract', symmetrical=False)
 
     # VIRTUAL fields
@@ -60,7 +60,7 @@ class BasePerson(AbstractPerson):
 
     # M2M fields
     m2m_base = models.ManyToManyField(Relation, related_name='m2m_base_rel')
-    friends_base = models.ManyToManyField('self', related_name='friends_base', symmetrical=True)
+    friends_base = models.ManyToManyField('self', symmetrical=True)
     following_base = models.ManyToManyField('self', related_name='followers_base', symmetrical=False)
 
     # VIRTUAL fields
@@ -88,7 +88,7 @@ class Person(BasePerson):
 
     # M2M Fields
     m2m_inherited = models.ManyToManyField(Relation, related_name='m2m_concrete_rel')
-    friends_inherited = models.ManyToManyField('self', related_name='friends_concrete', symmetrical=True)
+    friends_inherited = models.ManyToManyField('self', symmetrical=True)
     following_inherited = models.ManyToManyField('self', related_name='followers_concrete', symmetrical=False)
 
     # VIRTUAL fields

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 field_deconstruction.tests invalid_models_tests.test_relative_fields model_meta.models
: '>>>>> End Test Output'
git checkout 4fe3774c729f3fd5105b3001fe69a70bdca95ac3 tests/field_deconstruction/tests.py tests/invalid_models_tests/test_relative_fields.py tests/model_meta/models.py
