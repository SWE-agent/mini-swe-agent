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
git -c core.fileMode=false diff a4881f5e5d7ee38b7e83301331a0b4962845ef8a
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout a4881f5e5d7ee38b7e83301331a0b4962845ef8a tests/invalid_models_tests/test_relative_fields.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/invalid_models_tests/test_relative_fields.py b/tests/invalid_models_tests/test_relative_fields.py
--- a/tests/invalid_models_tests/test_relative_fields.py
+++ b/tests/invalid_models_tests/test_relative_fields.py
@@ -128,7 +128,36 @@ class ThroughModel(models.Model):
             ),
         ])
 
-    def test_ambiguous_relationship_model(self):
+    def test_ambiguous_relationship_model_from(self):
+        class Person(models.Model):
+            pass
+
+        class Group(models.Model):
+            field = models.ManyToManyField('Person', through='AmbiguousRelationship')
+
+        class AmbiguousRelationship(models.Model):
+            person = models.ForeignKey(Person, models.CASCADE)
+            first_group = models.ForeignKey(Group, models.CASCADE, related_name='first')
+            second_group = models.ForeignKey(Group, models.CASCADE, related_name='second')
+
+        field = Group._meta.get_field('field')
+        self.assertEqual(field.check(from_model=Group), [
+            Error(
+                "The model is used as an intermediate model by "
+                "'invalid_models_tests.Group.field', but it has more than one "
+                "foreign key from 'Group', which is ambiguous. You must "
+                "specify which foreign key Django should use via the "
+                "through_fields keyword argument.",
+                hint=(
+                    'If you want to create a recursive relationship, use '
+                    'ManyToManyField("self", through="AmbiguousRelationship").'
+                ),
+                obj=field,
+                id='fields.E334',
+            ),
+        ])
+
+    def test_ambiguous_relationship_model_to(self):
 
         class Person(models.Model):
             pass
@@ -152,7 +181,7 @@ class AmbiguousRelationship(models.Model):
                 "keyword argument.",
                 hint=(
                     'If you want to create a recursive relationship, use '
-                    'ForeignKey("self", symmetrical=False, through="AmbiguousRelationship").'
+                    'ManyToManyField("self", through="AmbiguousRelationship").'
                 ),
                 obj=field,
                 id='fields.E335',

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 invalid_models_tests.test_relative_fields
: '>>>>> End Test Output'
git checkout a4881f5e5d7ee38b7e83301331a0b4962845ef8a tests/invalid_models_tests/test_relative_fields.py
