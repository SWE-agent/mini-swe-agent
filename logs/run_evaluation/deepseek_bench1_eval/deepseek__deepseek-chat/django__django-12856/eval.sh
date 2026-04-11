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
git -c core.fileMode=false diff 8328811f048fed0dd22573224def8c65410c9f2e
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 8328811f048fed0dd22573224def8c65410c9f2e tests/invalid_models_tests/test_models.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/invalid_models_tests/test_models.py b/tests/invalid_models_tests/test_models.py
--- a/tests/invalid_models_tests/test_models.py
+++ b/tests/invalid_models_tests/test_models.py
@@ -1501,3 +1501,70 @@ class Meta:
                 ]
 
         self.assertEqual(Model.check(databases=self.databases), [])
+
+    def test_unique_constraint_pointing_to_missing_field(self):
+        class Model(models.Model):
+            class Meta:
+                constraints = [models.UniqueConstraint(fields=['missing_field'], name='name')]
+
+        self.assertEqual(Model.check(databases=self.databases), [
+            Error(
+                "'constraints' refers to the nonexistent field "
+                "'missing_field'.",
+                obj=Model,
+                id='models.E012',
+            ),
+        ])
+
+    def test_unique_constraint_pointing_to_m2m_field(self):
+        class Model(models.Model):
+            m2m = models.ManyToManyField('self')
+
+            class Meta:
+                constraints = [models.UniqueConstraint(fields=['m2m'], name='name')]
+
+        self.assertEqual(Model.check(databases=self.databases), [
+            Error(
+                "'constraints' refers to a ManyToManyField 'm2m', but "
+                "ManyToManyFields are not permitted in 'constraints'.",
+                obj=Model,
+                id='models.E013',
+            ),
+        ])
+
+    def test_unique_constraint_pointing_to_non_local_field(self):
+        class Parent(models.Model):
+            field1 = models.IntegerField()
+
+        class Child(Parent):
+            field2 = models.IntegerField()
+
+            class Meta:
+                constraints = [
+                    models.UniqueConstraint(fields=['field2', 'field1'], name='name'),
+                ]
+
+        self.assertEqual(Child.check(databases=self.databases), [
+            Error(
+                "'constraints' refers to field 'field1' which is not local to "
+                "model 'Child'.",
+                hint='This issue may be caused by multi-table inheritance.',
+                obj=Child,
+                id='models.E016',
+            ),
+        ])
+
+    def test_unique_constraint_pointing_to_fk(self):
+        class Target(models.Model):
+            pass
+
+        class Model(models.Model):
+            fk_1 = models.ForeignKey(Target, models.CASCADE, related_name='target_1')
+            fk_2 = models.ForeignKey(Target, models.CASCADE, related_name='target_2')
+
+            class Meta:
+                constraints = [
+                    models.UniqueConstraint(fields=['fk_1_id', 'fk_2'], name='name'),
+                ]
+
+        self.assertEqual(Model.check(databases=self.databases), [])

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 invalid_models_tests.test_models
: '>>>>> End Test Output'
git checkout 8328811f048fed0dd22573224def8c65410c9f2e tests/invalid_models_tests/test_models.py
