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
git -c core.fileMode=false diff fee75d2aed4e58ada6567c464cfd22e89dc65f4a
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout fee75d2aed4e58ada6567c464cfd22e89dc65f4a tests/invalid_models_tests/test_ordinary_fields.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/invalid_models_tests/test_ordinary_fields.py b/tests/invalid_models_tests/test_ordinary_fields.py
--- a/tests/invalid_models_tests/test_ordinary_fields.py
+++ b/tests/invalid_models_tests/test_ordinary_fields.py
@@ -304,6 +304,32 @@ class Model(models.Model):
 
         self.assertEqual(Model._meta.get_field('field').check(), [])
 
+    def test_choices_in_max_length(self):
+        class Model(models.Model):
+            field = models.CharField(
+                max_length=2, choices=[
+                    ('ABC', 'Value Too Long!'), ('OK', 'Good')
+                ],
+            )
+            group = models.CharField(
+                max_length=2, choices=[
+                    ('Nested', [('OK', 'Good'), ('Longer', 'Longer')]),
+                    ('Grouped', [('Bad', 'Bad')]),
+                ],
+            )
+
+        for name, choice_max_length in (('field', 3), ('group', 6)):
+            with self.subTest(name):
+                field = Model._meta.get_field(name)
+                self.assertEqual(field.check(), [
+                    Error(
+                        "'max_length' is too small to fit the longest value "
+                        "in 'choices' (%d characters)." % choice_max_length,
+                        obj=field,
+                        id='fields.E009',
+                    ),
+                ])
+
     def test_bad_db_index_value(self):
         class Model(models.Model):
             field = models.CharField(max_length=10, db_index='bad')

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 invalid_models_tests.test_ordinary_fields
: '>>>>> End Test Output'
git checkout fee75d2aed4e58ada6567c464cfd22e89dc65f4a tests/invalid_models_tests/test_ordinary_fields.py
