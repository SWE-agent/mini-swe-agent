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
git -c core.fileMode=false diff c5e373d48cbdd923575956fed477b63d66d9603f
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout c5e373d48cbdd923575956fed477b63d66d9603f tests/model_fields/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/model_fields/tests.py b/tests/model_fields/tests.py
--- a/tests/model_fields/tests.py
+++ b/tests/model_fields/tests.py
@@ -178,6 +178,19 @@ def get_foo_bar_display(self):
         f = FooBar(foo_bar=1)
         self.assertEqual(f.get_foo_bar_display(), 'something')
 
+    def test_overriding_inherited_FIELD_display(self):
+        class Base(models.Model):
+            foo = models.CharField(max_length=254, choices=[('A', 'Base A')])
+
+            class Meta:
+                abstract = True
+
+        class Child(Base):
+            foo = models.CharField(max_length=254, choices=[('A', 'Child A'), ('B', 'Child B')])
+
+        self.assertEqual(Child(foo='A').get_foo_display(), 'Child A')
+        self.assertEqual(Child(foo='B').get_foo_display(), 'Child B')
+
     def test_iterator_choices(self):
         """
         get_choices() works with Iterators.

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 model_fields.tests
: '>>>>> End Test Output'
git checkout c5e373d48cbdd923575956fed477b63d66d9603f tests/model_fields/tests.py
