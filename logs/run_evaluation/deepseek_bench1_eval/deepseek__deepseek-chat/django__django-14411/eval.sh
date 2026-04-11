#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff fa4e963ee7e6876581b5432363603571839ba00c
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout fa4e963ee7e6876581b5432363603571839ba00c tests/auth_tests/test_forms.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/auth_tests/test_forms.py b/tests/auth_tests/test_forms.py
--- a/tests/auth_tests/test_forms.py
+++ b/tests/auth_tests/test_forms.py
@@ -13,6 +13,7 @@
 from django.core import mail
 from django.core.exceptions import ValidationError
 from django.core.mail import EmailMultiAlternatives
+from django.forms import forms
 from django.forms.fields import CharField, Field, IntegerField
 from django.test import SimpleTestCase, TestCase, override_settings
 from django.utils import translation
@@ -1025,6 +1026,18 @@ def test_readonly_field_has_changed(self):
         self.assertIs(field.disabled, True)
         self.assertFalse(field.has_changed('aaa', 'bbb'))
 
+    def test_label(self):
+        """
+        ReadOnlyPasswordHashWidget doesn't contain a for attribute in the
+        <label> because it doesn't have any labelable elements.
+        """
+        class TestForm(forms.Form):
+            hash_field = ReadOnlyPasswordHashField()
+
+        bound_field = TestForm()['hash_field']
+        self.assertEqual(bound_field.field.widget.id_for_label('id'), None)
+        self.assertEqual(bound_field.label_tag(), '<label>Hash field:</label>')
+
 
 class AdminPasswordChangeFormTest(TestDataMixin, TestCase):
 

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 auth_tests.test_forms
: '>>>>> End Test Output'
git checkout fa4e963ee7e6876581b5432363603571839ba00c tests/auth_tests/test_forms.py
