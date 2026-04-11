#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 2c01ebb4be5d53cbf6450f356c10e436025d6d07
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 2c01ebb4be5d53cbf6450f356c10e436025d6d07 tests/forms_tests/field_tests/test_multivaluefield.py tests/forms_tests/field_tests/test_splitdatetimefield.py tests/postgres_tests/test_ranges.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/forms_tests/field_tests/test_multivaluefield.py b/tests/forms_tests/field_tests/test_multivaluefield.py
--- a/tests/forms_tests/field_tests/test_multivaluefield.py
+++ b/tests/forms_tests/field_tests/test_multivaluefield.py
@@ -141,7 +141,7 @@ def test_form_as_table(self):
         self.assertHTMLEqual(
             form.as_table(),
             """
-            <tr><th><label for="id_field1_0">Field1:</label></th>
+            <tr><th><label>Field1:</label></th>
             <td><input type="text" name="field1_0" id="id_field1_0" required>
             <select multiple name="field1_1" id="id_field1_1" required>
             <option value="J">John</option>
@@ -164,7 +164,7 @@ def test_form_as_table_data(self):
         self.assertHTMLEqual(
             form.as_table(),
             """
-            <tr><th><label for="id_field1_0">Field1:</label></th>
+            <tr><th><label>Field1:</label></th>
             <td><input type="text" name="field1_0" value="some text" id="id_field1_0" required>
             <select multiple name="field1_1" id="id_field1_1" required>
             <option value="J" selected>John</option>
diff --git a/tests/forms_tests/field_tests/test_splitdatetimefield.py b/tests/forms_tests/field_tests/test_splitdatetimefield.py
--- a/tests/forms_tests/field_tests/test_splitdatetimefield.py
+++ b/tests/forms_tests/field_tests/test_splitdatetimefield.py
@@ -1,7 +1,7 @@
 import datetime
 
 from django.core.exceptions import ValidationError
-from django.forms import SplitDateTimeField
+from django.forms import Form, SplitDateTimeField
 from django.forms.widgets import SplitDateTimeWidget
 from django.test import SimpleTestCase
 
@@ -60,3 +60,16 @@ def test_splitdatetimefield_changed(self):
         self.assertTrue(f.has_changed(datetime.datetime(2008, 5, 6, 12, 40, 00), ['2008-05-06', '12:40:00']))
         self.assertFalse(f.has_changed(datetime.datetime(2008, 5, 6, 12, 40, 00), ['06/05/2008', '12:40']))
         self.assertTrue(f.has_changed(datetime.datetime(2008, 5, 6, 12, 40, 00), ['06/05/2008', '12:41']))
+
+    def test_form_as_table(self):
+        class TestForm(Form):
+            datetime = SplitDateTimeField()
+
+        f = TestForm()
+        self.assertHTMLEqual(
+            f.as_table(),
+            '<tr><th><label>Datetime:</label></th><td>'
+            '<input type="text" name="datetime_0" required id="id_datetime_0">'
+            '<input type="text" name="datetime_1" required id="id_datetime_1">'
+            '</td></tr>',
+        )
diff --git a/tests/postgres_tests/test_ranges.py b/tests/postgres_tests/test_ranges.py
--- a/tests/postgres_tests/test_ranges.py
+++ b/tests/postgres_tests/test_ranges.py
@@ -665,7 +665,7 @@ class SplitForm(forms.Form):
         self.assertHTMLEqual(str(form), '''
             <tr>
                 <th>
-                <label for="id_field_0">Field:</label>
+                <label>Field:</label>
                 </th>
                 <td>
                     <input id="id_field_0_0" name="field_0_0" type="text">
@@ -700,7 +700,7 @@ class DateTimeRangeForm(forms.Form):
             form.as_table(),
             """
             <tr><th>
-            <label for="id_datetime_field_0">Datetime field:</label>
+            <label>Datetime field:</label>
             </th><td>
             <input type="text" name="datetime_field_0" id="id_datetime_field_0">
             <input type="text" name="datetime_field_1" id="id_datetime_field_1">
@@ -717,7 +717,7 @@ class DateTimeRangeForm(forms.Form):
             form.as_table(),
             """
             <tr><th>
-            <label for="id_datetime_field_0">Datetime field:</label>
+            <label>Datetime field:</label>
             </th><td>
             <input type="text" name="datetime_field_0"
             value="2010-01-01 11:13:00" id="id_datetime_field_0">
@@ -754,7 +754,7 @@ class RangeForm(forms.Form):
 
         self.assertHTMLEqual(str(RangeForm()), '''
         <tr>
-            <th><label for="id_ints_0">Ints:</label></th>
+            <th><label>Ints:</label></th>
             <td>
                 <input id="id_ints_0" name="ints_0" type="number">
                 <input id="id_ints_1" name="ints_1" type="number">

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 forms_tests.field_tests.test_multivaluefield forms_tests.field_tests.test_splitdatetimefield postgres_tests.test_ranges
: '>>>>> End Test Output'
git checkout 2c01ebb4be5d53cbf6450f356c10e436025d6d07 tests/forms_tests/field_tests/test_multivaluefield.py tests/forms_tests/field_tests/test_splitdatetimefield.py tests/postgres_tests/test_ranges.py
