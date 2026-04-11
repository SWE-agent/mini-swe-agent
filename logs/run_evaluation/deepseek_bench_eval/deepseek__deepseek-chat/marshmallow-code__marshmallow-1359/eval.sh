#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff b40a0f4e33823e6d0f341f7e8684e359a99060d1
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e '.[dev]'
git checkout b40a0f4e33823e6d0f341f7e8684e359a99060d1 tests/test_fields.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_fields.py b/tests/test_fields.py
--- a/tests/test_fields.py
+++ b/tests/test_fields.py
@@ -169,6 +169,20 @@ class OtherSchema(MySchema):
         assert schema2.fields["foo"].key_field.root == schema2
         assert schema2.fields["foo"].value_field.root == schema2
 
+    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/1357
+    def test_datetime_list_inner_format(self, schema):
+        class MySchema(Schema):
+            foo = fields.List(fields.DateTime())
+            bar = fields.Tuple((fields.DateTime(),))
+
+            class Meta:
+                datetimeformat = "iso8601"
+                dateformat = "iso8601"
+
+        schema = MySchema()
+        assert schema.fields["foo"].inner.format == "iso8601"
+        assert schema.fields["bar"].tuple_fields[0].format == "iso8601"
+
 
 class TestMetadata:
     @pytest.mark.parametrize("FieldClass", ALL_FIELDS)

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA tests/test_fields.py
: '>>>>> End Test Output'
git checkout b40a0f4e33823e6d0f341f7e8684e359a99060d1 tests/test_fields.py
