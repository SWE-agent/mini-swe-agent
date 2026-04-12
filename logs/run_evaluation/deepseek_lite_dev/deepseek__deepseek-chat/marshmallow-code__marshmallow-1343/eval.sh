#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 2be2d83a1a9a6d3d9b85804f3ab545cecc409bb0
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e '.[dev]'
git checkout 2be2d83a1a9a6d3d9b85804f3ab545cecc409bb0 tests/test_marshalling.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_marshalling.py b/tests/test_marshalling.py
--- a/tests/test_marshalling.py
+++ b/tests/test_marshalling.py
@@ -2,7 +2,7 @@
 
 import pytest
 
-from marshmallow import fields, Schema
+from marshmallow import fields, Schema, validates
 from marshmallow.marshalling import Marshaller, Unmarshaller, missing
 from marshmallow.exceptions import ValidationError
 
@@ -283,3 +283,24 @@ class TestSchema(Schema):
 
             assert result is None
             assert excinfo.value.messages == {'foo': {'_schema': ['Invalid input type.']}}
+
+    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/1342
+    def test_deserialize_wrong_nested_type_with_validates_method(self, unmarshal):
+        class TestSchema(Schema):
+            value = fields.String()
+
+            @validates('value')
+            def validate_value(self, value):
+                pass
+
+        data = {
+            'foo': 'not what we need'
+        }
+        fields_dict = {
+            'foo': fields.Nested(TestSchema, required=True)
+        }
+        with pytest.raises(ValidationError) as excinfo:
+            result = unmarshal.deserialize(data, fields_dict)
+
+            assert result is None
+            assert excinfo.value.messages == {'foo': {'_schema': ['Invalid input type.']}}

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA tests/test_marshalling.py
: '>>>>> End Test Output'
git checkout 2be2d83a1a9a6d3d9b85804f3ab545cecc409bb0 tests/test_marshalling.py
