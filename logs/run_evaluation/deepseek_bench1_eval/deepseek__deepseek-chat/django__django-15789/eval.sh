#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff d4d5427571b4bf3a21c902276c2a00215c2a37cc
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout d4d5427571b4bf3a21c902276c2a00215c2a37cc tests/utils_tests/test_html.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/utils_tests/test_html.py b/tests/utils_tests/test_html.py
--- a/tests/utils_tests/test_html.py
+++ b/tests/utils_tests/test_html.py
@@ -1,6 +1,7 @@
 import os
 from datetime import datetime
 
+from django.core.serializers.json import DjangoJSONEncoder
 from django.test import SimpleTestCase
 from django.utils.functional import lazystr
 from django.utils.html import (
@@ -211,6 +212,16 @@ def test_json_script(self):
             with self.subTest(arg=arg):
                 self.assertEqual(json_script(arg, "test_id"), expected)
 
+    def test_json_script_custom_encoder(self):
+        class CustomDjangoJSONEncoder(DjangoJSONEncoder):
+            def encode(self, o):
+                return '{"hello": "world"}'
+
+        self.assertHTMLEqual(
+            json_script({}, encoder=CustomDjangoJSONEncoder),
+            '<script type="application/json">{"hello": "world"}</script>',
+        )
+
     def test_json_script_without_id(self):
         self.assertHTMLEqual(
             json_script({"key": "value"}),

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 utils_tests.test_html
: '>>>>> End Test Output'
git checkout d4d5427571b4bf3a21c902276c2a00215c2a37cc tests/utils_tests/test_html.py
