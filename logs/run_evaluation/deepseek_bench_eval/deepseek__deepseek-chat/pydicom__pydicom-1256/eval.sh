#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 49a3da4a3d9c24d7e8427a25048a1c7d5c4f7724
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 49a3da4a3d9c24d7e8427a25048a1c7d5c4f7724 pydicom/tests/test_json.py
git apply -v - <<'EOF_114329324912'
diff --git a/pydicom/tests/test_json.py b/pydicom/tests/test_json.py
--- a/pydicom/tests/test_json.py
+++ b/pydicom/tests/test_json.py
@@ -354,3 +354,25 @@ def bulk_data_reader(tag, vr, value):
         ds = Dataset().from_json(json.dumps(json_data), bulk_data_reader)
 
         assert b'xyzzy' == ds[0x00091002].value
+
+    def test_bulk_data_reader_is_called_within_SQ(self):
+        def bulk_data_reader(_):
+            return b'xyzzy'
+
+        json_data = {
+            "003a0200": {
+                "vr": "SQ", 
+                "Value": [
+                    {
+                        "54001010": {
+                            "vr": "OW",
+                            "BulkDataURI": "https://a.dummy.url"
+                        }
+                    }
+                ]
+            }
+        }
+
+        ds = Dataset().from_json(json.dumps(json_data), bulk_data_reader)
+
+        assert b'xyzzy' == ds[0x003a0200].value[0][0x54001010].value

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA pydicom/tests/test_json.py
: '>>>>> End Test Output'
git checkout 49a3da4a3d9c24d7e8427a25048a1c7d5c4f7724 pydicom/tests/test_json.py
