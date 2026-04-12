#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff f8cf45b6c121e5a4bf4a43f71aba3bc64af3db9c
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout f8cf45b6c121e5a4bf4a43f71aba3bc64af3db9c pydicom/tests/test_json.py
git apply -v - <<'EOF_114329324912'
diff --git a/pydicom/tests/test_json.py b/pydicom/tests/test_json.py
--- a/pydicom/tests/test_json.py
+++ b/pydicom/tests/test_json.py
@@ -7,7 +7,7 @@
 
 from pydicom import dcmread
 from pydicom.data import get_testdata_file
-from pydicom.dataelem import DataElement
+from pydicom.dataelem import DataElement, RawDataElement
 from pydicom.dataset import Dataset
 from pydicom.tag import Tag, BaseTag
 from pydicom.valuerep import PersonName
@@ -284,7 +284,23 @@ def test_suppress_invalid_tags(self, _):
 
         ds_json = ds.to_json_dict(suppress_invalid_tags=True)
 
-        assert ds_json.get("00100010") is None
+        assert "00100010" not in ds_json
+
+    def test_suppress_invalid_tags_with_failed_dataelement(self):
+        """Test tags that raise exceptions don't if suppress_invalid_tags True.
+        """
+        ds = Dataset()
+        # we have to add a RawDataElement as creating a DataElement would
+        # already raise an exception
+        ds[0x00082128] = RawDataElement(
+            Tag(0x00082128), 'IS', 4, b'5.25', 0, True, True)
+
+        with pytest.raises(TypeError):
+            ds.to_json_dict()
+
+        ds_json = ds.to_json_dict(suppress_invalid_tags=True)
+
+        assert "00082128" not in ds_json
 
 
 class TestSequence:

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA pydicom/tests/test_json.py
: '>>>>> End Test Output'
git checkout f8cf45b6c121e5a4bf4a43f71aba3bc64af3db9c pydicom/tests/test_json.py
