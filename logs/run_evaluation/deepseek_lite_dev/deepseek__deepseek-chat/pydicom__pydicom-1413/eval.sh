#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff f909c76e31f759246cec3708dadd173c5d6e84b1
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout f909c76e31f759246cec3708dadd173c5d6e84b1 pydicom/tests/test_valuerep.py
git apply -v - <<'EOF_114329324912'
diff --git a/pydicom/tests/test_valuerep.py b/pydicom/tests/test_valuerep.py
--- a/pydicom/tests/test_valuerep.py
+++ b/pydicom/tests/test_valuerep.py
@@ -1546,3 +1546,16 @@ def test_set_value(vr, pytype, vm0, vmN, keyword):
     elem = ds[keyword]
     assert elem.value == list(vmN)
     assert list(vmN) == elem.value
+
+
+@pytest.mark.parametrize("vr, pytype, vm0, vmN, keyword", VALUE_REFERENCE)
+def test_assigning_bytes(vr, pytype, vm0, vmN, keyword):
+    """Test that byte VRs are excluded from the backslash check."""
+    if pytype == bytes:
+        ds = Dataset()
+        value = b"\x00\x01" + b"\\" + b"\x02\x03"
+        setattr(ds, keyword, value)
+        elem = ds[keyword]
+        assert elem.VR == vr
+        assert elem.value == value
+        assert elem.VM == 1

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA pydicom/tests/test_valuerep.py
: '>>>>> End Test Output'
git checkout f909c76e31f759246cec3708dadd173c5d6e84b1 pydicom/tests/test_valuerep.py
