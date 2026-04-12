#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 40e9e978c170bdde4eeee1547729417665dbc34c
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .[all]
git checkout 40e9e978c170bdde4eeee1547729417665dbc34c pvlib/tests/test_iam.py
git apply -v - <<'EOF_114329324912'
diff --git a/pvlib/tests/test_iam.py b/pvlib/tests/test_iam.py
--- a/pvlib/tests/test_iam.py
+++ b/pvlib/tests/test_iam.py
@@ -51,6 +51,18 @@ def test_physical():
     assert_series_equal(iam, expected)
 
 
+def test_physical_n1_L0():
+    aoi = np.array([0, 22.5, 45, 67.5, 90, 100, np.nan])
+    expected = np.array([1, 1, 1, 1, 0, 0, np.nan])
+    iam = _iam.physical(aoi, n=1, L=0)
+    assert_allclose(iam, expected, equal_nan=True)
+
+    aoi = pd.Series(aoi)
+    expected = pd.Series(expected)
+    iam = _iam.physical(aoi, n=1, L=0)
+    assert_series_equal(iam, expected)
+
+
 def test_physical_ar():
     aoi = np.array([0, 22.5, 45, 67.5, 90, 100, np.nan])
     expected = np.array([1, 0.99944171, 0.9917463, 0.91506158, 0, 0, np.nan])

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA pvlib/tests/test_iam.py
: '>>>>> End Test Output'
git checkout 40e9e978c170bdde4eeee1547729417665dbc34c pvlib/tests/test_iam.py
