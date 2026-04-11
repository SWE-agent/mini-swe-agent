#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff c78b50f4337ecbe536a961336ca91a1176efc0e8
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .[all]
git checkout c78b50f4337ecbe536a961336ca91a1176efc0e8 pvlib/tests/test_tools.py
git apply -v - <<'EOF_114329324912'
diff --git a/pvlib/tests/test_tools.py b/pvlib/tests/test_tools.py
--- a/pvlib/tests/test_tools.py
+++ b/pvlib/tests/test_tools.py
@@ -45,6 +45,22 @@ def test__golden_sect_DataFrame_vector():
     v, x = tools._golden_sect_DataFrame(params, lower, upper,
                                         _obj_test_golden_sect)
     assert np.allclose(x, expected, atol=1e-8)
+    # some upper and lower bounds equal
+    params = {'c': np.array([1., 2., 1.]), 'n': np.array([1., 1., 1.])}
+    lower = np.array([0., 0.001, 1.])
+    upper = np.array([1., 1.2, 1.])
+    expected = np.array([0.5, 0.25, 1.0])  # x values for maxima
+    v, x = tools._golden_sect_DataFrame(params, lower, upper,
+                                        _obj_test_golden_sect)
+    assert np.allclose(x, expected, atol=1e-8)
+    # all upper and lower bounds equal, arrays of length 1
+    params = {'c': np.array([1.]), 'n': np.array([1.])}
+    lower = np.array([1.])
+    upper = np.array([1.])
+    expected = np.array([1.])  # x values for maxima
+    v, x = tools._golden_sect_DataFrame(params, lower, upper,
+                                        _obj_test_golden_sect)
+    assert np.allclose(x, expected, atol=1e-8)
 
 
 def test__golden_sect_DataFrame_nans():

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA pvlib/tests/test_tools.py
: '>>>>> End Test Output'
git checkout c78b50f4337ecbe536a961336ca91a1176efc0e8 pvlib/tests/test_tools.py
