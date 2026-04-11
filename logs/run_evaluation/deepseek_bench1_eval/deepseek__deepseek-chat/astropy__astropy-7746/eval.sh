#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff d5bd3f68bb6d5ce3a61bdce9883ee750d1afade5
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .[test] --verbose
git checkout d5bd3f68bb6d5ce3a61bdce9883ee750d1afade5 astropy/wcs/tests/test_wcs.py
git apply -v - <<'EOF_114329324912'
diff --git a/astropy/wcs/tests/test_wcs.py b/astropy/wcs/tests/test_wcs.py
--- a/astropy/wcs/tests/test_wcs.py
+++ b/astropy/wcs/tests/test_wcs.py
@@ -1093,3 +1093,21 @@ def test_keyedsip():
     assert isinstance( w.sip, wcs.Sip )
     assert w.sip.crpix[0] == 2048
     assert w.sip.crpix[1] == 1026
+
+
+def test_zero_size_input():
+    with fits.open(get_pkg_data_filename('data/sip.fits')) as f:
+        w = wcs.WCS(f[0].header)
+
+    inp = np.zeros((0, 2))
+    assert_array_equal(inp, w.all_pix2world(inp, 0))
+    assert_array_equal(inp, w.all_world2pix(inp, 0))
+
+    inp = [], [1]
+    result = w.all_pix2world([], [1], 0)
+    assert_array_equal(inp[0], result[0])
+    assert_array_equal(inp[1], result[1])
+
+    result = w.all_world2pix([], [1], 0)
+    assert_array_equal(inp[0], result[0])
+    assert_array_equal(inp[1], result[1])

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA -vv -o console_output_style=classic --tb=no astropy/wcs/tests/test_wcs.py
: '>>>>> End Test Output'
git checkout d5bd3f68bb6d5ce3a61bdce9883ee750d1afade5 astropy/wcs/tests/test_wcs.py
