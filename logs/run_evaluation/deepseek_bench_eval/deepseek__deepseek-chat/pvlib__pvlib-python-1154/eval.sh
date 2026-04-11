#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 0b8f24c265d76320067a5ee908a57d475cd1bb24
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .[all]
git checkout 0b8f24c265d76320067a5ee908a57d475cd1bb24 pvlib/tests/test_irradiance.py
git apply -v - <<'EOF_114329324912'
diff --git a/pvlib/tests/test_irradiance.py b/pvlib/tests/test_irradiance.py
--- a/pvlib/tests/test_irradiance.py
+++ b/pvlib/tests/test_irradiance.py
@@ -203,7 +203,7 @@ def test_reindl(irrad_data, ephem_data, dni_et):
         40, 180, irrad_data['dhi'], irrad_data['dni'], irrad_data['ghi'],
         dni_et, ephem_data['apparent_zenith'], ephem_data['azimuth'])
     # values from matlab 1.4 code
-    assert_allclose(result, [np.nan, 27.9412, 104.1317, 34.1663], atol=1e-4)
+    assert_allclose(result, [0., 27.9412, 104.1317, 34.1663], atol=1e-4)
 
 
 def test_king(irrad_data, ephem_data):

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA pvlib/tests/test_irradiance.py
: '>>>>> End Test Output'
git checkout 0b8f24c265d76320067a5ee908a57d475cd1bb24 pvlib/tests/test_irradiance.py
