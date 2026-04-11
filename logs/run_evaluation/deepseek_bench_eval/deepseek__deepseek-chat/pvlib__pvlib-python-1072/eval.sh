#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 04a523fafbd61bc2e49420963b84ed8e2bd1b3cf
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .[all]
git checkout 04a523fafbd61bc2e49420963b84ed8e2bd1b3cf pvlib/tests/test_temperature.py
git apply -v - <<'EOF_114329324912'
diff --git a/pvlib/tests/test_temperature.py b/pvlib/tests/test_temperature.py
--- a/pvlib/tests/test_temperature.py
+++ b/pvlib/tests/test_temperature.py
@@ -190,3 +190,17 @@ def test_fuentes(filename, inoct):
     night_difference = expected_tcell[is_night] - actual_tcell[is_night]
     assert night_difference.max() < 6
     assert night_difference.min() > 0
+
+
+@pytest.mark.parametrize('tz', [None, 'Etc/GMT+5'])
+def test_fuentes_timezone(tz):
+    index = pd.date_range('2019-01-01', freq='h', periods=3, tz=tz)
+
+    df = pd.DataFrame({'poa_global': 1000, 'temp_air': 20, 'wind_speed': 1},
+                      index)
+
+    out = temperature.fuentes(df['poa_global'], df['temp_air'],
+                              df['wind_speed'], noct_installed=45)
+
+    assert_series_equal(out, pd.Series([47.85, 50.85, 50.85], index=index,
+                                       name='tmod'))

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA pvlib/tests/test_temperature.py
: '>>>>> End Test Output'
git checkout 04a523fafbd61bc2e49420963b84ed8e2bd1b3cf pvlib/tests/test_temperature.py
