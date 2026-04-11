#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 27a3a07ebc84b11014d3753e4923902adf9a38c0
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .[all]
git checkout 27a3a07ebc84b11014d3753e4923902adf9a38c0 pvlib/tests/test_pvsystem.py
git apply -v - <<'EOF_114329324912'
diff --git a/pvlib/tests/test_pvsystem.py b/pvlib/tests/test_pvsystem.py
--- a/pvlib/tests/test_pvsystem.py
+++ b/pvlib/tests/test_pvsystem.py
@@ -1887,8 +1887,6 @@ def test_PVSystem_multiple_array_creation():
     assert pv_system.arrays[0].module_parameters == {}
     assert pv_system.arrays[1].module_parameters == {'pdc0': 1}
     assert pv_system.arrays == (array_one, array_two)
-    with pytest.raises(TypeError):
-        pvsystem.PVSystem(arrays=array_one)
 
 
 def test_PVSystem_get_aoi():
@@ -2362,6 +2360,14 @@ def test_PVSystem_at_least_one_array():
         pvsystem.PVSystem(arrays=[])
 
 
+def test_PVSystem_single_array():
+    # GH 1831
+    single_array = pvsystem.Array(pvsystem.FixedMount())
+    system = pvsystem.PVSystem(arrays=single_array)
+    assert isinstance(system.arrays, tuple)
+    assert system.arrays[0] is single_array
+
+
 def test_combine_loss_factors():
     test_index = pd.date_range(start='1990/01/01T12:00', periods=365, freq='D')
     loss_1 = pd.Series(.10, index=test_index)

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA pvlib/tests/test_pvsystem.py
: '>>>>> End Test Output'
git checkout 27a3a07ebc84b11014d3753e4923902adf9a38c0 pvlib/tests/test_pvsystem.py
