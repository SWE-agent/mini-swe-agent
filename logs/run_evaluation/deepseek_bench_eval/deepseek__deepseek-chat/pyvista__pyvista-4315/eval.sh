#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff db6ee8dd4a747b8864caae36c5d05883976a3ae5
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout db6ee8dd4a747b8864caae36c5d05883976a3ae5 tests/test_grid.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_grid.py b/tests/test_grid.py
--- a/tests/test_grid.py
+++ b/tests/test_grid.py
@@ -735,6 +735,21 @@ def test_create_rectilinear_grid_from_specs():
     assert grid.n_cells == 9 * 3 * 19
     assert grid.n_points == 10 * 4 * 20
     assert grid.bounds == (-10.0, 8.0, -10.0, 5.0, -10.0, 9.0)
+
+    # with Sequence
+    xrng = [0, 1]
+    yrng = [0, 1, 2]
+    zrng = [0, 1, 2, 3]
+    grid = pyvista.RectilinearGrid(xrng)
+    assert grid.n_cells == 1
+    assert grid.n_points == 2
+    grid = pyvista.RectilinearGrid(xrng, yrng)
+    assert grid.n_cells == 2
+    assert grid.n_points == 6
+    grid = pyvista.RectilinearGrid(xrng, yrng, zrng)
+    assert grid.n_cells == 6
+    assert grid.n_points == 24
+
     # 2D example
     cell_spacings = np.array([1.0, 1.0, 2.0, 2.0, 5.0, 10.0])
     x_coordinates = np.cumsum(cell_spacings)

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA tests/test_grid.py
: '>>>>> End Test Output'
git checkout db6ee8dd4a747b8864caae36c5d05883976a3ae5 tests/test_grid.py
