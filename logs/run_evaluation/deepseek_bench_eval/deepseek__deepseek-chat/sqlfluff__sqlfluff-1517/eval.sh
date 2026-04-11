#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 304a197829f98e7425a46d872ada73176137e5ae
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 304a197829f98e7425a46d872ada73176137e5ae test/dialects/ansi_test.py
git apply -v - <<'EOF_114329324912'
diff --git a/test/dialects/ansi_test.py b/test/dialects/ansi_test.py
--- a/test/dialects/ansi_test.py
+++ b/test/dialects/ansi_test.py
@@ -3,7 +3,7 @@
 import pytest
 import logging
 
-from sqlfluff.core import FluffConfig, Linter
+from sqlfluff.core import FluffConfig, Linter, SQLParseError
 from sqlfluff.core.parser import Lexer
 
 
@@ -214,3 +214,29 @@ def test__dialect__ansi_parse_indented_joins(sql_string, indented_joins, meta_lo
         idx for idx, raw_seg in enumerate(parsed.tree.iter_raw_seg()) if raw_seg.is_meta
     )
     assert res_meta_locs == meta_loc
+
+
+@pytest.mark.parametrize(
+    "raw,expected_message",
+    [
+        (";;", "Line 1, Position 1: Found unparsable section: ';;'"),
+        ("select id from tbl;", ""),
+        ("select id from tbl;;", "Could not parse: ;"),
+        ("select id from tbl;;;;;;", "Could not parse: ;;;;;"),
+        ("select id from tbl;select id2 from tbl2;", ""),
+        (
+            "select id from tbl;;select id2 from tbl2;",
+            "Could not parse: ;select id2 from tbl2;",
+        ),
+    ],
+)
+def test__dialect__ansi_multiple_semicolons(raw: str, expected_message: str) -> None:
+    """Multiple semicolons should be properly handled."""
+    lnt = Linter()
+    parsed = lnt.parse_string(raw)
+
+    assert len(parsed.violations) == (1 if expected_message else 0)
+    if expected_message:
+        violation = parsed.violations[0]
+        assert isinstance(violation, SQLParseError)
+        assert violation.desc() == expected_message

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA test/dialects/ansi_test.py
: '>>>>> End Test Output'
git checkout 304a197829f98e7425a46d872ada73176137e5ae test/dialects/ansi_test.py
