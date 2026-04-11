#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff a10057635e5b2559293a676486f0b730981f037a
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout a10057635e5b2559293a676486f0b730981f037a test/core/linter_test.py
git apply -v - <<'EOF_114329324912'
diff --git a/test/core/linter_test.py b/test/core/linter_test.py
--- a/test/core/linter_test.py
+++ b/test/core/linter_test.py
@@ -641,3 +641,56 @@ def test__attempt_to_change_templater_warning(caplog):
         assert "Attempt to set templater to " in caplog.text
     finally:
         logger.propagate = original_propagate_value
+
+
+@pytest.mark.parametrize(
+    "case",
+    [
+        dict(
+            name="utf8_create",
+            fname="test.sql",
+            encoding="utf-8",
+            existing=None,
+            update="def",
+            expected="def",
+        ),
+        dict(
+            name="utf8_update",
+            fname="test.sql",
+            encoding="utf-8",
+            existing="abc",
+            update="def",
+            expected="def",
+        ),
+        dict(
+            name="utf8_special_char",
+            fname="test.sql",
+            encoding="utf-8",
+            existing="abc",
+            update="→",  # Special utf-8 character
+            expected="→",
+        ),
+        dict(
+            name="incorrect_encoding",
+            fname="test.sql",
+            encoding="Windows-1252",
+            existing="abc",
+            update="→",  # Not valid in Windows-1252
+            expected="abc",  # File should be unchanged
+        ),
+    ],
+    ids=lambda case: case["name"],
+)
+def test_safe_create_replace_file(case, tmp_path):
+    """Test creating or updating .sql files, various content and encoding."""
+    p = tmp_path / case["fname"]
+    if case["existing"]:
+        p.write_text(case["existing"])
+    try:
+        linter.LintedFile._safe_create_replace_file(
+            str(p), case["update"], case["encoding"]
+        )
+    except:  # noqa: E722
+        pass
+    actual = p.read_text(encoding=case["encoding"])
+    assert case["expected"] == actual

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA test/core/linter_test.py
: '>>>>> End Test Output'
git checkout a10057635e5b2559293a676486f0b730981f037a test/core/linter_test.py
