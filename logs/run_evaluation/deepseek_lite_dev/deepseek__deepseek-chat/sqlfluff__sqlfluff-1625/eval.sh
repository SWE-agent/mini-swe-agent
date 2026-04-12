#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 14e1a23a3166b9a645a16de96f694c77a5d4abb7
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 14e1a23a3166b9a645a16de96f694c77a5d4abb7 test/cli/commands_test.py
git apply -v - <<'EOF_114329324912'
diff --git a/test/cli/commands_test.py b/test/cli/commands_test.py
--- a/test/cli/commands_test.py
+++ b/test/cli/commands_test.py
@@ -49,7 +49,7 @@ def invoke_assert_code(
 expected_output = """== [test/fixtures/linter/indentation_error_simple.sql] FAIL
 L:   2 | P:   4 | L003 | Indentation not hanging or a multiple of 4 spaces
 L:   5 | P:  10 | L010 | Keywords must be consistently upper case.
-L:   5 | P:  13 | L031 | Avoid using aliases in join condition
+L:   5 | P:  13 | L031 | Avoid aliases in from clauses and join conditions.
 """
 
 

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA test/cli/commands_test.py
: '>>>>> End Test Output'
git checkout 14e1a23a3166b9a645a16de96f694c77a5d4abb7 test/cli/commands_test.py
