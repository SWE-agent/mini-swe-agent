#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 4fd3044ca0135da903a70dfb66992293f529ecf1
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 4fd3044ca0135da903a70dfb66992293f529ecf1 tests/forms_tests/field_tests/test_urlfield.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/forms_tests/field_tests/test_urlfield.py b/tests/forms_tests/field_tests/test_urlfield.py
--- a/tests/forms_tests/field_tests/test_urlfield.py
+++ b/tests/forms_tests/field_tests/test_urlfield.py
@@ -100,6 +100,10 @@ def test_urlfield_clean_invalid(self):
             # even on domains that don't fail the domain label length check in
             # the regex.
             'http://%s' % ("X" * 200,),
+            # urlsplit() raises ValueError.
+            '////]@N.AN',
+            # Empty hostname.
+            '#@A.bO',
         ]
         msg = "'Enter a valid URL.'"
         for value in tests:

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 forms_tests.field_tests.test_urlfield
: '>>>>> End Test Output'
git checkout 4fd3044ca0135da903a70dfb66992293f529ecf1 tests/forms_tests/field_tests/test_urlfield.py
