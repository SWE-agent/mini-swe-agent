#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en
export LC_ALL=en_US.UTF-8
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 3bc4240d979812bd11365ede04c028ea13fdc8c6
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 3bc4240d979812bd11365ede04c028ea13fdc8c6 tests/utils_tests/test_text.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/utils_tests/test_text.py b/tests/utils_tests/test_text.py
--- a/tests/utils_tests/test_text.py
+++ b/tests/utils_tests/test_text.py
@@ -192,6 +192,13 @@ def test_slugify(self):
             # given - expected - Unicode?
             ('Hello, World!', 'hello-world', False),
             ('spam & eggs', 'spam-eggs', False),
+            (' multiple---dash and  space ', 'multiple-dash-and-space', False),
+            ('\t whitespace-in-value \n', 'whitespace-in-value', False),
+            ('underscore_in-value', 'underscore_in-value', False),
+            ('__strip__underscore-value___', 'strip__underscore-value', False),
+            ('--strip-dash-value---', 'strip-dash-value', False),
+            ('__strip-mixed-value---', 'strip-mixed-value', False),
+            ('_ -strip-mixed-value _-', 'strip-mixed-value', False),
             ('spam & ıçüş', 'spam-ıçüş', True),
             ('foo ıç bar', 'foo-ıç-bar', True),
             ('    foo ıç bar', 'foo-ıç-bar', True),

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 utils_tests.test_text
: '>>>>> End Test Output'
git checkout 3bc4240d979812bd11365ede04c028ea13fdc8c6 tests/utils_tests/test_text.py
