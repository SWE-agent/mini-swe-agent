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
git -c core.fileMode=false diff 49ae7ce50a874f8a04cd910882fb9571ff3a0d7a
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 49ae7ce50a874f8a04cd910882fb9571ff3a0d7a tests/queries/test_qs_combinators.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/queries/test_qs_combinators.py b/tests/queries/test_qs_combinators.py
--- a/tests/queries/test_qs_combinators.py
+++ b/tests/queries/test_qs_combinators.py
@@ -272,6 +272,7 @@ def test_unsupported_operations_on_combined_qs(self):
                 'annotate',
                 'defer',
                 'delete',
+                'distinct',
                 'exclude',
                 'extra',
                 'filter',

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 queries.test_qs_combinators
: '>>>>> End Test Output'
git checkout 49ae7ce50a874f8a04cd910882fb9571ff3a0d7a tests/queries/test_qs_combinators.py
