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
git -c core.fileMode=false diff 17455e924e243e7a55e8a38f45966d8cbb27c273
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 17455e924e243e7a55e8a38f45966d8cbb27c273 tests/model_fields/test_durationfield.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/model_fields/test_durationfield.py b/tests/model_fields/test_durationfield.py
--- a/tests/model_fields/test_durationfield.py
+++ b/tests/model_fields/test_durationfield.py
@@ -75,7 +75,7 @@ def test_invalid_string(self):
         self.assertEqual(
             cm.exception.message % cm.exception.params,
             "'not a datetime' value has an invalid format. "
-            "It must be in [DD] [HH:[MM:]]ss[.uuuuuu] format."
+            "It must be in [DD] [[HH:]MM:]ss[.uuuuuu] format."
         )
 
 

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 model_fields.test_durationfield
: '>>>>> End Test Output'
git checkout 17455e924e243e7a55e8a38f45966d8cbb27c273 tests/model_fields/test_durationfield.py
