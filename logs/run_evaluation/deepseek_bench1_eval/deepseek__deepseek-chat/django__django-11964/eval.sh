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
git -c core.fileMode=false diff fc2b1cc926e34041953738e58fa6ad3053059b22
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout fc2b1cc926e34041953738e58fa6ad3053059b22 tests/model_enums/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/model_enums/tests.py b/tests/model_enums/tests.py
--- a/tests/model_enums/tests.py
+++ b/tests/model_enums/tests.py
@@ -143,6 +143,12 @@ class Fruit(models.IntegerChoices):
                 APPLE = 1, 'Apple'
                 PINEAPPLE = 1, 'Pineapple'
 
+    def test_str(self):
+        for test in [Gender, Suit, YearInSchool, Vehicle]:
+            for member in test:
+                with self.subTest(member=member):
+                    self.assertEqual(str(test[member.name]), str(member.value))
+
 
 class Separator(bytes, models.Choices):
     FS = b'\x1c', 'File Separator'

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 model_enums.tests
: '>>>>> End Test Output'
git checkout fc2b1cc926e34041953738e58fa6ad3053059b22 tests/model_enums/tests.py
