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
git -c core.fileMode=false diff 879cc3da6249e920b8d54518a0ae06de835d7373
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 879cc3da6249e920b8d54518a0ae06de835d7373 tests/httpwrappers/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/httpwrappers/tests.py b/tests/httpwrappers/tests.py
--- a/tests/httpwrappers/tests.py
+++ b/tests/httpwrappers/tests.py
@@ -366,6 +366,10 @@ def test_non_string_content(self):
         r.content = 12345
         self.assertEqual(r.content, b'12345')
 
+    def test_memoryview_content(self):
+        r = HttpResponse(memoryview(b'memoryview'))
+        self.assertEqual(r.content, b'memoryview')
+
     def test_iter_content(self):
         r = HttpResponse(['abc', 'def', 'ghi'])
         self.assertEqual(r.content, b'abcdefghi')

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 httpwrappers.tests
: '>>>>> End Test Output'
git checkout 879cc3da6249e920b8d54518a0ae06de835d7373 tests/httpwrappers/tests.py
