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
git -c core.fileMode=false diff 50c3ac6fa9b7c8a94a6d1dc87edf775e3bc4d575
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 50c3ac6fa9b7c8a94a6d1dc87edf775e3bc4d575 tests/shell/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/shell/tests.py b/tests/shell/tests.py
--- a/tests/shell/tests.py
+++ b/tests/shell/tests.py
@@ -9,6 +9,13 @@
 
 
 class ShellCommandTestCase(SimpleTestCase):
+    script_globals = 'print("__name__" in globals())'
+    script_with_inline_function = (
+        'import django\n'
+        'def f():\n'
+        '    print(django.__version__)\n'
+        'f()'
+    )
 
     def test_command_option(self):
         with self.assertLogs('test', 'INFO') as cm:
@@ -21,6 +28,16 @@ def test_command_option(self):
             )
         self.assertEqual(cm.records[0].getMessage(), __version__)
 
+    def test_command_option_globals(self):
+        with captured_stdout() as stdout:
+            call_command('shell', command=self.script_globals)
+        self.assertEqual(stdout.getvalue().strip(), 'True')
+
+    def test_command_option_inline_function_call(self):
+        with captured_stdout() as stdout:
+            call_command('shell', command=self.script_with_inline_function)
+        self.assertEqual(stdout.getvalue().strip(), __version__)
+
     @unittest.skipIf(sys.platform == 'win32', "Windows select() doesn't support file descriptors.")
     @mock.patch('django.core.management.commands.shell.select')
     def test_stdin_read(self, select):
@@ -30,6 +47,30 @@ def test_stdin_read(self, select):
             call_command('shell')
         self.assertEqual(stdout.getvalue().strip(), '100')
 
+    @unittest.skipIf(
+        sys.platform == 'win32',
+        "Windows select() doesn't support file descriptors.",
+    )
+    @mock.patch('django.core.management.commands.shell.select')  # [1]
+    def test_stdin_read_globals(self, select):
+        with captured_stdin() as stdin, captured_stdout() as stdout:
+            stdin.write(self.script_globals)
+            stdin.seek(0)
+            call_command('shell')
+        self.assertEqual(stdout.getvalue().strip(), 'True')
+
+    @unittest.skipIf(
+        sys.platform == 'win32',
+        "Windows select() doesn't support file descriptors.",
+    )
+    @mock.patch('django.core.management.commands.shell.select')  # [1]
+    def test_stdin_read_inline_function_call(self, select):
+        with captured_stdin() as stdin, captured_stdout() as stdout:
+            stdin.write(self.script_with_inline_function)
+            stdin.seek(0)
+            call_command('shell')
+        self.assertEqual(stdout.getvalue().strip(), __version__)
+
     @mock.patch('django.core.management.commands.shell.select.select')  # [1]
     @mock.patch.dict('sys.modules', {'IPython': None})
     def test_shell_with_ipython_not_installed(self, select):

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 shell.tests
: '>>>>> End Test Output'
git checkout 50c3ac6fa9b7c8a94a6d1dc87edf775e3bc4d575 tests/shell/tests.py
