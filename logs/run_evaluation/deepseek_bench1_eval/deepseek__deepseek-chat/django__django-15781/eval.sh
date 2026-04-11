#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 8d160f154f0240a423e83ffe0690e472f837373c
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 8d160f154f0240a423e83ffe0690e472f837373c tests/user_commands/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/user_commands/tests.py b/tests/user_commands/tests.py
--- a/tests/user_commands/tests.py
+++ b/tests/user_commands/tests.py
@@ -1,4 +1,5 @@
 import os
+from argparse import ArgumentDefaultsHelpFormatter
 from io import StringIO
 from unittest import mock
 
@@ -408,8 +409,14 @@ def test_subparser_invalid_option(self):
     def test_create_parser_kwargs(self):
         """BaseCommand.create_parser() passes kwargs to CommandParser."""
         epilog = "some epilog text"
-        parser = BaseCommand().create_parser("prog_name", "subcommand", epilog=epilog)
+        parser = BaseCommand().create_parser(
+            "prog_name",
+            "subcommand",
+            epilog=epilog,
+            formatter_class=ArgumentDefaultsHelpFormatter,
+        )
         self.assertEqual(parser.epilog, epilog)
+        self.assertEqual(parser.formatter_class, ArgumentDefaultsHelpFormatter)
 
     def test_outputwrapper_flush(self):
         out = StringIO()

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 user_commands.tests
: '>>>>> End Test Output'
git checkout 8d160f154f0240a423e83ffe0690e472f837373c tests/user_commands/tests.py
