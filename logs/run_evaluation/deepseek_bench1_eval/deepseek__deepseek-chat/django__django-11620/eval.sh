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
git -c core.fileMode=false diff 514efa3129792ec2abb2444f3e7aeb3f21a38386
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 514efa3129792ec2abb2444f3e7aeb3f21a38386 tests/view_tests/tests/test_debug.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/view_tests/tests/test_debug.py b/tests/view_tests/tests/test_debug.py
--- a/tests/view_tests/tests/test_debug.py
+++ b/tests/view_tests/tests/test_debug.py
@@ -12,11 +12,13 @@
 from django.core import mail
 from django.core.files.uploadedfile import SimpleUploadedFile
 from django.db import DatabaseError, connection
+from django.http import Http404
 from django.shortcuts import render
 from django.template import TemplateDoesNotExist
 from django.test import RequestFactory, SimpleTestCase, override_settings
 from django.test.utils import LoggingCaptureMixin
 from django.urls import path, reverse
+from django.urls.converters import IntConverter
 from django.utils.functional import SimpleLazyObject
 from django.utils.safestring import mark_safe
 from django.views.debug import (
@@ -237,6 +239,11 @@ def test_template_encoding(self):
             technical_404_response(mock.MagicMock(), mock.Mock())
             m.assert_called_once_with(encoding='utf-8')
 
+    def test_technical_404_converter_raise_404(self):
+        with mock.patch.object(IntConverter, 'to_python', side_effect=Http404):
+            response = self.client.get('/path-post/1/')
+            self.assertContains(response, 'Page not found', status_code=404)
+
 
 class DebugViewQueriesAllowedTests(SimpleTestCase):
     # May need a query to initialize MySQL connection

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 view_tests.tests.test_debug
: '>>>>> End Test Output'
git checkout 514efa3129792ec2abb2444f3e7aeb3f21a38386 tests/view_tests/tests/test_debug.py
