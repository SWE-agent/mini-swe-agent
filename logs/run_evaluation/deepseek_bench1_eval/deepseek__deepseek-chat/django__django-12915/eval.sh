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
git -c core.fileMode=false diff 4652f1f0aa459a7b980441d629648707c32e36bf
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 4652f1f0aa459a7b980441d629648707c32e36bf tests/asgi/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/asgi/project/static/file.txt b/tests/asgi/project/static/file.txt
new file mode 100644
--- /dev/null
+++ b/tests/asgi/project/static/file.txt
@@ -0,0 +1 @@
+test
diff --git a/tests/asgi/tests.py b/tests/asgi/tests.py
--- a/tests/asgi/tests.py
+++ b/tests/asgi/tests.py
@@ -1,18 +1,25 @@
 import asyncio
 import sys
 import threading
+from pathlib import Path
 from unittest import skipIf
 
 from asgiref.sync import SyncToAsync
 from asgiref.testing import ApplicationCommunicator
 
+from django.contrib.staticfiles.handlers import ASGIStaticFilesHandler
 from django.core.asgi import get_asgi_application
 from django.core.signals import request_finished, request_started
 from django.db import close_old_connections
-from django.test import AsyncRequestFactory, SimpleTestCase, override_settings
+from django.test import (
+    AsyncRequestFactory, SimpleTestCase, modify_settings, override_settings,
+)
+from django.utils.http import http_date
 
 from .urls import test_filename
 
+TEST_STATIC_ROOT = Path(__file__).parent / 'project' / 'static'
+
 
 @skipIf(sys.platform == 'win32' and (3, 8, 0) < sys.version_info < (3, 8, 1), 'https://bugs.python.org/issue38563')
 @override_settings(ROOT_URLCONF='asgi.urls')
@@ -79,6 +86,45 @@ async def test_file_response(self):
         # Allow response.close() to finish.
         await communicator.wait()
 
+    @modify_settings(INSTALLED_APPS={'append': 'django.contrib.staticfiles'})
+    @override_settings(
+        STATIC_URL='/static/',
+        STATIC_ROOT=TEST_STATIC_ROOT,
+        STATICFILES_DIRS=[TEST_STATIC_ROOT],
+        STATICFILES_FINDERS=[
+            'django.contrib.staticfiles.finders.FileSystemFinder',
+        ],
+    )
+    async def test_static_file_response(self):
+        application = ASGIStaticFilesHandler(get_asgi_application())
+        # Construct HTTP request.
+        scope = self.async_request_factory._base_scope(path='/static/file.txt')
+        communicator = ApplicationCommunicator(application, scope)
+        await communicator.send_input({'type': 'http.request'})
+        # Get the file content.
+        file_path = TEST_STATIC_ROOT / 'file.txt'
+        with open(file_path, 'rb') as test_file:
+            test_file_contents = test_file.read()
+        # Read the response.
+        stat = file_path.stat()
+        response_start = await communicator.receive_output()
+        self.assertEqual(response_start['type'], 'http.response.start')
+        self.assertEqual(response_start['status'], 200)
+        self.assertEqual(
+            set(response_start['headers']),
+            {
+                (b'Content-Length', str(len(test_file_contents)).encode('ascii')),
+                (b'Content-Type', b'text/plain'),
+                (b'Content-Disposition', b'inline; filename="file.txt"'),
+                (b'Last-Modified', http_date(stat.st_mtime).encode('ascii')),
+            },
+        )
+        response_body = await communicator.receive_output()
+        self.assertEqual(response_body['type'], 'http.response.body')
+        self.assertEqual(response_body['body'], test_file_contents)
+        # Allow response.close() to finish.
+        await communicator.wait()
+
     async def test_headers(self):
         application = get_asgi_application()
         communicator = ApplicationCommunicator(
diff --git a/tests/staticfiles_tests/test_handlers.py b/tests/staticfiles_tests/test_handlers.py
new file mode 100644
--- /dev/null
+++ b/tests/staticfiles_tests/test_handlers.py
@@ -0,0 +1,22 @@
+from django.contrib.staticfiles.handlers import ASGIStaticFilesHandler
+from django.core.handlers.asgi import ASGIHandler
+from django.test import AsyncRequestFactory
+
+from .cases import StaticFilesTestCase
+
+
+class TestASGIStaticFilesHandler(StaticFilesTestCase):
+    async_request_factory = AsyncRequestFactory()
+
+    async def test_get_async_response(self):
+        request = self.async_request_factory.get('/static/test/file.txt')
+        handler = ASGIStaticFilesHandler(ASGIHandler())
+        response = await handler.get_response_async(request)
+        response.close()
+        self.assertEqual(response.status_code, 200)
+
+    async def test_get_async_response_not_found(self):
+        request = self.async_request_factory.get('/static/test/not-found.txt')
+        handler = ASGIStaticFilesHandler(ASGIHandler())
+        response = await handler.get_response_async(request)
+        self.assertEqual(response.status_code, 404)

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 asgi.tests staticfiles_tests.test_handlers
: '>>>>> End Test Output'
git checkout 4652f1f0aa459a7b980441d629648707c32e36bf tests/asgi/tests.py
