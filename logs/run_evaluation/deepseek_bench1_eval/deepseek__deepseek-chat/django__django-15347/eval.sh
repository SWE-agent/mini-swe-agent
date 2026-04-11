#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 7c4f3965098baad2396e24501e09237425a7bd6f
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 7c4f3965098baad2396e24501e09237425a7bd6f tests/messages_tests/test_cookie.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/messages_tests/test_cookie.py b/tests/messages_tests/test_cookie.py
--- a/tests/messages_tests/test_cookie.py
+++ b/tests/messages_tests/test_cookie.py
@@ -52,6 +52,12 @@ class CookieTests(BaseTests, SimpleTestCase):
     def stored_messages_count(self, storage, response):
         return stored_cookie_messages_count(storage, response)
 
+    def encode_decode(self, *args, **kwargs):
+        storage = self.get_storage()
+        message = Message(constants.DEBUG, *args, **kwargs)
+        encoded = storage._encode(message)
+        return storage._decode(encoded)
+
     def test_get(self):
         storage = self.storage_class(self.get_request())
         # Set initial data.
@@ -168,12 +174,23 @@ def test_safedata(self):
         A message containing SafeData is keeping its safe status when
         retrieved from the message storage.
         """
-        def encode_decode(data):
-            message = Message(constants.DEBUG, data)
-            encoded = storage._encode(message)
-            decoded = storage._decode(encoded)
-            return decoded.message
+        self.assertIsInstance(
+            self.encode_decode(mark_safe('<b>Hello Django!</b>')).message,
+            SafeData,
+        )
+        self.assertNotIsInstance(
+            self.encode_decode('<b>Hello Django!</b>').message,
+            SafeData,
+        )
 
-        storage = self.get_storage()
-        self.assertIsInstance(encode_decode(mark_safe("<b>Hello Django!</b>")), SafeData)
-        self.assertNotIsInstance(encode_decode("<b>Hello Django!</b>"), SafeData)
+    def test_extra_tags(self):
+        """
+        A message's extra_tags attribute is correctly preserved when retrieved
+        from the message storage.
+        """
+        for extra_tags in ['', None, 'some tags']:
+            with self.subTest(extra_tags=extra_tags):
+                self.assertEqual(
+                    self.encode_decode('message', extra_tags=extra_tags).extra_tags,
+                    extra_tags,
+                )

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 messages_tests.test_cookie
: '>>>>> End Test Output'
git checkout 7c4f3965098baad2396e24501e09237425a7bd6f tests/messages_tests/test_cookie.py
