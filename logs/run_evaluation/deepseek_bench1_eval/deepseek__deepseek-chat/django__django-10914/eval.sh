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
git -c core.fileMode=false diff e7fd69d051eaa67cb17f172a39b57253e9cb831a
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout e7fd69d051eaa67cb17f172a39b57253e9cb831a tests/test_utils/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_utils/tests.py b/tests/test_utils/tests.py
--- a/tests/test_utils/tests.py
+++ b/tests/test_utils/tests.py
@@ -1099,7 +1099,7 @@ def test_override_file_upload_permissions(self):
         the file_permissions_mode attribute of
         django.core.files.storage.default_storage.
         """
-        self.assertIsNone(default_storage.file_permissions_mode)
+        self.assertEqual(default_storage.file_permissions_mode, 0o644)
         with self.settings(FILE_UPLOAD_PERMISSIONS=0o777):
             self.assertEqual(default_storage.file_permissions_mode, 0o777)
 

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 test_utils.tests
: '>>>>> End Test Output'
git checkout e7fd69d051eaa67cb17f172a39b57253e9cb831a tests/test_utils/tests.py
