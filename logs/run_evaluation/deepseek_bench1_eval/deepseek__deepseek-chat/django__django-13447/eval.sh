#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 0456d3e42795481a186db05719300691fe2a1029
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 0456d3e42795481a186db05719300691fe2a1029 tests/admin_views/test_adminsite.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/admin_views/test_adminsite.py b/tests/admin_views/test_adminsite.py
--- a/tests/admin_views/test_adminsite.py
+++ b/tests/admin_views/test_adminsite.py
@@ -55,7 +55,9 @@ def test_available_apps(self):
         admin_views = apps[0]
         self.assertEqual(admin_views['app_label'], 'admin_views')
         self.assertEqual(len(admin_views['models']), 1)
-        self.assertEqual(admin_views['models'][0]['object_name'], 'Article')
+        article = admin_views['models'][0]
+        self.assertEqual(article['object_name'], 'Article')
+        self.assertEqual(article['model'], Article)
 
         # auth.User
         auth = apps[1]
@@ -63,6 +65,7 @@ def test_available_apps(self):
         self.assertEqual(len(auth['models']), 1)
         user = auth['models'][0]
         self.assertEqual(user['object_name'], 'User')
+        self.assertEqual(user['model'], User)
 
         self.assertEqual(auth['app_url'], '/test_admin/admin/auth/')
         self.assertIs(auth['has_module_perms'], True)

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 admin_views.test_adminsite
: '>>>>> End Test Output'
git checkout 0456d3e42795481a186db05719300691fe2a1029 tests/admin_views/test_adminsite.py
