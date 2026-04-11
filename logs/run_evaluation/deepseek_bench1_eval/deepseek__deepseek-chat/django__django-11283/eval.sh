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
git -c core.fileMode=false diff 08a4ee06510ae45562c228eefbdcaac84bd38c7a
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 08a4ee06510ae45562c228eefbdcaac84bd38c7a tests/auth_tests/test_migrations.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/auth_tests/test_migrations.py b/tests/auth_tests/test_migrations.py
--- a/tests/auth_tests/test_migrations.py
+++ b/tests/auth_tests/test_migrations.py
@@ -4,6 +4,7 @@
 from django.contrib.auth.models import Permission, User
 from django.contrib.contenttypes.models import ContentType
 from django.test import TestCase
+from django.test.utils import captured_stdout
 
 from .models import Proxy, UserProxy
 
@@ -152,3 +153,27 @@ def test_user_keeps_same_permissions_after_migrating_backward(self):
         user = User._default_manager.get(pk=user.pk)
         for permission in [self.default_permission, self.custom_permission]:
             self.assertTrue(user.has_perm('auth_tests.' + permission.codename))
+
+    def test_migrate_with_existing_target_permission(self):
+        """
+        Permissions may already exist:
+
+        - Old workaround was to manually create permissions for proxy models.
+        - Model may have been concrete and then converted to proxy.
+
+        Output a reminder to audit relevant permissions.
+        """
+        proxy_model_content_type = ContentType.objects.get_for_model(Proxy, for_concrete_model=False)
+        Permission.objects.create(
+            content_type=proxy_model_content_type,
+            codename='add_proxy',
+            name='Can add proxy',
+        )
+        Permission.objects.create(
+            content_type=proxy_model_content_type,
+            codename='display_proxys',
+            name='May display proxys information',
+        )
+        with captured_stdout() as stdout:
+            update_proxy_permissions.update_proxy_model_permissions(apps, None)
+        self.assertIn('A problem arose migrating proxy model permissions', stdout.getvalue())

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 auth_tests.test_migrations
: '>>>>> End Test Output'
git checkout 08a4ee06510ae45562c228eefbdcaac84bd38c7a tests/auth_tests/test_migrations.py
