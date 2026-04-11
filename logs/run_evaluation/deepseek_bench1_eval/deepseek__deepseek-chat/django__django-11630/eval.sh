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
git -c core.fileMode=false diff 65e86948b80262574058a94ccaae3a9b59c3faea
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 65e86948b80262574058a94ccaae3a9b59c3faea tests/check_framework/test_model_checks.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/check_framework/test_model_checks.py b/tests/check_framework/test_model_checks.py
--- a/tests/check_framework/test_model_checks.py
+++ b/tests/check_framework/test_model_checks.py
@@ -1,12 +1,16 @@
 from django.core import checks
-from django.core.checks import Error
+from django.core.checks import Error, Warning
 from django.db import models
 from django.test import SimpleTestCase, TestCase, skipUnlessDBFeature
 from django.test.utils import (
-    isolate_apps, modify_settings, override_system_checks,
+    isolate_apps, modify_settings, override_settings, override_system_checks,
 )
 
 
+class EmptyRouter:
+    pass
+
+
 @isolate_apps('check_framework', attr_name='apps')
 @override_system_checks([checks.model_checks.check_all_models])
 class DuplicateDBTableTests(SimpleTestCase):
@@ -28,6 +32,30 @@ class Meta:
             )
         ])
 
+    @override_settings(DATABASE_ROUTERS=['check_framework.test_model_checks.EmptyRouter'])
+    def test_collision_in_same_app_database_routers_installed(self):
+        class Model1(models.Model):
+            class Meta:
+                db_table = 'test_table'
+
+        class Model2(models.Model):
+            class Meta:
+                db_table = 'test_table'
+
+        self.assertEqual(checks.run_checks(app_configs=self.apps.get_app_configs()), [
+            Warning(
+                "db_table 'test_table' is used by multiple models: "
+                "check_framework.Model1, check_framework.Model2.",
+                hint=(
+                    'You have configured settings.DATABASE_ROUTERS. Verify '
+                    'that check_framework.Model1, check_framework.Model2 are '
+                    'correctly routed to separate databases.'
+                ),
+                obj='test_table',
+                id='models.W035',
+            )
+        ])
+
     @modify_settings(INSTALLED_APPS={'append': 'basic'})
     @isolate_apps('basic', 'check_framework', kwarg_name='apps')
     def test_collision_across_apps(self, apps):
@@ -50,6 +78,34 @@ class Meta:
             )
         ])
 
+    @modify_settings(INSTALLED_APPS={'append': 'basic'})
+    @override_settings(DATABASE_ROUTERS=['check_framework.test_model_checks.EmptyRouter'])
+    @isolate_apps('basic', 'check_framework', kwarg_name='apps')
+    def test_collision_across_apps_database_routers_installed(self, apps):
+        class Model1(models.Model):
+            class Meta:
+                app_label = 'basic'
+                db_table = 'test_table'
+
+        class Model2(models.Model):
+            class Meta:
+                app_label = 'check_framework'
+                db_table = 'test_table'
+
+        self.assertEqual(checks.run_checks(app_configs=apps.get_app_configs()), [
+            Warning(
+                "db_table 'test_table' is used by multiple models: "
+                "basic.Model1, check_framework.Model2.",
+                hint=(
+                    'You have configured settings.DATABASE_ROUTERS. Verify '
+                    'that basic.Model1, check_framework.Model2 are correctly '
+                    'routed to separate databases.'
+                ),
+                obj='test_table',
+                id='models.W035',
+            )
+        ])
+
     def test_no_collision_for_unmanaged_models(self):
         class Unmanaged(models.Model):
             class Meta:

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 check_framework.test_model_checks
: '>>>>> End Test Output'
git checkout 65e86948b80262574058a94ccaae3a9b59c3faea tests/check_framework/test_model_checks.py
