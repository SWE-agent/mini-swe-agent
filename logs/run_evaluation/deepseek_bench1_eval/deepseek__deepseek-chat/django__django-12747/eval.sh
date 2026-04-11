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
git -c core.fileMode=false diff c86201b6ed4f8256b0a0520c08aa674f623d4127
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout c86201b6ed4f8256b0a0520c08aa674f623d4127 tests/delete/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/delete/tests.py b/tests/delete/tests.py
--- a/tests/delete/tests.py
+++ b/tests/delete/tests.py
@@ -522,11 +522,10 @@ def test_queryset_delete_returns_num_rows(self):
         existed_objs = {
             R._meta.label: R.objects.count(),
             HiddenUser._meta.label: HiddenUser.objects.count(),
-            A._meta.label: A.objects.count(),
-            MR._meta.label: MR.objects.count(),
             HiddenUserProfile._meta.label: HiddenUserProfile.objects.count(),
         }
         deleted, deleted_objs = R.objects.all().delete()
+        self.assertCountEqual(deleted_objs.keys(), existed_objs.keys())
         for k, v in existed_objs.items():
             self.assertEqual(deleted_objs[k], v)
 
@@ -550,13 +549,13 @@ def test_model_delete_returns_num_rows(self):
         existed_objs = {
             R._meta.label: R.objects.count(),
             HiddenUser._meta.label: HiddenUser.objects.count(),
-            A._meta.label: A.objects.count(),
             MR._meta.label: MR.objects.count(),
             HiddenUserProfile._meta.label: HiddenUserProfile.objects.count(),
             M.m2m.through._meta.label: M.m2m.through.objects.count(),
         }
         deleted, deleted_objs = r.delete()
         self.assertEqual(deleted, sum(existed_objs.values()))
+        self.assertCountEqual(deleted_objs.keys(), existed_objs.keys())
         for k, v in existed_objs.items():
             self.assertEqual(deleted_objs[k], v)
 
@@ -694,7 +693,7 @@ def test_fast_delete_empty_no_update_can_self_select(self):
         with self.assertNumQueries(1):
             self.assertEqual(
                 User.objects.filter(avatar__desc='missing').delete(),
-                (0, {'delete.User': 0})
+                (0, {}),
             )
 
     def test_fast_delete_combined_relationships(self):

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 delete.tests
: '>>>>> End Test Output'
git checkout c86201b6ed4f8256b0a0520c08aa674f623d4127 tests/delete/tests.py
