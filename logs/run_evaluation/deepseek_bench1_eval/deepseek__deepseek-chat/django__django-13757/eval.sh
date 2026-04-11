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
git -c core.fileMode=false diff 3f140dde51c0fe6c350acb7727bbe489a99f0632
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 3f140dde51c0fe6c350acb7727bbe489a99f0632 tests/model_fields/test_jsonfield.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/model_fields/test_jsonfield.py b/tests/model_fields/test_jsonfield.py
--- a/tests/model_fields/test_jsonfield.py
+++ b/tests/model_fields/test_jsonfield.py
@@ -586,6 +586,10 @@ def test_isnull_key(self):
             NullableJSONModel.objects.filter(value__a__isnull=True),
             self.objs[:3] + self.objs[5:],
         )
+        self.assertSequenceEqual(
+            NullableJSONModel.objects.filter(value__j__isnull=True),
+            self.objs[:4] + self.objs[5:],
+        )
         self.assertSequenceEqual(
             NullableJSONModel.objects.filter(value__a__isnull=False),
             [self.objs[3], self.objs[4]],

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 model_fields.test_jsonfield
: '>>>>> End Test Output'
git checkout 3f140dde51c0fe6c350acb7727bbe489a99f0632 tests/model_fields/test_jsonfield.py
