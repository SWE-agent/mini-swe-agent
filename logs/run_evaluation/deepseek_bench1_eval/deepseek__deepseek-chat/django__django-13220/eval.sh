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
git -c core.fileMode=false diff 16218c20606d8cd89c5393970c83da04598a3e04
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 16218c20606d8cd89c5393970c83da04598a3e04 tests/test_exceptions/test_validation_error.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_exceptions/test_validation_error.py b/tests/test_exceptions/test_validation_error.py
--- a/tests/test_exceptions/test_validation_error.py
+++ b/tests/test_exceptions/test_validation_error.py
@@ -1,4 +1,5 @@
 import unittest
+from unittest import mock
 
 from django.core.exceptions import ValidationError
 
@@ -14,3 +15,271 @@ def test_messages_concatenates_error_dict_values(self):
         message_dict['field2'] = ['E3', 'E4']
         exception = ValidationError(message_dict)
         self.assertEqual(sorted(exception.messages), ['E1', 'E2', 'E3', 'E4'])
+
+    def test_eq(self):
+        error1 = ValidationError('message')
+        error2 = ValidationError('message', code='my_code1')
+        error3 = ValidationError('message', code='my_code2')
+        error4 = ValidationError(
+            'error %(parm1)s %(parm2)s',
+            code='my_code1',
+            params={'parm1': 'val1', 'parm2': 'val2'},
+        )
+        error5 = ValidationError({'field1': 'message', 'field2': 'other'})
+        error6 = ValidationError({'field1': 'message'})
+        error7 = ValidationError([
+            ValidationError({'field1': 'field error', 'field2': 'other'}),
+            'message',
+        ])
+
+        self.assertEqual(error1, ValidationError('message'))
+        self.assertNotEqual(error1, ValidationError('message2'))
+        self.assertNotEqual(error1, error2)
+        self.assertNotEqual(error1, error4)
+        self.assertNotEqual(error1, error5)
+        self.assertNotEqual(error1, error6)
+        self.assertNotEqual(error1, error7)
+        self.assertEqual(error1, mock.ANY)
+        self.assertEqual(error2, ValidationError('message', code='my_code1'))
+        self.assertNotEqual(error2, ValidationError('other', code='my_code1'))
+        self.assertNotEqual(error2, error3)
+        self.assertNotEqual(error2, error4)
+        self.assertNotEqual(error2, error5)
+        self.assertNotEqual(error2, error6)
+        self.assertNotEqual(error2, error7)
+
+        self.assertEqual(error4, ValidationError(
+            'error %(parm1)s %(parm2)s',
+            code='my_code1',
+            params={'parm1': 'val1', 'parm2': 'val2'},
+        ))
+        self.assertNotEqual(error4, ValidationError(
+            'error %(parm1)s %(parm2)s',
+            code='my_code2',
+            params={'parm1': 'val1', 'parm2': 'val2'},
+        ))
+        self.assertNotEqual(error4, ValidationError(
+            'error %(parm1)s %(parm2)s',
+            code='my_code1',
+            params={'parm2': 'val2'},
+        ))
+        self.assertNotEqual(error4, ValidationError(
+            'error %(parm1)s %(parm2)s',
+            code='my_code1',
+            params={'parm2': 'val1', 'parm1': 'val2'},
+        ))
+        self.assertNotEqual(error4, ValidationError(
+            'error val1 val2',
+            code='my_code1',
+        ))
+        # params ordering is ignored.
+        self.assertEqual(error4, ValidationError(
+            'error %(parm1)s %(parm2)s',
+            code='my_code1',
+            params={'parm2': 'val2', 'parm1': 'val1'},
+        ))
+
+        self.assertEqual(
+            error5,
+            ValidationError({'field1': 'message', 'field2': 'other'}),
+        )
+        self.assertNotEqual(
+            error5,
+            ValidationError({'field1': 'message', 'field2': 'other2'}),
+        )
+        self.assertNotEqual(
+            error5,
+            ValidationError({'field1': 'message', 'field3': 'other'}),
+        )
+        self.assertNotEqual(error5, error6)
+        # fields ordering is ignored.
+        self.assertEqual(
+            error5,
+            ValidationError({'field2': 'other', 'field1': 'message'}),
+        )
+
+        self.assertNotEqual(error7, ValidationError(error7.error_list[1:]))
+        self.assertNotEqual(
+            ValidationError(['message']),
+            ValidationError([ValidationError('message', code='my_code')]),
+        )
+        # messages ordering is ignored.
+        self.assertEqual(
+            error7,
+            ValidationError(list(reversed(error7.error_list))),
+        )
+
+        self.assertNotEqual(error4, ValidationError([error4]))
+        self.assertNotEqual(ValidationError([error4]), error4)
+        self.assertNotEqual(error4, ValidationError({'field1': error4}))
+        self.assertNotEqual(ValidationError({'field1': error4}), error4)
+
+    def test_eq_nested(self):
+        error_dict = {
+            'field1': ValidationError(
+                'error %(parm1)s %(parm2)s',
+                code='my_code',
+                params={'parm1': 'val1', 'parm2': 'val2'},
+            ),
+            'field2': 'other',
+        }
+        error = ValidationError(error_dict)
+        self.assertEqual(error, ValidationError(dict(error_dict)))
+        self.assertEqual(error, ValidationError({
+            'field1': ValidationError(
+                'error %(parm1)s %(parm2)s',
+                code='my_code',
+                params={'parm2': 'val2', 'parm1': 'val1'},
+            ),
+            'field2': 'other',
+        }))
+        self.assertNotEqual(error, ValidationError(
+            {**error_dict, 'field2': 'message'},
+        ))
+        self.assertNotEqual(error, ValidationError({
+            'field1': ValidationError(
+                'error %(parm1)s val2',
+                code='my_code',
+                params={'parm1': 'val1'},
+            ),
+            'field2': 'other',
+        }))
+
+    def test_hash(self):
+        error1 = ValidationError('message')
+        error2 = ValidationError('message', code='my_code1')
+        error3 = ValidationError('message', code='my_code2')
+        error4 = ValidationError(
+            'error %(parm1)s %(parm2)s',
+            code='my_code1',
+            params={'parm1': 'val1', 'parm2': 'val2'},
+        )
+        error5 = ValidationError({'field1': 'message', 'field2': 'other'})
+        error6 = ValidationError({'field1': 'message'})
+        error7 = ValidationError([
+            ValidationError({'field1': 'field error', 'field2': 'other'}),
+            'message',
+        ])
+
+        self.assertEqual(hash(error1), hash(ValidationError('message')))
+        self.assertNotEqual(hash(error1), hash(ValidationError('message2')))
+        self.assertNotEqual(hash(error1), hash(error2))
+        self.assertNotEqual(hash(error1), hash(error4))
+        self.assertNotEqual(hash(error1), hash(error5))
+        self.assertNotEqual(hash(error1), hash(error6))
+        self.assertNotEqual(hash(error1), hash(error7))
+        self.assertEqual(
+            hash(error2),
+            hash(ValidationError('message', code='my_code1')),
+        )
+        self.assertNotEqual(
+            hash(error2),
+            hash(ValidationError('other', code='my_code1')),
+        )
+        self.assertNotEqual(hash(error2), hash(error3))
+        self.assertNotEqual(hash(error2), hash(error4))
+        self.assertNotEqual(hash(error2), hash(error5))
+        self.assertNotEqual(hash(error2), hash(error6))
+        self.assertNotEqual(hash(error2), hash(error7))
+
+        self.assertEqual(hash(error4), hash(ValidationError(
+            'error %(parm1)s %(parm2)s',
+            code='my_code1',
+            params={'parm1': 'val1', 'parm2': 'val2'},
+        )))
+        self.assertNotEqual(hash(error4), hash(ValidationError(
+            'error %(parm1)s %(parm2)s',
+            code='my_code2',
+            params={'parm1': 'val1', 'parm2': 'val2'},
+        )))
+        self.assertNotEqual(hash(error4), hash(ValidationError(
+            'error %(parm1)s %(parm2)s',
+            code='my_code1',
+            params={'parm2': 'val2'},
+        )))
+        self.assertNotEqual(hash(error4), hash(ValidationError(
+            'error %(parm1)s %(parm2)s',
+            code='my_code1',
+            params={'parm2': 'val1', 'parm1': 'val2'},
+        )))
+        self.assertNotEqual(hash(error4), hash(ValidationError(
+            'error val1 val2',
+            code='my_code1',
+        )))
+        # params ordering is ignored.
+        self.assertEqual(hash(error4), hash(ValidationError(
+            'error %(parm1)s %(parm2)s',
+            code='my_code1',
+            params={'parm2': 'val2', 'parm1': 'val1'},
+        )))
+
+        self.assertEqual(
+            hash(error5),
+            hash(ValidationError({'field1': 'message', 'field2': 'other'})),
+        )
+        self.assertNotEqual(
+            hash(error5),
+            hash(ValidationError({'field1': 'message', 'field2': 'other2'})),
+        )
+        self.assertNotEqual(
+            hash(error5),
+            hash(ValidationError({'field1': 'message', 'field3': 'other'})),
+        )
+        self.assertNotEqual(error5, error6)
+        # fields ordering is ignored.
+        self.assertEqual(
+            hash(error5),
+            hash(ValidationError({'field2': 'other', 'field1': 'message'})),
+        )
+
+        self.assertNotEqual(
+            hash(error7),
+            hash(ValidationError(error7.error_list[1:])),
+        )
+        self.assertNotEqual(
+            hash(ValidationError(['message'])),
+            hash(ValidationError([ValidationError('message', code='my_code')])),
+        )
+        # messages ordering is ignored.
+        self.assertEqual(
+            hash(error7),
+            hash(ValidationError(list(reversed(error7.error_list)))),
+        )
+
+        self.assertNotEqual(hash(error4), hash(ValidationError([error4])))
+        self.assertNotEqual(hash(ValidationError([error4])), hash(error4))
+        self.assertNotEqual(
+            hash(error4),
+            hash(ValidationError({'field1': error4})),
+        )
+
+    def test_hash_nested(self):
+        error_dict = {
+            'field1': ValidationError(
+                'error %(parm1)s %(parm2)s',
+                code='my_code',
+                params={'parm2': 'val2', 'parm1': 'val1'},
+            ),
+            'field2': 'other',
+        }
+        error = ValidationError(error_dict)
+        self.assertEqual(hash(error), hash(ValidationError(dict(error_dict))))
+        self.assertEqual(hash(error), hash(ValidationError({
+            'field1': ValidationError(
+                'error %(parm1)s %(parm2)s',
+                code='my_code',
+                params={'parm1': 'val1', 'parm2': 'val2'},
+            ),
+            'field2': 'other',
+        })))
+        self.assertNotEqual(hash(error), hash(ValidationError(
+            {**error_dict, 'field2': 'message'},
+        )))
+        self.assertNotEqual(hash(error), hash(ValidationError({
+            'field1': ValidationError(
+                'error %(parm1)s val2',
+                code='my_code',
+                params={'parm1': 'val1'},
+            ),
+            'field2': 'other',
+        })))

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 test_exceptions.test_validation_error
: '>>>>> End Test Output'
git checkout 16218c20606d8cd89c5393970c83da04598a3e04 tests/test_exceptions/test_validation_error.py
