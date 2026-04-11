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
git -c core.fileMode=false diff b2b0711b555fa292751763c2df4fe577c396f265
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout b2b0711b555fa292751763c2df4fe577c396f265 tests/migrations/test_autodetector.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/migrations/test_autodetector.py b/tests/migrations/test_autodetector.py
--- a/tests/migrations/test_autodetector.py
+++ b/tests/migrations/test_autodetector.py
@@ -2151,6 +2151,115 @@ def test_add_model_order_with_respect_to(self):
         )
         self.assertNotIn("_order", [name for name, field in changes['testapp'][0].operations[0].fields])
 
+    def test_add_model_order_with_respect_to_index_foo_together(self):
+        changes = self.get_changes([], [
+            self.book,
+            ModelState('testapp', 'Author', [
+                ('id', models.AutoField(primary_key=True)),
+                ('name', models.CharField(max_length=200)),
+                ('book', models.ForeignKey('otherapp.Book', models.CASCADE)),
+            ], options={
+                'order_with_respect_to': 'book',
+                'index_together': {('name', '_order')},
+                'unique_together': {('id', '_order')},
+            }),
+        ])
+        self.assertNumberMigrations(changes, 'testapp', 1)
+        self.assertOperationTypes(changes, 'testapp', 0, ['CreateModel'])
+        self.assertOperationAttributes(
+            changes,
+            'testapp',
+            0,
+            0,
+            name='Author',
+            options={
+                'order_with_respect_to': 'book',
+                'index_together': {('name', '_order')},
+                'unique_together': {('id', '_order')},
+            },
+        )
+
+    def test_add_model_order_with_respect_to_index_constraint(self):
+        tests = [
+            (
+                'AddIndex',
+                {'indexes': [
+                    models.Index(fields=['_order'], name='book_order_idx'),
+                ]},
+            ),
+            (
+                'AddConstraint',
+                {'constraints': [
+                    models.CheckConstraint(
+                        check=models.Q(_order__gt=1),
+                        name='book_order_gt_1',
+                    ),
+                ]},
+            ),
+        ]
+        for operation, extra_option in tests:
+            with self.subTest(operation=operation):
+                after = ModelState('testapp', 'Author', [
+                    ('id', models.AutoField(primary_key=True)),
+                    ('name', models.CharField(max_length=200)),
+                    ('book', models.ForeignKey('otherapp.Book', models.CASCADE)),
+                ], options={
+                    'order_with_respect_to': 'book',
+                    **extra_option,
+                })
+                changes = self.get_changes([], [self.book, after])
+                self.assertNumberMigrations(changes, 'testapp', 1)
+                self.assertOperationTypes(changes, 'testapp', 0, [
+                    'CreateModel', operation,
+                ])
+                self.assertOperationAttributes(
+                    changes,
+                    'testapp',
+                    0,
+                    0,
+                    name='Author',
+                    options={'order_with_respect_to': 'book'},
+                )
+
+    def test_set_alter_order_with_respect_to_index_constraint_foo_together(self):
+        tests = [
+            (
+                'AddIndex',
+                {'indexes': [
+                    models.Index(fields=['_order'], name='book_order_idx'),
+                ]},
+            ),
+            (
+                'AddConstraint',
+                {'constraints': [
+                    models.CheckConstraint(
+                        check=models.Q(_order__gt=1),
+                        name='book_order_gt_1',
+                    ),
+                ]},
+            ),
+            ('AlterIndexTogether', {'index_together': {('name', '_order')}}),
+            ('AlterUniqueTogether', {'unique_together': {('id', '_order')}}),
+        ]
+        for operation, extra_option in tests:
+            with self.subTest(operation=operation):
+                after = ModelState('testapp', 'Author', [
+                    ('id', models.AutoField(primary_key=True)),
+                    ('name', models.CharField(max_length=200)),
+                    ('book', models.ForeignKey('otherapp.Book', models.CASCADE)),
+                ], options={
+                    'order_with_respect_to': 'book',
+                    **extra_option,
+                })
+                changes = self.get_changes(
+                    [self.book, self.author_with_book],
+                    [self.book, after],
+                )
+                self.assertNumberMigrations(changes, 'testapp', 1)
+                self.assertOperationTypes(changes, 'testapp', 0, [
+                    'AlterOrderWithRespectTo', operation,
+                ])
+
     def test_alter_model_managers(self):
         """
         Changing the model managers adds a new operation.

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 migrations.test_autodetector
: '>>>>> End Test Output'
git checkout b2b0711b555fa292751763c2df4fe577c396f265 tests/migrations/test_autodetector.py
