#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 6f73eb9d90cfec684529aab48d517e3d6449ba8c
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 6f73eb9d90cfec684529aab48d517e3d6449ba8c tests/migrations/test_autodetector.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/migrations/test_autodetector.py b/tests/migrations/test_autodetector.py
--- a/tests/migrations/test_autodetector.py
+++ b/tests/migrations/test_autodetector.py
@@ -868,6 +868,18 @@ class AutodetectorTests(TestCase):
             "unique_together": {("title", "newfield2")},
         },
     )
+    book_unique_together = ModelState(
+        "otherapp",
+        "Book",
+        [
+            ("id", models.AutoField(primary_key=True)),
+            ("author", models.ForeignKey("testapp.Author", models.CASCADE)),
+            ("title", models.CharField(max_length=200)),
+        ],
+        {
+            "unique_together": {("author", "title")},
+        },
+    )
     attribution = ModelState(
         "otherapp",
         "Attribution",
@@ -3798,16 +3810,16 @@ def test_many_to_many_changed_to_concrete_field(self):
         # Right number/type of migrations?
         self.assertNumberMigrations(changes, "testapp", 1)
         self.assertOperationTypes(
-            changes, "testapp", 0, ["RemoveField", "AddField", "DeleteModel"]
+            changes, "testapp", 0, ["RemoveField", "DeleteModel", "AddField"]
         )
         self.assertOperationAttributes(
             changes, "testapp", 0, 0, name="publishers", model_name="author"
         )
+        self.assertOperationAttributes(changes, "testapp", 0, 1, name="Publisher")
         self.assertOperationAttributes(
-            changes, "testapp", 0, 1, name="publishers", model_name="author"
+            changes, "testapp", 0, 2, name="publishers", model_name="author"
         )
-        self.assertOperationAttributes(changes, "testapp", 0, 2, name="Publisher")
-        self.assertOperationFieldAttributes(changes, "testapp", 0, 1, max_length=100)
+        self.assertOperationFieldAttributes(changes, "testapp", 0, 2, max_length=100)
 
     def test_non_circular_foreignkey_dependency_removal(self):
         """
@@ -4346,6 +4358,36 @@ def test_fk_dependency_other_app(self):
             changes, "testapp", 0, [("otherapp", "__first__")]
         )
 
+    def test_alter_unique_together_fk_to_m2m(self):
+        changes = self.get_changes(
+            [self.author_name, self.book_unique_together],
+            [
+                self.author_name,
+                ModelState(
+                    "otherapp",
+                    "Book",
+                    [
+                        ("id", models.AutoField(primary_key=True)),
+                        ("author", models.ManyToManyField("testapp.Author")),
+                        ("title", models.CharField(max_length=200)),
+                    ],
+                ),
+            ],
+        )
+        self.assertNumberMigrations(changes, "otherapp", 1)
+        self.assertOperationTypes(
+            changes, "otherapp", 0, ["AlterUniqueTogether", "RemoveField", "AddField"]
+        )
+        self.assertOperationAttributes(
+            changes, "otherapp", 0, 0, name="book", unique_together=set()
+        )
+        self.assertOperationAttributes(
+            changes, "otherapp", 0, 1, model_name="book", name="author"
+        )
+        self.assertOperationAttributes(
+            changes, "otherapp", 0, 2, model_name="book", name="author"
+        )
+
     def test_alter_field_to_fk_dependency_other_app(self):
         changes = self.get_changes(
             [self.author_empty, self.book_with_no_author_fk],

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 migrations.test_autodetector
: '>>>>> End Test Output'
git checkout 6f73eb9d90cfec684529aab48d517e3d6449ba8c tests/migrations/test_autodetector.py
