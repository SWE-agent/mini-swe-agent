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
git -c core.fileMode=false diff 184a6eebb0ef56d5f1b1315a8e666830e37f3f81
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 184a6eebb0ef56d5f1b1315a8e666830e37f3f81 tests/syndication_tests/feeds.py tests/syndication_tests/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/syndication_tests/feeds.py b/tests/syndication_tests/feeds.py
--- a/tests/syndication_tests/feeds.py
+++ b/tests/syndication_tests/feeds.py
@@ -29,6 +29,9 @@ def item_pubdate(self, item):
     def item_updateddate(self, item):
         return item.updated
 
+    def item_comments(self, item):
+        return "%scomments" % item.get_absolute_url()
+
     item_author_name = 'Sally Smith'
     item_author_email = 'test@example.com'
     item_author_link = 'http://www.example.com/'
diff --git a/tests/syndication_tests/tests.py b/tests/syndication_tests/tests.py
--- a/tests/syndication_tests/tests.py
+++ b/tests/syndication_tests/tests.py
@@ -136,10 +136,20 @@ def test_rss2_feed(self):
             'guid': 'http://example.com/blog/1/',
             'pubDate': pub_date,
             'author': 'test@example.com (Sally Smith)',
+            'comments': '/blog/1/comments',
         })
         self.assertCategories(items[0], ['python', 'testing'])
         for item in items:
-            self.assertChildNodes(item, ['title', 'link', 'description', 'guid', 'category', 'pubDate', 'author'])
+            self.assertChildNodes(item, [
+                'title',
+                'link',
+                'description',
+                'guid',
+                'category',
+                'pubDate',
+                'author',
+                'comments',
+            ])
             # Assert that <guid> does not have any 'isPermaLink' attribute
             self.assertIsNone(item.getElementsByTagName(
                 'guid')[0].attributes.get('isPermaLink'))

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 syndication_tests.feeds syndication_tests.tests
: '>>>>> End Test Output'
git checkout 184a6eebb0ef56d5f1b1315a8e666830e37f3f81 tests/syndication_tests/feeds.py tests/syndication_tests/tests.py
