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
git -c core.fileMode=false diff 93e892bb645b16ebaf287beb5fe7f3ffe8d10408
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 93e892bb645b16ebaf287beb5fe7f3ffe8d10408 tests/admin_inlines/tests.py tests/admin_widgets/test_autocomplete_widget.py tests/forms_tests/tests/test_media.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/admin_inlines/tests.py b/tests/admin_inlines/tests.py
--- a/tests/admin_inlines/tests.py
+++ b/tests/admin_inlines/tests.py
@@ -497,10 +497,10 @@ def test_inline_media_only_inline(self):
             response.context['inline_admin_formsets'][0].media._js,
             [
                 'admin/js/vendor/jquery/jquery.min.js',
-                'admin/js/jquery.init.js',
-                'admin/js/inlines.min.js',
                 'my_awesome_inline_scripts.js',
                 'custom_number.js',
+                'admin/js/jquery.init.js',
+                'admin/js/inlines.min.js',
             ]
         )
         self.assertContains(response, 'my_awesome_inline_scripts.js')
diff --git a/tests/admin_widgets/test_autocomplete_widget.py b/tests/admin_widgets/test_autocomplete_widget.py
--- a/tests/admin_widgets/test_autocomplete_widget.py
+++ b/tests/admin_widgets/test_autocomplete_widget.py
@@ -139,4 +139,4 @@ def test_media(self):
                 else:
                     expected_files = base_files
                 with translation.override(lang):
-                    self.assertEqual(AutocompleteSelect(rel, admin.site).media._js, expected_files)
+                    self.assertEqual(AutocompleteSelect(rel, admin.site).media._js, list(expected_files))
diff --git a/tests/forms_tests/tests/test_media.py b/tests/forms_tests/tests/test_media.py
--- a/tests/forms_tests/tests/test_media.py
+++ b/tests/forms_tests/tests/test_media.py
@@ -25,8 +25,8 @@ def test_construction(self):
         )
         self.assertEqual(
             repr(m),
-            "Media(css={'all': ('path/to/css1', '/path/to/css2')}, "
-            "js=('/path/to/js1', 'http://media.other.com/path/to/js2', 'https://secure.other.com/path/to/js3'))"
+            "Media(css={'all': ['path/to/css1', '/path/to/css2']}, "
+            "js=['/path/to/js1', 'http://media.other.com/path/to/js2', 'https://secure.other.com/path/to/js3'])"
         )
 
         class Foo:
@@ -125,8 +125,8 @@ class Media:
 <link href="/path/to/css3" type="text/css" media="all" rel="stylesheet">
 <script type="text/javascript" src="/path/to/js1"></script>
 <script type="text/javascript" src="http://media.other.com/path/to/js2"></script>
-<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>
-<script type="text/javascript" src="/path/to/js4"></script>"""
+<script type="text/javascript" src="/path/to/js4"></script>
+<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>"""
         )
 
         # media addition hasn't affected the original objects
@@ -151,6 +151,17 @@ class Media:
         self.assertEqual(str(w4.media), """<link href="/path/to/css1" type="text/css" media="all" rel="stylesheet">
 <script type="text/javascript" src="/path/to/js1"></script>""")
 
+    def test_media_deduplication(self):
+        # A deduplication test applied directly to a Media object, to confirm
+        # that the deduplication doesn't only happen at the point of merging
+        # two or more media objects.
+        media = Media(
+            css={'all': ('/path/to/css1', '/path/to/css1')},
+            js=('/path/to/js1', '/path/to/js1'),
+        )
+        self.assertEqual(str(media), """<link href="/path/to/css1" type="text/css" media="all" rel="stylesheet">
+<script type="text/javascript" src="/path/to/js1"></script>""")
+
     def test_media_property(self):
         ###############################################################
         # Property-based media definitions
@@ -197,12 +208,12 @@ def _media(self):
         self.assertEqual(
             str(w6.media),
             """<link href="http://media.example.com/static/path/to/css1" type="text/css" media="all" rel="stylesheet">
-<link href="/path/to/css2" type="text/css" media="all" rel="stylesheet">
 <link href="/other/path" type="text/css" media="all" rel="stylesheet">
+<link href="/path/to/css2" type="text/css" media="all" rel="stylesheet">
 <script type="text/javascript" src="/path/to/js1"></script>
+<script type="text/javascript" src="/other/js"></script>
 <script type="text/javascript" src="http://media.other.com/path/to/js2"></script>
-<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>
-<script type="text/javascript" src="/other/js"></script>"""
+<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>"""
         )
 
     def test_media_inheritance(self):
@@ -247,8 +258,8 @@ class Media:
 <link href="/path/to/css2" type="text/css" media="all" rel="stylesheet">
 <script type="text/javascript" src="/path/to/js1"></script>
 <script type="text/javascript" src="http://media.other.com/path/to/js2"></script>
-<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>
-<script type="text/javascript" src="/path/to/js4"></script>"""
+<script type="text/javascript" src="/path/to/js4"></script>
+<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>"""
         )
 
     def test_media_inheritance_from_property(self):
@@ -322,8 +333,8 @@ class Media:
 <link href="/path/to/css2" type="text/css" media="all" rel="stylesheet">
 <script type="text/javascript" src="/path/to/js1"></script>
 <script type="text/javascript" src="http://media.other.com/path/to/js2"></script>
-<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>
-<script type="text/javascript" src="/path/to/js4"></script>"""
+<script type="text/javascript" src="/path/to/js4"></script>
+<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>"""
         )
 
     def test_media_inheritance_single_type(self):
@@ -420,8 +431,8 @@ def __init__(self, attrs=None):
 <link href="/path/to/css3" type="text/css" media="all" rel="stylesheet">
 <script type="text/javascript" src="/path/to/js1"></script>
 <script type="text/javascript" src="http://media.other.com/path/to/js2"></script>
-<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>
-<script type="text/javascript" src="/path/to/js4"></script>"""
+<script type="text/javascript" src="/path/to/js4"></script>
+<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>"""
         )
 
     def test_form_media(self):
@@ -462,8 +473,8 @@ class MyForm(Form):
 <link href="/path/to/css3" type="text/css" media="all" rel="stylesheet">
 <script type="text/javascript" src="/path/to/js1"></script>
 <script type="text/javascript" src="http://media.other.com/path/to/js2"></script>
-<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>
-<script type="text/javascript" src="/path/to/js4"></script>"""
+<script type="text/javascript" src="/path/to/js4"></script>
+<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>"""
         )
 
         # Form media can be combined to produce a single media definition.
@@ -477,8 +488,8 @@ class AnotherForm(Form):
 <link href="/path/to/css3" type="text/css" media="all" rel="stylesheet">
 <script type="text/javascript" src="/path/to/js1"></script>
 <script type="text/javascript" src="http://media.other.com/path/to/js2"></script>
-<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>
-<script type="text/javascript" src="/path/to/js4"></script>"""
+<script type="text/javascript" src="/path/to/js4"></script>
+<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>"""
         )
 
         # Forms can also define media, following the same rules as widgets.
@@ -495,28 +506,28 @@ class Media:
         self.assertEqual(
             str(f3.media),
             """<link href="http://media.example.com/static/path/to/css1" type="text/css" media="all" rel="stylesheet">
+<link href="/some/form/css" type="text/css" media="all" rel="stylesheet">
 <link href="/path/to/css2" type="text/css" media="all" rel="stylesheet">
 <link href="/path/to/css3" type="text/css" media="all" rel="stylesheet">
-<link href="/some/form/css" type="text/css" media="all" rel="stylesheet">
 <script type="text/javascript" src="/path/to/js1"></script>
+<script type="text/javascript" src="/some/form/javascript"></script>
 <script type="text/javascript" src="http://media.other.com/path/to/js2"></script>
-<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>
 <script type="text/javascript" src="/path/to/js4"></script>
-<script type="text/javascript" src="/some/form/javascript"></script>"""
+<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>"""
         )
 
         # Media works in templates
         self.assertEqual(
             Template("{{ form.media.js }}{{ form.media.css }}").render(Context({'form': f3})),
             """<script type="text/javascript" src="/path/to/js1"></script>
+<script type="text/javascript" src="/some/form/javascript"></script>
 <script type="text/javascript" src="http://media.other.com/path/to/js2"></script>
-<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>
 <script type="text/javascript" src="/path/to/js4"></script>
-<script type="text/javascript" src="/some/form/javascript"></script>"""
+<script type="text/javascript" src="https://secure.other.com/path/to/js3"></script>"""
             """<link href="http://media.example.com/static/path/to/css1" type="text/css" media="all" rel="stylesheet">
+<link href="/some/form/css" type="text/css" media="all" rel="stylesheet">
 <link href="/path/to/css2" type="text/css" media="all" rel="stylesheet">
-<link href="/path/to/css3" type="text/css" media="all" rel="stylesheet">
-<link href="/some/form/css" type="text/css" media="all" rel="stylesheet">"""
+<link href="/path/to/css3" type="text/css" media="all" rel="stylesheet">"""
         )
 
     def test_html_safe(self):
@@ -526,19 +537,23 @@ def test_html_safe(self):
 
     def test_merge(self):
         test_values = (
-            (([1, 2], [3, 4]), [1, 2, 3, 4]),
+            (([1, 2], [3, 4]), [1, 3, 2, 4]),
             (([1, 2], [2, 3]), [1, 2, 3]),
             (([2, 3], [1, 2]), [1, 2, 3]),
             (([1, 3], [2, 3]), [1, 2, 3]),
             (([1, 2], [1, 3]), [1, 2, 3]),
             (([1, 2], [3, 2]), [1, 3, 2]),
+            (([1, 2], [1, 2]), [1, 2]),
+            ([[1, 2], [1, 3], [2, 3], [5, 7], [5, 6], [6, 7, 9], [8, 9]], [1, 5, 8, 2, 6, 3, 7, 9]),
+            ((), []),
+            (([1, 2],), [1, 2]),
         )
-        for (list1, list2), expected in test_values:
-            with self.subTest(list1=list1, list2=list2):
-                self.assertEqual(Media.merge(list1, list2), expected)
+        for lists, expected in test_values:
+            with self.subTest(lists=lists):
+                self.assertEqual(Media.merge(*lists), expected)
 
     def test_merge_warning(self):
-        msg = 'Detected duplicate Media files in an opposite order:\n1\n2'
+        msg = 'Detected duplicate Media files in an opposite order: [1, 2], [2, 1]'
         with self.assertWarnsMessage(RuntimeWarning, msg):
             self.assertEqual(Media.merge([1, 2], [2, 1]), [1, 2])
 
@@ -546,28 +561,30 @@ def test_merge_js_three_way(self):
         """
         The relative order of scripts is preserved in a three-way merge.
         """
-        # custom_widget.js doesn't depend on jquery.js.
-        widget1 = Media(js=['custom_widget.js'])
-        widget2 = Media(js=['jquery.js', 'uses_jquery.js'])
-        form_media = widget1 + widget2
-        # The relative ordering of custom_widget.js and jquery.js has been
-        # established (but without a real need to).
-        self.assertEqual(form_media._js, ['custom_widget.js', 'jquery.js', 'uses_jquery.js'])
-        # The inline also uses custom_widget.js. This time, it's at the end.
-        inline_media = Media(js=['jquery.js', 'also_jquery.js']) + Media(js=['custom_widget.js'])
-        merged = form_media + inline_media
-        self.assertEqual(merged._js, ['custom_widget.js', 'jquery.js', 'uses_jquery.js', 'also_jquery.js'])
+        widget1 = Media(js=['color-picker.js'])
+        widget2 = Media(js=['text-editor.js'])
+        widget3 = Media(js=['text-editor.js', 'text-editor-extras.js', 'color-picker.js'])
+        merged = widget1 + widget2 + widget3
+        self.assertEqual(merged._js, ['text-editor.js', 'text-editor-extras.js', 'color-picker.js'])
+
+    def test_merge_js_three_way2(self):
+        # The merge prefers to place 'c' before 'b' and 'g' before 'h' to
+        # preserve the original order. The preference 'c'->'b' is overridden by
+        # widget3's media, but 'g'->'h' survives in the final ordering.
+        widget1 = Media(js=['a', 'c', 'f', 'g', 'k'])
+        widget2 = Media(js=['a', 'b', 'f', 'h', 'k'])
+        widget3 = Media(js=['b', 'c', 'f', 'k'])
+        merged = widget1 + widget2 + widget3
+        self.assertEqual(merged._js, ['a', 'b', 'c', 'f', 'g', 'h', 'k'])
 
     def test_merge_css_three_way(self):
-        widget1 = Media(css={'screen': ['a.css']})
-        widget2 = Media(css={'screen': ['b.css']})
-        widget3 = Media(css={'all': ['c.css']})
-        form1 = widget1 + widget2
-        form2 = widget2 + widget1
-        # form1 and form2 have a.css and b.css in different order...
-        self.assertEqual(form1._css, {'screen': ['a.css', 'b.css']})
-        self.assertEqual(form2._css, {'screen': ['b.css', 'a.css']})
-        # ...but merging succeeds as the relative ordering of a.css and b.css
-        # was never specified.
-        merged = widget3 + form1 + form2
-        self.assertEqual(merged._css, {'screen': ['a.css', 'b.css'], 'all': ['c.css']})
+        widget1 = Media(css={'screen': ['c.css'], 'all': ['d.css', 'e.css']})
+        widget2 = Media(css={'screen': ['a.css']})
+        widget3 = Media(css={'screen': ['a.css', 'b.css', 'c.css'], 'all': ['e.css']})
+        merged = widget1 + widget2
+        # c.css comes before a.css because widget1 + widget2 establishes this
+        # order.
+        self.assertEqual(merged._css, {'screen': ['c.css', 'a.css'], 'all': ['d.css', 'e.css']})
+        merged = merged + widget3
+        # widget3 contains an explicit ordering of c.css and a.css.
+        self.assertEqual(merged._css, {'screen': ['a.css', 'b.css', 'c.css'], 'all': ['d.css', 'e.css']})

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 admin_inlines.tests admin_widgets.test_autocomplete_widget forms_tests.tests.test_media
: '>>>>> End Test Output'
git checkout 93e892bb645b16ebaf287beb5fe7f3ffe8d10408 tests/admin_inlines/tests.py tests/admin_widgets/test_autocomplete_widget.py tests/forms_tests/tests/test_media.py
