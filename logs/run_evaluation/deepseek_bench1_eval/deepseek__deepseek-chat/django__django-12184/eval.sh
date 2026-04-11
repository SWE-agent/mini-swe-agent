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
git -c core.fileMode=false diff 5d674eac871a306405b0fbbaeb17bbeba9c68bf3
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 5d674eac871a306405b0fbbaeb17bbeba9c68bf3 tests/urlpatterns/path_urls.py tests/urlpatterns/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/urlpatterns/path_urls.py b/tests/urlpatterns/path_urls.py
--- a/tests/urlpatterns/path_urls.py
+++ b/tests/urlpatterns/path_urls.py
@@ -12,6 +12,11 @@
     path('included_urls/', include('urlpatterns.included_urls')),
     re_path(r'^regex/(?P<pk>[0-9]+)/$', views.empty_view, name='regex'),
     re_path(r'^regex_optional/(?P<arg1>\d+)/(?:(?P<arg2>\d+)/)?', views.empty_view, name='regex_optional'),
+    re_path(
+        r'^regex_only_optional/(?:(?P<arg1>\d+)/)?',
+        views.empty_view,
+        name='regex_only_optional',
+    ),
     path('', include('urlpatterns.more_urls')),
     path('<lang>/<path:url>/', views.empty_view, name='lang-and-path'),
 ]
diff --git a/tests/urlpatterns/tests.py b/tests/urlpatterns/tests.py
--- a/tests/urlpatterns/tests.py
+++ b/tests/urlpatterns/tests.py
@@ -68,6 +68,16 @@ def test_re_path_with_optional_parameter(self):
                     r'^regex_optional/(?P<arg1>\d+)/(?:(?P<arg2>\d+)/)?',
                 )
 
+    def test_re_path_with_missing_optional_parameter(self):
+        match = resolve('/regex_only_optional/')
+        self.assertEqual(match.url_name, 'regex_only_optional')
+        self.assertEqual(match.kwargs, {})
+        self.assertEqual(match.args, ())
+        self.assertEqual(
+            match.route,
+            r'^regex_only_optional/(?:(?P<arg1>\d+)/)?',
+        )
+
     def test_path_lookup_with_inclusion(self):
         match = resolve('/included_urls/extra/something/')
         self.assertEqual(match.url_name, 'inner-extra')

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 urlpatterns.path_urls urlpatterns.tests
: '>>>>> End Test Output'
git checkout 5d674eac871a306405b0fbbaeb17bbeba9c68bf3 tests/urlpatterns/path_urls.py tests/urlpatterns/tests.py
