#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 1bd6a7a0acc11e249fca11c017505ad39f15ebf6
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 1bd6a7a0acc11e249fca11c017505ad39f15ebf6 tests/admin_inlines/tests.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/admin_inlines/tests.py b/tests/admin_inlines/tests.py
--- a/tests/admin_inlines/tests.py
+++ b/tests/admin_inlines/tests.py
@@ -967,6 +967,55 @@ def test_extra_inlines_are_not_shown(self):
 class TestVerboseNameInlineForms(TestDataMixin, TestCase):
     factory = RequestFactory()
 
+    def test_verbose_name_inline(self):
+        class NonVerboseProfileInline(TabularInline):
+            model = Profile
+            verbose_name = 'Non-verbose childs'
+
+        class VerboseNameProfileInline(TabularInline):
+            model = VerboseNameProfile
+            verbose_name = 'Childs with verbose name'
+
+        class VerboseNamePluralProfileInline(TabularInline):
+            model = VerboseNamePluralProfile
+            verbose_name = 'Childs with verbose name plural'
+
+        class BothVerboseNameProfileInline(TabularInline):
+            model = BothVerboseNameProfile
+            verbose_name = 'Childs with both verbose names'
+
+        modeladmin = ModelAdmin(ProfileCollection, admin_site)
+        modeladmin.inlines = [
+            NonVerboseProfileInline,
+            VerboseNameProfileInline,
+            VerboseNamePluralProfileInline,
+            BothVerboseNameProfileInline,
+        ]
+        obj = ProfileCollection.objects.create()
+        url = reverse('admin:admin_inlines_profilecollection_change', args=(obj.pk,))
+        request = self.factory.get(url)
+        request.user = self.superuser
+        response = modeladmin.changeform_view(request)
+        self.assertNotContains(response, 'Add another Profile')
+        # Non-verbose model.
+        self.assertContains(response, '<h2>Non-verbose childss</h2>')
+        self.assertContains(response, 'Add another Non-verbose child')
+        self.assertNotContains(response, '<h2>Profiles</h2>')
+        # Model with verbose name.
+        self.assertContains(response, '<h2>Childs with verbose names</h2>')
+        self.assertContains(response, 'Add another Childs with verbose name')
+        self.assertNotContains(response, '<h2>Model with verbose name onlys</h2>')
+        self.assertNotContains(response, 'Add another Model with verbose name only')
+        # Model with verbose name plural.
+        self.assertContains(response, '<h2>Childs with verbose name plurals</h2>')
+        self.assertContains(response, 'Add another Childs with verbose name plural')
+        self.assertNotContains(response, '<h2>Model with verbose name plural only</h2>')
+        # Model with both verbose names.
+        self.assertContains(response, '<h2>Childs with both verbose namess</h2>')
+        self.assertContains(response, 'Add another Childs with both verbose names')
+        self.assertNotContains(response, '<h2>Model with both - plural name</h2>')
+        self.assertNotContains(response, 'Add another Model with both - name')
+
     def test_verbose_name_plural_inline(self):
         class NonVerboseProfileInline(TabularInline):
             model = Profile

EOF_114329324912
: '>>>>> Start Test Output'
./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 admin_inlines.tests
: '>>>>> End Test Output'
git checkout 1bd6a7a0acc11e249fca11c017505ad39f15ebf6 tests/admin_inlines/tests.py
