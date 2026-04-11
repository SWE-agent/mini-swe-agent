#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff b9fb05c177b685bf683f7f57b2d57374eb7d882d
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout b9fb05c177b685bf683f7f57b2d57374eb7d882d pydicom/tests/test_valuerep.py
git apply -v - <<'EOF_114329324912'
diff --git a/pydicom/tests/test_valuerep.py b/pydicom/tests/test_valuerep.py
--- a/pydicom/tests/test_valuerep.py
+++ b/pydicom/tests/test_valuerep.py
@@ -427,6 +427,62 @@ def test_hash(self):
         )
         assert hash(pn1) == hash(pn2)
 
+    def test_next(self):
+        """Test that the next function works on it's own"""
+        # Test getting the first character
+        pn1 = PersonName("John^Doe^^Dr", encodings=default_encoding)
+        pn1_itr = iter(pn1)
+        assert next(pn1_itr) == "J"
+
+        # Test getting multiple characters
+        pn2 = PersonName(
+            "Yamada^Tarou=山田^太郎=やまだ^たろう", [default_encoding, "iso2022_jp"]
+        )
+        pn2_itr = iter(pn2)
+        assert next(pn2_itr) == "Y"
+        assert next(pn2_itr) == "a"
+
+        # Test getting all characters
+        pn3 = PersonName("SomeName")
+        pn3_itr = iter(pn3)
+        assert next(pn3_itr) == "S"
+        assert next(pn3_itr) == "o"
+        assert next(pn3_itr) == "m"
+        assert next(pn3_itr) == "e"
+        assert next(pn3_itr) == "N"
+        assert next(pn3_itr) == "a"
+        assert next(pn3_itr) == "m"
+        assert next(pn3_itr) == "e"
+
+        # Attempting to get next characeter should stop the iteration
+        # I.e. next can only start once
+        with pytest.raises(StopIteration):
+            next(pn3_itr)
+
+        # Test that next() doesn't work without instantiating an iterator
+        pn4 = PersonName("SomeName")
+        with pytest.raises(AttributeError):
+            next(pn4)
+
+    def test_iterator(self):
+        """Test that iterators can be corretly constructed"""
+        name_str = "John^Doe^^Dr"
+        pn1 = PersonName(name_str)
+        
+        for i, c in enumerate(pn1):
+            assert name_str[i] == c
+
+        # Ensure that multiple iterators can be created on the same variable
+        for i, c in enumerate(pn1):
+            assert name_str[i] == c
+
+    def test_contains(self):
+        """Test that characters can be check if they are within the name"""
+        pn1 = PersonName("John^Doe")
+        assert ("J" in pn1) == True
+        assert ("o" in pn1) == True
+        assert ("x" in pn1) == False
+
 
 class TestDateTime:
     """Unit tests for DA, DT, TM conversion to datetime objects"""

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA pydicom/tests/test_valuerep.py
: '>>>>> End Test Output'
git checkout b9fb05c177b685bf683f7f57b2d57374eb7d882d pydicom/tests/test_valuerep.py
