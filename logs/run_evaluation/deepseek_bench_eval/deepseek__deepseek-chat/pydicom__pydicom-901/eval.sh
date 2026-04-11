#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff 3746878d8edf1cbda6fbcf35eec69f9ba79301ca
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git checkout 3746878d8edf1cbda6fbcf35eec69f9ba79301ca 
git apply -v - <<'EOF_114329324912'
diff --git a/pydicom/tests/test_config.py b/pydicom/tests/test_config.py
new file mode 100644
--- /dev/null
+++ b/pydicom/tests/test_config.py
@@ -0,0 +1,107 @@
+# Copyright 2008-2019 pydicom authors. See LICENSE file for details.
+"""Unit tests for the pydicom.config module."""
+
+import logging
+import sys
+
+import pytest
+
+from pydicom import dcmread
+from pydicom.config import debug
+from pydicom.data import get_testdata_files
+
+
+DS_PATH = get_testdata_files("CT_small.dcm")[0]
+PYTEST = [int(x) for x in pytest.__version__.split('.')]
+
+
+@pytest.mark.skipif(PYTEST[:2] < [3, 4], reason='no caplog')
+class TestDebug(object):
+    """Tests for config.debug()."""
+    def setup(self):
+        self.logger = logging.getLogger('pydicom')
+
+    def teardown(self):
+        # Reset to just NullHandler
+        self.logger.handlers = [self.logger.handlers[0]]
+
+    def test_default(self, caplog):
+        """Test that the default logging handler is a NullHandler."""
+        assert 1 == len(self.logger.handlers)
+        assert isinstance(self.logger.handlers[0], logging.NullHandler)
+
+        with caplog.at_level(logging.DEBUG, logger='pydicom'):
+            ds = dcmread(DS_PATH)
+
+            assert "Call to dcmread()" not in caplog.text
+            assert "Reading File Meta Information preamble..." in caplog.text
+            assert "Reading File Meta Information prefix..." in caplog.text
+            assert "00000080: 'DICM' prefix found" in caplog.text
+
+    def test_debug_on_handler_null(self, caplog):
+        """Test debug(True, False)."""
+        debug(True, False)
+        assert 1 == len(self.logger.handlers)
+        assert isinstance(self.logger.handlers[0], logging.NullHandler)
+
+        with caplog.at_level(logging.DEBUG, logger='pydicom'):
+            ds = dcmread(DS_PATH)
+
+            assert "Call to dcmread()" in caplog.text
+            assert "Reading File Meta Information preamble..." in caplog.text
+            assert "Reading File Meta Information prefix..." in caplog.text
+            assert "00000080: 'DICM' prefix found" in caplog.text
+            msg = (
+                "00009848: fc ff fc ff 4f 42 00 00 7e 00 00 00    "
+                "(fffc, fffc) OB Length: 126"
+            )
+            assert msg in caplog.text
+
+    def test_debug_off_handler_null(self, caplog):
+        """Test debug(False, False)."""
+        debug(False, False)
+        assert 1 == len(self.logger.handlers)
+        assert isinstance(self.logger.handlers[0], logging.NullHandler)
+
+        with caplog.at_level(logging.DEBUG, logger='pydicom'):
+            ds = dcmread(DS_PATH)
+
+            assert "Call to dcmread()" not in caplog.text
+            assert "Reading File Meta Information preamble..." in caplog.text
+            assert "Reading File Meta Information prefix..." in caplog.text
+            assert "00000080: 'DICM' prefix found" in caplog.text
+
+    def test_debug_on_handler_stream(self, caplog):
+        """Test debug(True, True)."""
+        debug(True, True)
+        assert 2 == len(self.logger.handlers)
+        assert isinstance(self.logger.handlers[0], logging.NullHandler)
+        assert isinstance(self.logger.handlers[1], logging.StreamHandler)
+
+        with caplog.at_level(logging.DEBUG, logger='pydicom'):
+            ds = dcmread(DS_PATH)
+
+            assert "Call to dcmread()" in caplog.text
+            assert "Reading File Meta Information preamble..." in caplog.text
+            assert "Reading File Meta Information prefix..." in caplog.text
+            assert "00000080: 'DICM' prefix found" in caplog.text
+            msg = (
+                "00009848: fc ff fc ff 4f 42 00 00 7e 00 00 00    "
+                "(fffc, fffc) OB Length: 126"
+            )
+            assert msg in caplog.text
+
+    def test_debug_off_handler_stream(self, caplog):
+        """Test debug(False, True)."""
+        debug(False, True)
+        assert 2 == len(self.logger.handlers)
+        assert isinstance(self.logger.handlers[0], logging.NullHandler)
+        assert isinstance(self.logger.handlers[1], logging.StreamHandler)
+
+        with caplog.at_level(logging.DEBUG, logger='pydicom'):
+            ds = dcmread(DS_PATH)
+
+            assert "Call to dcmread()" not in caplog.text
+            assert "Reading File Meta Information preamble..." in caplog.text
+            assert "Reading File Meta Information prefix..." in caplog.text
+            assert "00000080: 'DICM' prefix found" in caplog.text

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA pydicom/tests/test_config.py
: '>>>>> End Test Output'
git checkout 3746878d8edf1cbda6fbcf35eec69f9ba79301ca 
