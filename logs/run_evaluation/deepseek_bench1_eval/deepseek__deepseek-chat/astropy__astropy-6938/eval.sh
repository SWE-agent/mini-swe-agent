#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff c76af9ed6bb89bfba45b9f5bc1e635188278e2fa
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .[test] --verbose
git checkout c76af9ed6bb89bfba45b9f5bc1e635188278e2fa astropy/io/fits/tests/test_checksum.py astropy/io/fits/tests/test_table.py
git apply -v - <<'EOF_114329324912'
diff --git a/astropy/io/fits/tests/test_checksum.py b/astropy/io/fits/tests/test_checksum.py
--- a/astropy/io/fits/tests/test_checksum.py
+++ b/astropy/io/fits/tests/test_checksum.py
@@ -205,9 +205,9 @@ def test_ascii_table_data(self):
                 # The checksum ends up being different on Windows, possibly due
                 # to slight floating point differences
                 assert 'CHECKSUM' in hdul[1].header
-                assert hdul[1].header['CHECKSUM'] == '51IDA1G981GCA1G9'
+                assert hdul[1].header['CHECKSUM'] == '3rKFAoI94oICAoI9'
                 assert 'DATASUM' in hdul[1].header
-                assert hdul[1].header['DATASUM'] == '1948208413'
+                assert hdul[1].header['DATASUM'] == '1914653725'
 
     def test_compressed_image_data(self):
         with fits.open(self.data('comp.fits')) as h1:
diff --git a/astropy/io/fits/tests/test_table.py b/astropy/io/fits/tests/test_table.py
--- a/astropy/io/fits/tests/test_table.py
+++ b/astropy/io/fits/tests/test_table.py
@@ -298,6 +298,19 @@ def test_ascii_table(self):
         hdul = fits.open(self.temp('toto.fits'))
         assert comparerecords(hdu.data, hdul[1].data)
         hdul.close()
+
+        # Test Scaling
+
+        r1 = np.array([11., 12.])
+        c2 = fits.Column(name='def', format='D', array=r1, bscale=2.3,
+                         bzero=0.6)
+        hdu = fits.TableHDU.from_columns([c2])
+        hdu.writeto(self.temp('toto.fits'), overwrite=True)
+        with open(self.temp('toto.fits')) as f:
+            assert '4.95652173913043548D+00' in f.read()
+        with fits.open(self.temp('toto.fits')) as hdul:
+            assert comparerecords(hdu.data, hdul[1].data)
+
         a.close()
 
     def test_endianness(self):

EOF_114329324912
: '>>>>> Start Test Output'
pytest -rA -vv -o console_output_style=classic --tb=no astropy/io/fits/tests/test_checksum.py astropy/io/fits/tests/test_table.py
: '>>>>> End Test Output'
git checkout c76af9ed6bb89bfba45b9f5bc1e635188278e2fa astropy/io/fits/tests/test_checksum.py astropy/io/fits/tests/test_table.py
