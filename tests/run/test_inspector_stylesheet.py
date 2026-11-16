import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("comment", "padding"), [("custom style", 2), ("\u4e2d\u6587\u6837\u5f0f \U0001f680 \u00e9", 3)]
)
def test_utf8_stylesheet_without_utf8_mode(tmp_path, comment, padding):
    stylesheet = tmp_path / "custom.tcss"
    stylesheet.write_text(f"/* {comment} */\n#content {{ padding: {padding}; }}\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio, sys\n"
            "from pathlib import Path\n"
            "from minisweagent.run.utilities.inspector import TrajectoryInspector\n"
            "app = TrajectoryInspector([])\n"
            "assert app.CSS == Path(sys.argv[1]).read_text(encoding='utf-8')\n"
            "async def check():\n"
            "    async with app.run_test():\n"
            "        assert app.query_one('#content').styles.padding.top == int(sys.argv[2])\n"
            "asyncio.run(check())\n",
            str(stylesheet),
            str(padding),
        ],
        env=os.environ
        | {
            "MSWEA_INSPECTOR_STYLE_PATH": str(stylesheet),
            "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "config"),
            "MSWEA_SILENT_STARTUP": "1",
            "PYTHONUTF8": "0",
            "PYTHONCOERCECLOCALE": "0",
            "LC_ALL": "C",
        },
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
