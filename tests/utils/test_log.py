import logging

from minisweagent.utils.log import add_file_handler


def test_file_logging_preserves_unicode(tmp_path):
    path = tmp_path / "agent.log"
    logger = logging.getLogger("minisweagent")
    add_file_handler(path, print_path=False)
    handler = logger.handlers[-1]
    try:
        logger.debug("Task output: \u4e2d\u6587 \U0001f680 \u03b1")
        handler.flush()
        assert "Task output: \u4e2d\u6587 \U0001f680 \u03b1" in path.read_text(encoding="utf-8")
    finally:
        logger.removeHandler(handler)
        handler.close()
