import importlib

import prompt_toolkit.shortcuts


def test_sessions_are_not_constructed_at_import(monkeypatch):
    """Importing prompt_user must not construct a PromptSession.

    PromptSession probes the terminal when it is constructed; on Windows with a
    non-console stdout (redirected output, or pytest's captured stdout) that
    probe raises NoConsoleScreenBufferError, so importing the interactive agent
    crashed there while the Linux-only CI never saw it. The sessions are wrapped
    so construction is deferred to the first prompt, when a real terminal exists.
    """
    import minisweagent.agents.utils.prompt_user as prompt_user

    def _fail(*args, **kwargs):
        raise RuntimeError("PromptSession must not be constructed at import time")

    monkeypatch.setattr(prompt_toolkit.shortcuts, "PromptSession", _fail)
    try:
        importlib.reload(prompt_user)  # must not raise while PromptSession is unusable
    finally:
        monkeypatch.undo()
        importlib.reload(prompt_user)  # restore the real binding for other tests
