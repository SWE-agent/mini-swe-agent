from prompt_toolkit.formatted_text.html import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession

from minisweagent import global_config_dir

_history = FileHistory(global_config_dir / "interactive_history.txt")


class _LazyPromptSession:
    """Defers PromptSession construction until the first prompt.

    PromptSession probes the terminal when it is constructed; on Windows with a
    non-console stdout (redirected output, or pytest's captured stdout) that
    probe raises NoConsoleScreenBufferError. Building the sessions at import
    time therefore broke importing the interactive agent on Windows, which the
    Linux-only CI never saw. Constructing on first use defers the probe to the
    only moment a real terminal is guaranteed to exist.
    """

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
        self._session: PromptSession | None = None

    def prompt(self, *args, **kwargs):
        if self._session is None:
            self._session = PromptSession(**self._kwargs)
        return self._session.prompt(*args, **kwargs)


prompt_session = _LazyPromptSession(history=_history)
_multiline_prompt_session = _LazyPromptSession(history=_history, multiline=True)


def _multiline_prompt() -> str:
    return _multiline_prompt_session.prompt(
        "",
        bottom_toolbar=HTML(
            "Submit message: <b fg='yellow' bg='black'>Esc, then Enter</b> | "
            "Navigate history: <b fg='yellow' bg='black'>Arrow Up/Down</b> | "
            "Search history: <b fg='yellow' bg='black'>Ctrl+R</b>"
        ),
    )
