"""Structured ask-the-user action for interactive runs.

The model can emit an ``ask_user`` action instead of a shell command when it
needs mid-run clarification — e.g. picking between two valid approaches or
confirming an assumption. The question is rendered through the existing
prompt sessions and the answer is injected back as a normal observation.

An ``ask_user`` action is any action dict with a ``question`` key:

    {"question": "Apply the migration to all tenants or just tenant A?",
     "options": ["all tenants", "tenant A"]}            # optional choices

``options`` is optional; when present the user answers with a number or the
option text. Malformed, empty or cancelled answers become explicit
placeholder observations rather than silently attributed text (mirroring the
deepagents ``ask_user`` contract: never pretend the user said something they
didn't).

Multiple ``ask_user`` actions in one message are asked in order and answers
paired 1:1.
"""

from rich.console import Console

from minisweagent.agents.utils.prompt_user import prompt_session

console = Console(highlight=False)

_CANCELLED = "(cancelled)"


def is_ask_user(action: dict) -> bool:
    return isinstance(action.get("question"), str) and bool(action["question"].strip())


def ask_user(action: dict) -> dict:
    """Render the question through the prompt session and return an env-like
    output dict (``output``/``returncode``/``extra``) for the observation
    formatter. Never raises on bad input — every path yields an explicit
    answer string or a typed placeholder."""
    options = action.get("options")
    if not isinstance(options, list):
        options = []
    options = [str(o) for o in options if str(o).strip()]

    prompt_lines = [f"[bold yellow]{action['question'].strip()}[/bold yellow]"]
    for i, option in enumerate(options, start=1):
        prompt_lines.append(f"  [bold]{i}.[/bold] {option}")
    if options:
        prompt_lines.append("[dim]Answer with the option number, the option text, or free-form text.[/dim]")
    prompt_lines.append("[bold yellow]>[/bold yellow] ")
    console.print("\n".join(prompt_lines), end="")

    try:
        raw = prompt_session.prompt("")
    except (KeyboardInterrupt, EOFError):
        answer = _CANCELLED
    else:
        answer = raw.strip()
        if not answer:
            answer = "(no answer provided)"
        elif options:
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                answer = options[int(answer) - 1]
            elif answer not in options:
                # Free-form text is allowed even when options were offered;
                # keep it as-is rather than coercing to the nearest option.
                pass

    return {
        "output": answer,
        "returncode": 0,
        "exception_info": "",
        "extra": {
            "ask_user": True,
            "question": action["question"].strip(),
            "options": options,
            "cancelled": answer == _CANCELLED,
        },
    }
