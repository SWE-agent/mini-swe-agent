# Controlled local resume

`mini-extra resumable` continues a task after an explicitly recorded pause. It preserves
the conversation, task variables, cumulative cost and call count, consecutive format
errors, and elapsed active time. Historical commands are never replayed by the loader.

This is an optional `ResumableAgent`, using `LocalEnvironment`. Commands run without
interactive confirmation; `agent_class` and `mode` settings are ignored with a notice.
The CLI supports `litellm` and `litellm_textbased`; the same
adapter, model, prompts, and environment configuration are used for continuation.

## Start and continue

Configure your model credentials as usual. Use a new directory outside the editable
workspace for the run records:

```bash
mini-extra resumable start \
  --task "Fix the failing test" \
  --cwd /path/to/project \
  --run-dir /path/to/runs/fix-test \
  -c mini.yaml -c model.model_name=openai/gpt-5 \
  --step-limit 10

mini-extra resumable resume \
  --run-dir /path/to/runs/fix-test \
  --step-limit 20
```

The second command raises the **whole task's** call limit to 20. It does not grant
20 additional calls. `--cost-limit` and `--wall-time-limit-seconds` also set cumulative
limits; zero disables a limit. As in `DefaultAgent`, limits are checked before the next
model request, so one request can exceed the cost limit. Time spent paused is excluded.
Process-wide `GLOBAL_MODEL_STATS` retains its existing per-process semantics; the
restored budgets belong to this task.
Cost limits depend on available pricing; `cost_tracking: ignore_errors` can record
zero cost for an unregistered model and therefore cannot enforce a monetary budget.

Use `--pause-after 2` to pause after two model calls in the current session, once the
resulting actions or format-error feedback have been fully recorded. This does not
change the cumulative limits. A submitted task is finished, not paused. A format error
still counts as a call and retains its billed cost and error streak across resumes.

The CLI prints the exit status and run directory. Submission and controlled pauses
return exit code 0; failures such as `EnvironmentError` and `RepeatedFormatError`
return a nonzero exit code, even when their trajectories were saved successfully.

For text-based actions, start with `-c default.yaml -c model.model_class=litellm_textbased`
and your model name. Resume reads that choice from the checkpoint.

## What can be resumed

Only checkpoints produced by this runner, marked `paused` after a controlled stop,
are accepted. The stop may be an explicit pause or a call, cost, or active-time limit
reached before the next request. All actions in a multi-action response must have
matching results. Legacy trajectories, incomplete histories, interrupted requests,
command timeouts, detected execution interruptions, ordinary exceptions, and completed
tasks cannot be resumed.
A reported execution exception or a negative command-shell return code stops the
batch; remaining tool calls are recorded as not executed. Positive exit codes are
completed observations and do not prevent a later controlled pause or resume.
The shell may report a child terminated by a signal as a positive code such as `137`,
or a wrapper may hide that failure. These cases do not stop the batch and are not
detected as interruptions. A controlled pause does not guarantee that child processes
completed successfully.

Keep the original run directory and working directory. The loader checks the recorded
paths and configuration, but does not snapshot or verify working-directory contents.
Do not modify the workspace externally while paused, or use background commands,
remote environments, or another agent operating on the same files. The model may
propose a command again in a later response; preventing that is outside the loader's
guarantee.

Resume only trusted, unmodified run records. Validation checks consistency but does
not authenticate their origin. Commands run with your user permissions, including
access outside the working directory. Configuration comparison covers saved settings,
not inherited environment variables. As with other trajectories, literal credentials
in model or environment settings are persisted; prefer environment-based model
credentials and keep run records private.

## Files and interruption behavior

Each session writes `<checkpoint-id>.traj.json`. The `latest` file selects the current
session. Trajectories retain the existing format and add versioned `info.resume`
metadata; a continuation records `parent_checkpoint_id` and leaves its parent intact.

Before any continuation request, the runner exclusively creates a
`<parent-checkpoint-id>.claimed` directory. This permanent claim prevents duplicate or
concurrent continuation of the same checkpoint. It is deliberately not released on
failure: after a crash, the runner cannot know whether an action already changed files.
Do not delete claims or rewind `latest` to retry an interrupted continuation.

The new session is saved as `running` before it becomes `latest` or issues a request.
Only a controlled stop publishes `paused`. Files use same-directory atomic replacement
so an interrupted write cannot leave a half-written destination. This covers local
process interruption, not machine power loss, filesystem damage, or distributed locks.
If a claim succeeds but subsequent setup fails, the checkpoint remains consumed and
requires manual investigation.

The deterministic regression tests cover A/B file effects, fresh-process continuation,
budget and format-error accounting, incomplete tool results, simultaneous resumes,
and processes killed after claiming or after an action but before its observation.
