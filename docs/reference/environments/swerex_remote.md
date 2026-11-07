# SWE-rex Remote

!!! note "SWE-rex Remote Environment class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/environments/extra/swerex_remote.py)

Runs commands on a `swerex-remote` server that you started yourself. The server needs `env`, `uname`, and `bash` unless you change `interpreter`.

On the remote machine, install SWE-ReX and start the server. Bind it to localhost, because it has no TLS and otherwise listens on all interfaces. Don't export the token, because commands inherit the server's environment. An empty token turns authentication off:

```bash
pip install swe-rex
SWEREX_AUTH_TOKEN=$(openssl rand -hex 32); echo "$SWEREX_AUTH_TOKEN"  # do not export
swerex-remote --host 127.0.0.1 --port 8000 --auth-token "${SWEREX_AUTH_TOKEN:?}"
```

On your machine, install the extras, forward the port and set the same token. Use the environment variable or your `.env` file, because the token is not accepted as a config option:

```bash
pip install "mini-swe-agent[full]"
ssh -N -L 8000:127.0.0.1:8000 my-remote-machine &
export SWEREX_AUTH_TOKEN=...
```

```yaml
environment:
  environment_class: swerex_remote
  cwd: /path/to/repo
```

!!! warning "Things to keep in mind"

    - **Use one server per worker.** All runs on a server share its filesystem, so parallel or batch runs can mix up each other's changes. The server also runs one command at a time, so a long command makes other clients fail their alive check. Prepare the workspace yourself: benchmark images are not applied.
    - **The agent can read the token.** It is visible in the server's process list, so it can end up in trajectories. Run the server as a separate user and change the token before you share trajectories.
    - **`swebench.yaml` assumes root.** It sets `BASH_ENV` to `/root/.bashrc`. As any other user, override `environment.env.BASH_ENV`, because the resulting error hides the submission from the agent.
    - **Nothing is cleaned up.** Children of a timed-out command keep running, and so does anything else the agent starts.
    - **Output is not limited.** Very large command output can make the server run out of memory.

::: minisweagent.environments.extra.swerex_remote

{% include-markdown "../../_footer.md" %}
