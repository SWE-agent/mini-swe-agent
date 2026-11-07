# SWE-rex Remote

!!! note "SWE-rex Remote Environment class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/environments/extra/swerex_remote.py)

Runs commands on a `swerex-remote` server that you started yourself. The server needs `bash`, and `env` unless you change `interpreter`.

On the remote machine, bind the server to localhost, because it has no TLS and otherwise listens on all interfaces:

```bash
swerex-remote --host 127.0.0.1 --port 8000 --auth-token "$SWEREX_AUTH_TOKEN"
```

On your machine, forward the port and set the same token. Use the environment variable or your `.env` file, because the token is not accepted as a config option:

```bash
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
    - **Nothing is cleaned up.** Children of a timed-out command keep running, and so does anything else the agent starts.
    - **Output is not limited.** Very large command output can make the server run out of memory.

::: minisweagent.environments.extra.swerex_remote

{% include-markdown "../../_footer.md" %}
