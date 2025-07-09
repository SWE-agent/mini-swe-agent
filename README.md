<div align="center">

<img src="docs/assets/micro-swe-agent-banner.svg" alt="micro-swe-agent banner" style="height: 12em"/>
<h1>The 100 line AI agent that solves GitHub issues & more</h1>
</div>

- **🐜 Minimal**: Just [100 lines of python](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/agents/default.py) (+100 for [env](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/environments/local.py),
[model model](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/models/litellm_model.py), [script](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/run/hello_world.py)) — no fancy dependencies!
Optionally, a few more lines for interactive UIs.
- **💪 Powerful:** Resolves XX% of GitHub issues in the [SWE-bench verified benchmark](https://www.swebench.com/).
- **🤗 Friendly:** Comes with **two convenient UIs** that will turn this into your daily dev swiss army knife!
- **🎓 Cutting edge:** Built by the Princeton & Stanford team behind [SWE-bench](https://swe-bench.com) and [SWE-agent](https://swe-agent.com).

Use it to

- 🔥 Instantly solve problems: `pip install pipx && pipx run micro-swe-agent`
- ⚙️ Take full control & quickly build custom agents
- 🏋 Fine-tune & RL with a minimal, assumption-free agent
- 🐳 Deploy seamlessly to sandboxed environments & CI/CD


<details>

<summary>More motivation (for research)</summary>

[SWE-agent](https://swe-agent.com/latest/) jump-started the development of AI agents in 2024. Back then, we placed a lot of emphasis on tools and special interfaces for the agent.
However, one year later, a lot of this is not needed at all to build a useful agent!
In fact, micro-SWE-agent

- Does not have any tools other than bash — it doesn't even use the tool-calling interface of the LMs.
  This means that you can run it with literally any model. When running in sandboxed environments you also don't need to to take care
  of installing a single package — all it needs is bash.
- Has a completely linear history — every step of the agent just appends to the messages and that's it.
  So there's no difference between the trajectory and the messages that you pass on to the LM.
- Executes actions with `subprocess.run` — every action is completely independent (as opposed to keeping a stateful shell session running).
  This makes it trivial to execute the actions in sandboxes (literally just switch out `subprocess.run` with `docker exec`) and to
  scale up effortlessly.

This makes it perfect as a baseline system and for a system that puts the language model (rather than
the agent scaffold) in the middle of our attention.

</details>

<details>
<summary>More motivation (as a tool)</summary>

Some agents are overfitted research artifacts.
Others are UI-heavy tools, highly optimized for a specific user experience.
Both variants are hard to understand.

`micro` wants to be

- **Simple** enough to understand at a glance
- **Convenient** enough to use in daily workflows
- **Flexible** to extend

A hackable tool, not a black box.

Unlike other agents (including our own [swe-agent](https://swe-agent.com/latest/)),
it is radically simpler, because it

- Does not have any tools other than bash — it doesn't even use the tool-calling interface of the LMs.
- Has a completely linear history — every step of the agent just appends to the messages and that's it.
- Executes actions with `subprocess.run` — every action is completely independent (as opposed to keeping a stateful shell session running).

</details>

<div align="center">

**Simple UI**<br/>
<img width=600px src="docs/assets/micro.png">

**Textual UI**<br/>
<img width=600px src="docs/assets/micro2.png">

</div>

## 🔥 Try it!

(This will get simpler once we publish to pypi)

```bash
pip install pipx
# Simple UI
pipx run --spec git+ssh://git@github.com/SWE-agent/micro-swe-agent micro
# Textual UI
pipx run --spec git+ssh://git@github.com/SWE-agent/micro-swe-agent micro2
```

<details>

<summary>Details</summary>

[pipx](https://pipx.pypa.io/stable/) will install `micro-swe-agent` in an isolated environment and directly run it.
If the invocation doesn't immediately work, you might need to run `pipx ensurepath`.
After the first time, it's enough to just run `pipx run micro-swe-agent`.

</details>

<details>

<summary>Example prompts</summary>

- Implement a Sudoku solver in python in the `sudoku` folder. Make sure the codebase is modular and well tested with pytest.
- Please run pytest on the current project, discover failing unittests and help me fix them. Always make sure to test the final solution.
- Help me document & type my codebase by adding short docstrings and type hints.

</details>

## 🚀 Developer version & python bindings

```bash
git clone https://github.com/SWE-agent/micro-swe-agent.git
cd micro-swe-agent
pip install -e .
```

And welcome your new friend:

```bash
# Simple UI (microswea/run/local.py)
micro
# Textual UI (microswea/run/local2.py)
micro2
```

Or with python bindings:

```python
agent = DefaultAgent(
    AnthropicModel(model_name="claude-sonnet-4-20250514"),
    LocalEnvironment(),
    "Write a python sudoku game for me",
)
agent.run()
```

## ⚙️ Configure

All global configuration can be either set as environment variables, or
in the `.env` file (the exact location is printed when you run `micro`).

```bash
# set default config for micro
MSWEA_LOCAL_CONFIG_PATH="/path/to/your/own/config"
# set default model
MSWEA_MODEL_NAME="claude-sonnet-4-20250514"
```

## ⚡️ Power up <a target="powerup"/>

We provide several different entry points to the agent,
for example [hello world](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/run/hello_world.py),
or the [default when calling `micro`](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/run/local.py).

Want to cook up your custom version and the config is not enough?
Just follow the recipe below:

1. What's the control flow you need? Pick an [agent class](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/agents) (e.g., [simplest example](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/agents/default.py), [with human in the loop](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/agents/interactive.py))
2. How should actions be executed? Pick an [environment class](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/environments) (e.g., [local](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/environments/local.py), or [docker](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/environments/docker.py))
3. How is the LM queried? Pick a [model class](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/models) (e.g., [litellm](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/models/litellm_model.py))
4. How to invoke the agent? Bind them all together in a [run script](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/run), possibly reading from a [config](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/config) (e.g., [hello world](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/run/hello_world.py), or [`micro` entry point](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microswea/run/local.py))

We aim to keep all of these components very simple, but offer lots of choice between them -- enough to cover a broad range of
things that you might want to do.

You can override the default entry point by setting the `MSWEA_DEFAULT_RUN` environment variable to the import path of your run script.

## ❤️ Contributing

We happily accept contributions!

<details>
<summary>Areas of help</summary>

- Support for more models (anything where `litellm` doesn't work out of the box)
- Documentation, examples, tutorials, etc.
- Support for more environments & deployments (e.g., run it as a github action, etc.)
- Take a look at the [issues](https://github.com/SWE-agent/micro-swe-agent/issues) and see if there's anything you'd like to work on!

</details>

<details>
<summary>Design & Architecture</summary>

- `micro-swe-agent` aims to stay minimalistic & hackable
- To extend features, we prefer to add a new version of the one of the four components above, rather than making the existing components more complex
- Components should be relatively self-contained, but if there are utilities that might be shared, add a `utils` folder (like [this one](https://github.com/SWE-agent/micro-swe-agent/tree/main/src/microswea/models/utils)). But keep it simple!
- If your component is a bit more specific, add it into an `extra` folder (like [this one](https://github.com/SWE-agent/micro-swe-agent/tree/main/src/microswea/run/extra))
- Our target audience is anyone who doesn't shy away from modifying a bit of code (especially a run script) to get what they want
- Therefore, not everything needs to be configurable with the config files, but it should be easy to use with a run script
- Many LMs write very verbose code -- please clean it up! Same goes for the tests. They should still be concise and readable.
- Please install `pre-commit` (`pip install pre-commit && pre-commit install`) and run it before committing. This will enforce our style guide.

</details>
