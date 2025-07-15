<div align="center">

<img src="docs/assets/micro-swe-agent-banner.svg" alt="micro-swe-agent banner" style="height: 12em"/>
<h1>The 100 line AI agent that solves GitHub issues & more</h1>
</div>

In 2024, [SWE-bench](https://swe-bench.com) & [SWE-agent](https://swe-agent.com) helped kickstart the agentic AI for software revolution. We now ask:
**What if SWE-agent was 100x smaller, and still worked nearly as well?**

`micro` is for

- 🧪 **Researchers** who want to **benchmark, fine-tune or RL** without assumptions, bloat, or surprises
- 🧑‍💻 **Hackers & power users** who like their tools like their scripts: **short, sharp, and readable**
- 🐳 **Engineers** who want something **trivial to sandbox & to deploy anywhere**

Here's some details:

- **🐜 Minimal**: Just [100 lines of python](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microsweagent/agents/default.py) (+100 total for [env](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microsweagent/environments/local.py),
[model model](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microsweagent/models/litellm_model.py), [script](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microsweagent/run/hello_world.py)) — no fancy dependencies!
- **💪 Powerful:** Resolves XX% of GitHub issues in the [SWE-bench verified benchmark](https://www.swebench.com/).
- **🤗 Friendly:** Comes with **two convenient UIs** that will turn this into your daily dev swiss army knife!
- **🍀 Environments:** In addition to local envs, you can use **docker**, **podman**, **singularity**, **apptainer**, and more
- **🎓 Cutting edge:** Built by the Princeton & Stanford team behind [SWE-bench](https://swe-bench.com) and [SWE-agent](https://swe-agent.com).


<details>

<summary>More motivation (for research)</summary>

[SWE-agent](https://swe-agent.com/latest/) jump-started the development of AI agents in 2024. Back then, we placed a lot of emphasis on tools and special interfaces for the agent.
However, one year later, as LMs have become more capable, a lot of this is not needed at all to build a useful agent!
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

`micro` strives to be

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
<table>
<tr>
<td width="50%">
<strong>Simple UI</strong> (<code>micro</code>)
</td>
<td>
<strong>Textual UI</strong> (<code>micro2</code>)
</td>
</tr>
<tr>
<td width="50%">
<img width="600px" src="docs/assets/micro.png">
</td>
<td>
<img width="600px" src="docs/assets/micro2.png">
</td>
</tr>
<tr>
<td>
<strong>Python bindings</strong>
</td>
<td>
<strong>More in the docs</strong>
</td>
</tr>
<tr>
<td>

```python
agent = DefaultAgent(
    LitellmModel(model_name=...),
    LocalEnvironment(),
)
agent.run("Write a python sudoku game for me")
```
</td>
<td>

- TBD
</td>
</tr>
</table>

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

## 🚀 Developer version

```bash
git clone https://github.com/SWE-agent/micro-swe-agent.git
cd micro-swe-agent
pip install -e .
```

And welcome your new friend:

```bash
# Simple UI (microsweagent/run/local.py)
micro
# Textual UI (microsweagent/run/local2.py)
micro2
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



## ❤️ Contributing


