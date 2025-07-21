<div align="center">

<a href="https://mellow-pegasus-562d44.netlify.app/"><img src="https://github.com/SWE-agent/micro-swe-agent/raw/main/docs/assets/micro-swe-agent-banner.svg" alt="micro-swe-agent banner" style="height: 7em"/></a>

<h1>The 100 line AI agent that solves GitHub issues & more</h1>

[![Docs](https://img.shields.io/badge/Docs-green?style=for-the-badge&logo=materialformkdocs&logoColor=white)](https://mellow-pegasus-562d44.netlify.app/)
[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://join.slack.com/t/swe-bench/shared_invite/zt-36pj9bu5s-o3_yXPZbaH2wVnxnss1EkQ)

</div>

In 2024, [SWE-bench](https://swebench.com) & [SWE-agent](https://swe-agent.com) helped kickstart the agentic AI for software revolution.

We now ask: **What if SWE-agent was 100x smaller, and still worked nearly as well?**

`micro` is for

- 🧪 **Researchers** who want to **benchmark, fine-tune or RL** without assumptions, bloat, or surprises
- 🧑‍💻 **Hackers & power users** who like their tools like their scripts: **short, sharp, and readable**
- 🐳 **Engineers** who want something **trivial to sandbox & to deploy anywhere**

Here's some details:

- **🐜 Minimal**: Just [100 lines of python](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microsweagent/agents/default.py) (+100 total for [env](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microsweagent/environments/local.py),
[model](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microsweagent/models/litellm_model.py), [script](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microsweagent/run/hello_world.py)) — no fancy dependencies!
- **💪 Powerful:** Resolves XX% of GitHub issues in the [SWE-bench verified benchmark](https://www.swebench.com/).
- **🤗 Friendly:** Comes with **two convenient UIs** that will turn this into your daily dev swiss army knife!
- **🍀 Environments:** In addition to local envs, you can use **docker**, **podman**, **singularity**, **apptainer**, and more
- **🧪 Tested:** [![codecov](https://codecov.io/gh/SWE-agent/micro-swe-agent/graph/badge.svg?token=NlwwZAO3o5)](https://codecov.io/gh/SWE-agent/micro-swe-agent)
- **🎓 Cutting edge:** Built by the Princeton & Stanford team behind [SWE-bench](https://swebench.com) and [SWE-agent](https://swe-agent.com).

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
<a href="https://mellow-pegasus-562d44.netlify.app/usage/micro/"><strong>Simple UI</strong></a> (<code>micro</code>)
</td>
<td>
<a href="https://mellow-pegasus-562d44.netlify.app/usage/micro_v/"><strong>Visual UI</strong></a> (<code>micro -v</code>)
</td>
</tr>
<tr>
<td width="50%">

  ![micro](https://github.com/SWE-agent/swe-agent-media/blob/main/media/micro/gif/micro.gif?raw=true)

</td>
<td>

  ![microv](https://github.com/SWE-agent/swe-agent-media/blob/main/media/micro/gif/micro2.gif?raw=true)

</td>
</tr>
<tr>
  <td>
    <a href="https://mellow-pegasus-562d44.netlify.app/usage/swebench/"><strong>Batch inference</strong></a>
  </td>
  <td>
    <a href="https://mellow-pegasus-562d44.netlify.app/usage/inspector/"><strong>Trajectory browser</strong></a>
  </td>
<tr>
<tr>

<td>

![swebench](https://github.com/SWE-agent/swe-agent-media/blob/main/media/micro/gif/swebench.gif?raw=true)

</td>

<td>

![inspector](https://github.com/SWE-agent/swe-agent-media/blob/main/media/micro/gif/inspector.gif?raw=true)

</td>

</tr>
<td>
<a href="https://mellow-pegasus-562d44.netlify.app/advanced/cookbook/"><strong>Python bindings</strong></a>
</td>
<td>
<a href="https://mellow-pegasus-562d44.netlify.app"><strong>More in the docs</strong></a>
</td>
</tr>
<tr>
<td>

```python
agent = DefaultAgent(
    LitellmModel(model_name=...),
    LocalEnvironment(),
)
agent.run("Write a sudoku game")
```
</td>
<td>

* [Quick start](https://mellow-pegasus-562d44.netlify.app/quickstart/)
* [`micro`](https://mellow-pegasus-562d44.netlify.app/usage/micro/)
* [FAQ](https://mellow-pegasus-562d44.netlify.app/faq/)
* [Configuration](https://mellow-pegasus-562d44.netlify.app/advanced/configuration/)
* [Power up](https://mellow-pegasus-562d44.netlify.app/advanced/cookbook/)

</td>
</tr>
</table>

## 🔥 Let's get started!

(This will get simpler once we publish to PyPI)

```bash
pip install pipx
pipx run --spec git+ssh://git@github.com/SWE-agent/micro-swe-agent micro [-v]
```

Read more in our [documentation](https://mellow-pegasus-562d44.netlify.app/):

* [Quick start guide](https://mellow-pegasus-562d44.netlify.app/quickstart/)
* More on [`micro`](https://mellow-pegasus-562d44.netlify.app/usage/micro/) and [`micro -v`](https://mellow-pegasus-562d44.netlify.app/usage/micro_v/)
* [Configuration](https://mellow-pegasus-562d44.netlify.app/advanced/configuration/)
* [Power up with the cookbook](https://mellow-pegasus-562d44.netlify.app/advanced/cookbook/)
* [FAQ](https://mellow-pegasus-562d44.netlify.app/faq/)
* [Contribute!](https://mellow-pegasus-562d44.netlify.app/contributing/)

## 👀 More agentic AI

<div align="center">
  <a href="https://github.com/SWE-agent/SWE-agent"><img src="https://github.com/SWE-agent/micro-swe-agent/raw/main/docs/assets/sweagent_logo_text_below.svg" alt="SWE-agent" height="120px"></a>
   &nbsp;&nbsp;
  <a href="https://github.com/SWE-agent/SWE-ReX"><img src="https://github.com/SWE-agent/micro-swe-agent/raw/main/docs/assets/swerex_logo_text_below.svg" alt="SWE-ReX" height="120px"></a>
   &nbsp;&nbsp;
  <a href="https://github.com/SWE-bench/SWE-bench"><img src="https://github.com/SWE-agent/micro-swe-agent/raw/main/docs/assets/swebench_logo_text_below.svg" alt="SWE-bench" height="120px"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/SWE-bench/SWE-smith"><img src="https://github.com/SWE-agent/micro-swe-agent/raw/main/docs/assets/swesmith_logo_text_below.svg" alt="SWE-smith" height="120px"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/SWE-bench/sb-cli"><img src="https://github.com/SWE-agent/micro-swe-agent/raw/main/docs/assets/sbcli_logo_text_below.svg" alt="sb-cli" height="120px"></a>
</div>

