<div align="center">
<a href="https://mini-swe-agent.com/latest/"><img src="https://github.com/SWE-agent/mini-swe-agent/raw/main/docs/assets/mini-swe-agent-banner.svg" alt="mini-swe-agent banner" style="height: 7em"/></a>
</div>

<h1>The 100 line AI agent that solves GitHub issues & more</h1>

[![Docs](https://img.shields.io/badge/Docs-green?style=for-the-badge&logo=materialformkdocs&logoColor=white)](https://mini-swe-agent.com/latest/)
[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://join.slack.com/t/swe-bench/shared_invite/zt-36pj9bu5s-o3_yXPZbaH2wVnxnss1EkQ)
[![PyPI - Version](https://img.shields.io/pypi/v/mini-swe-agent?style=for-the-badge&logo=python&logoColor=white&labelColor=black&color=deeppink)](https://pypi.org/project/mini-swe-agent/)

In 2024, [SWE-bench](https://github.com/swe-bench/SWE-bench) & [SWE-agent](https://github.com/swe-agent/swe-agent) helped kickstart the agentic AI for software revolution.

We now ask: **What if SWE-agent was 100x smaller, and still worked nearly as well?**

`mini` is for

- **Researchers** who want to **benchmark, fine-tune or RL** without assumptions, bloat, or surprises
- **Developers** who like their tools like their scripts: **short, sharp, and readable**
- **Engineers** who want something **trivial to sandbox & to deploy anywhere**

Here's some details:

- **Minimal**: Just [100 lines of python](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py) (+100 total for [env](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments/local.py),
[model](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/models/litellm_model.py), [script](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/run/hello_world.py)) — no fancy dependencies!
- **Powerful:** Resolves 65% of GitHub issues in the [SWE-bench verified benchmark](https://www.swebench.com/) (with Claude Sonnet 4).
- **Convenient:** Comes with UIs that turn this into your daily dev swiss army knife!
- **Deployable:** In addition to local envs, you can use **docker**, **podman**, **singularity**, **apptainer**, and more
- **Tested:** [![Codecov](https://img.shields.io/codecov/c/github/swe-agent/mini-swe-agent?style=flat-square)](https://codecov.io/gh/SWE-agent/mini-swe-agent)
- **Cutting edge:** Built by the Princeton & Stanford team behind [SWE-bench](https://swebench.com) and [SWE-agent](https://swe-agent.com).

<details>

<summary>More motivation (for research)</summary>

[SWE-agent](https://swe-agent.com/latest/) jump-started the development of AI agents in 2024. Back then, we placed a lot of emphasis on tools and special interfaces for the agent.
However, one year later, as LMs have become more capable, a lot of this is not needed at all to build a useful agent!
In fact, mini-SWE-agent

- Does not have any tools other than bash — it doesn't even use the tool-calling interface of the LMs.
  This means that you can run it with literally any model. When running in sandboxed environments you also don't need to to take care
  of installing a single package — all it needs is bash.
- Has a completely linear history — every step of the agent just appends to the messages and that's it.
  So there's no difference between the trajectory and the messages that you pass on to the LM.
- Executes actions with `subprocess.run` — every action is completely independent (as opposed to keeping a stateful shell session running).
  This makes it trivial to execute the actions in sandboxes (literally just switch out `subprocess.run` with `docker exec`) and to
  scale up effortlessly. Seriously, this is [a big deal](https://mini-swe-agent.com/latest/faq/#why-no-shell-session), trust me.

This makes it perfect as a baseline system and for a system that puts the language model (rather than
the agent scaffold) in the middle of our attention.

</details>

<details>
<summary>More motivation (as a tool)</summary>

Some agents are overfitted research artifacts. Others are UI-heavy frontend monsters.

`mini` wants to be a hackable tool, not a black box.

- **Simple** enough to understand at a glance
- **Convenient** enough to use in daily workflows
- **Flexible** to extend

Unlike other agents (including our own [swe-agent](https://swe-agent.com/latest/)), it is radically simpler, because it:

- Does not have any tools other than bash — it doesn't even use the tool-calling interface of the LMs.
- Has a completely linear history — every step of the agent just appends to the messages and that's it.
- Executes actions with `subprocess.run` — every action is completely independent (as opposed to keeping a stateful shell session running). This is [a big deal](https://mini-swe-agent.com/latest/faq/#why-no-shell-session), trust me.

</details>

<details>
<summary>Should I use SWE-agent or mini-SWE-agent?</summary>

You should use `mini-swe-agent` if

- You want a quick command line tool that works locally
- You want an agent with a very simple control flow
- You want even faster, simpler & more stable sandboxing & benchmark evaluations
- You are doing FT or RL and don't want to overfit to a specific agent scaffold

You should use `swe-agent` if

- You need specific tools or want to experiment with different tools
- You want to experiment with different history processors
- You want very powerful yaml configuration without touching code

What you get with both

- Excellent performance on SWE-Bench
- A trajectory browser

</details>

<table>
<tr>
<td width="50%">
<a href="https://mini-swe-agent.com/latest/usage/mini/"><strong>Simple UI</strong></a> (<code>mini</code>)
</td>
<td>
<a href="https://mini-swe-agent.com/latest/usage/mini_v/"><strong>Visual UI</strong></a> (<code>mini -v</code>)
</td>
</tr>
<tr>
<td width="50%">

  ![mini](https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/mini.gif?raw=true)

</td>
<td>

  ![miniv](https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/mini2.gif?raw=true)

</td>
</tr>
<tr>
  <td>
    <a href="https://mini-swe-agent.com/latest/usage/swebench/"><strong>Batch inference</strong></a>
  </td>
  <td>
    <a href="https://mini-swe-agent.com/latest/usage/inspector/"><strong>Trajectory browser</strong></a>
  </td>
<tr>
<tr>

<td>

![swebench](https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/swebench.gif?raw=true)

</td>

<td>

![inspector](https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/inspector.gif?raw=true)

</td>

</tr>
<td>
<a href="https://mini-swe-agent.com/latest/advanced/cookbook/"><strong>Python bindings</strong></a>
</td>
<td>
<a href="https://mini-swe-agent.com"><strong>More in the docs</strong></a>
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

* [Quick start](https://mini-swe-agent.com/latest/quickstart/)
* [`mini`](https://mini-swe-agent.com/latest/usage/mini/)
* [FAQ](https://mini-swe-agent.com/latest/faq/)
* [Configuration](https://mini-swe-agent.com/latest/advanced/configuration/)
* [Power up](https://mini-swe-agent.com/latest/advanced/cookbook/)

</td>
</tr>
</table>

## Let's get started!

Install + run in virtual environment

```bash
pip install uv && uvx mini-swe-agent [-v]
# or
pip install pipx && pipx ensurepath && pipx run mini-swe-agent [-v]
```

Alternative: Install in current environment

```bash
pip install mini-swe-agent && mini [-v]
```

Alternative: Install from source

```bash
git clone https://github.com/SWE-agent/mini-swe-agent.git
cd mini-swe-agent
pip install -e .
mini [-v]
```

Read more in our [documentation](https://mini-swe-agent.com/latest/):

* [Quick start guide](https://mini-swe-agent.com/latest/quickstart/)
* More on [`mini`](https://mini-swe-agent.com/latest/usage/mini/) and [`mini -v`](https://mini-swe-agent.com/latest/usage/mini_v/)
* [Configuration](https://mini-swe-agent.com/latest/advanced/configuration/)
* [Power up with the cookbook](https://mini-swe-agent.com/latest/advanced/cookbook/)
* [FAQ](https://mini-swe-agent.com/latest/faq/)
* [Contribute!](https://mini-swe-agent.com/latest/contributing/)

## Bottom line

If you found this work helpful, please consider citing

<details>
<summary> SWE-agent citation</summary>

```bibtex
@inproceedings{yang2024sweagent,
  title={{SWE}-agent: Agent-Computer Interfaces Enable Automated Software Engineering},
  author={John Yang and Carlos E Jimenez and Alexander Wettig and Kilian Lieret and Shunyu Yao and Karthik R Narasimhan and Ofir Press},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://arxiv.org/abs/2405.15793}
}
```
</details>

More agentic AI:

<div align="center">
  <a href="https://github.com/SWE-agent/SWE-agent"><img src="https://github.com/SWE-agent/mini-swe-agent/raw/main/docs/assets/sweagent_logo_text_below.svg" alt="SWE-agent" height="120px"></a>
   &nbsp;&nbsp;
  <a href="https://github.com/SWE-agent/SWE-ReX"><img src="https://github.com/SWE-agent/mini-swe-agent/raw/main/docs/assets/swerex_logo_text_below.svg" alt="SWE-ReX" height="120px"></a>
   &nbsp;&nbsp;
  <a href="https://github.com/SWE-bench/SWE-bench"><img src="https://github.com/SWE-agent/mini-swe-agent/raw/main/docs/assets/swebench_logo_text_below.svg" alt="SWE-bench" height="120px"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/SWE-bench/SWE-smith"><img src="https://github.com/SWE-agent/mini-swe-agent/raw/main/docs/assets/swesmith_logo_text_below.svg" alt="SWE-smith" height="120px"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/SWE-bench/sb-cli"><img src="https://github.com/SWE-agent/mini-swe-agent/raw/main/docs/assets/sbcli_logo_text_below.svg" alt="sb-cli" height="120px"></a>
</div>

