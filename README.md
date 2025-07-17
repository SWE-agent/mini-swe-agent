<div align="center">

<a href="https://mellow-pegasus-562d44.netlify.app/"><img src="docs/assets/micro-swe-agent-banner.svg" alt="micro-swe-agent banner" style="height: 12em"/></a>
<h1>The 100 line AI agent that solves GitHub issues & more</h1>
</div>

In 2024, [SWE-bench](https://swe-bench.com) & [SWE-agent](https://swe-agent.com) helped kickstart the agentic AI for software revolution.

We now ask: **What if SWE-agent was 100x smaller, and still worked nearly as well?**

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
<a href="https://mellow-pegasus-562d44.netlify.app/usage/micro/"><strong>Simple UI</strong></a> (<code>micro</code>)
</td>
<td>
<a href="https://mellow-pegasus-562d44.netlify.app/usage/micro2/"><strong>Visual UI</strong></a> (<code>micro -v</code>)
</td>
</tr>
<tr>
<td width="50%">

  ![micro](https://github.com/user-attachments/assets/b4ac427a-8626-4516-8b5d-8b42e389e0be)

</td>
<td>

  ![microv](https://github.com/user-attachments/assets/bbdea603-1ddc-4608-8429-b11ba59cfe99)

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

![swebench](https://github.com/user-attachments/assets/64caad12-c47d-4148-a65c-df22f2c35cde)

</td>

<td>

![inspector](https://github.com/user-attachments/assets/d4ec52a9-a48e-4560-b4ff-36ba4658274f)

</td>

</tr>
<td>
<a href="https://mellow-pegasus-562d44.netlify.app/cookbook/"><strong>Python bindings</strong></a>
</td>
<td>
<a href="https://mellow-pegasus-562d44.netlify.app/cookbook/"><strong>More in the docs</strong></a>
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
* [Configuration](https://mellow-pegasus-562d44.netlify.app/configuration/)
* [Power up](https://mellow-pegasus-562d44.netlify.app/cookbook/)

</td>
</tr>
</table>

## 🔥 Let's get started!

(This will get simpler once we publish to pypi)

```bash
pip install pipx
pipx run --spec git+ssh://git@github.com/SWE-agent/micro-swe-agent micro [-v]
```

Read more in our [documentation](https://mellow-pegasus-562d44.netlify.app/):

* [Quick start guide](https://mellow-pegasus-562d44.netlify.app/quickstart/)
* More on [`micro`](https://mellow-pegasus-562d44.netlify.app/usage/micro/) and [`micro2`](https://mellow-pegasus-562d44.netlify.app/usage/micro2/)
* [Configuration](https://mellow-pegasus-562d44.netlify.app/configuration/)
* [Power up with the cookbook](https://mellow-pegasus-562d44.netlify.app/cookbook/)
* [FAQ](https://mellow-pegasus-562d44.netlify.app/faq/)
* [Contribute!](https://mellow-pegasus-562d44.netlify.app/contributing/)

