<div align="center">
<img src="assets/micro-swe-agent-banner.svg" alt="micro-swe-agent banner" style="height: 7em"/>
<h1>The 100 line AI agent that's actually useful</h1>
</div>


In 2024, [SWE-bench](https://swebench.com) & [SWE-agent](https://swe-agent.com) helped kickstart the agentic AI for software revolution. In 2025, we ask:
**What if the agent was 100x smaller, and still worked nearly as well?**

`micro` is for

- 🧪 **Researchers** who want to **benchmark, fine-tune or RL** without assumptions, bloat, or surprises
- 🧑‍💻 **Hackers & power users** who like their tools like their scripts: **short, sharp, and readable**
- 🐳 **Engineers** who want something **trivial to sandbox & to deploy anywhere**

Here's some details:

- **🐜 Minimal**: Just [100 lines of python](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microsweagent/agents/default.py) (+100 total for [env](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microsweagent/environments/micro.py),
[model model](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microsweagent/models/litellm_model.py), [script](https://github.com/SWE-agent/micro-swe-agent/blob/main/src/microsweagent/run/hello_world.py)) — no fancy dependencies!
- **💪 Powerful:** Resolves XX% of GitHub issues in the [SWE-bench verified benchmark](https://www.swebench.com/).
- **🤗 Friendly:** Comes with **two convenient UIs** that will turn this into your daily dev swiss army knife!
- **🍀 Environments:** In addition to local envs, you can use **docker**, **podman**, **singularity**, **apptainer**, and more
- **🎓 Cutting edge:** Built by the Princeton & Stanford team behind [SWE-bench](https://swebench.com) and [SWE-agent](https://swe-agent.com).

??? note "Why use micro-SWE-agent for research?"

    [SWE-agent](https://swe-agent.com/latest/) jump-started the development of AI agents in 2024. Back then, we placed a lot of emphasis on tools and special interfaces for the agent. However, one year later, a lot of this is not needed at all to build a useful agent!

    In fact, micro-SWE-agent:

    - **Does not have any tools other than bash** — it doesn't even use the tool-calling interface of the LMs. This means that you can run it with literally any model. When running in sandboxed environments you also don't need to take care of installing a single package — all it needs is bash.
    - **Has a completely linear history** — every step of the agent just appends to the messages and that's it. So there's no difference between the trajectory and the messages that you pass on to the LM.
    - **Executes actions with `subprocess.run`** — every action is completely independent (as opposed to keeping a stateful shell session running). This makes it trivial to execute the actions in sandboxes (literally just switch out `subprocess.run` with `docker exec`) and to scale up effortlessly.

    This makes it perfect as a baseline system and for a system that puts the language model (rather than the agent scaffold) in the middle of our attention.

??? note "Why use micro-SWE-agent as a tool?"

    Some agents are overfitted research artifacts. Others are UI-heavy tools, highly optimized for a specific user experience. Both variants are hard to understand.

    `micro` wants to be:

    - **Simple** enough to understand at a glance
    - **Convenient** enough to use in daily workflows
    - **Flexible** to extend

    A hackable tool, not a black box.

    Unlike other agents (including our own [swe-agent](https://swe-agent.com/latest/)), it is radically simpler, because it:

    - Does not have any tools other than bash — it doesn't even use the tool-calling interface of the LMs.
    - Has a completely linear history — every step of the agent just appends to the messages and that's it.
    - Executes actions with `subprocess.run` — every action is completely independent (as opposed to keeping a stateful shell session running).

??? note "Should I use SWE-agent or micro-SWE-agent?"

    You should use micro-swe-agent if

    - You want a quick command line tool that works locally
    - You want an agent with a very simple control flow
    - You want even faster, simpler & more stable sandboxing & benchmark evaluations

    You should use swe-agent if

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
<a href="https://mellow-pegasus-562d44.netlify.app/usage/micro/"><strong>Simple UI</strong></a> (<code>micro</code>)
</td>
<td>
<a href="https://mellow-pegasus-562d44.netlify.app/usage/micro_v/"><strong>Visual UI</strong></a> (<code>micro -v</code>)
</td>
</tr>
<tr>
<td width="50%">
  <img src="https://github.com/SWE-agent/swe-agent-media/blob/main/media/micro/gif/micro.gif?raw=true" alt="micro" />
</td>
<td>
  <img src="https://github.com/SWE-agent/swe-agent-media/blob/main/media/micro/gif/micro2.gif?raw=true" alt="microv" />
</td>
</tr>
<tr>
<td>
<a href="https://mellow-pegasus-562d44.netlify.app/usage/swebench/"><strong>Batch inference</strong></a>
</td>
<td>
<a href="https://mellow-pegasus-562d44.netlify.app/usage/inspector/"><strong>Trajectory browser</strong></a>
</td>
</tr>
<tr>
<td>
<img src="https://github.com/SWE-agent/swe-agent-media/blob/main/media/micro/gif/swebench.gif?raw=true" alt="swebench" />
</td>
<td>
<img src="https://github.com/SWE-agent/swe-agent-media/blob/main/media/micro/gif/inspector.gif?raw=true" alt="inspector" />
</td>
</tr>
<tr>
<td>
<a href="https://mellow-pegasus-562d44.netlify.app/advanced/cookbook/"><strong>Python bindings</strong></a>
</td>
<td>
<a href="https://mellow-pegasus-562d44.netlify.app"><strong>More in the docs</strong></a>
</td>
</tr>
<tr>
<td>
<pre><code class="language-python">agent = DefaultAgent(
    LitellmModel(model_name=...),
    LocalEnvironment(),
)
agent.run("Write a sudoku game")</code></pre>
</td>
<td>
<ul>
<li><a href="https://mellow-pegasus-562d44.netlify.app/quickstart/">Quick start</a></li>
<li><a href="https://mellow-pegasus-562d44.netlify.app/usage/micro/"><code>micro</code></a></li>
<li><a href="https://mellow-pegasus-562d44.netlify.app/faq/">FAQ</a></li>
<li><a href="https://mellow-pegasus-562d44.netlify.app/advanced/configuration/">Configuration</a></li>
<li><a href="https://mellow-pegasus-562d44.netlify.app/advanced/cookbook/">Power up</a></li>
</ul>
</td>
</tr>
</table>



## Continue reading:

<div class="grid cards">
  <a href="quickstart/" class="nav-card-link">
    <div class="nav-card">
      <div class="nav-card-header">
        <span class="material-icons nav-card-icon">launch</span>
        <span class="nav-card-title">Installation & Quick Start</span>
      </div>
      <p class="nav-card-description">Get started with micro-SWE-agent</p>
    </div>
  </a>

  <a href="usage/micro/" class="nav-card-link">
    <div class="nav-card">
      <div class="nav-card-header">
        <span class="material-icons nav-card-icon">flash_on</span>
        <span class="nav-card-title">Usage: Simple UI</span>
      </div>
      <p class="nav-card-description">Learn to use the <code>micro</code> command</p>
    </div>
  </a>

  <a href="usage/micro_v/" class="nav-card-link">
    <div class="nav-card">
      <div class="nav-card-header">
        <span class="material-icons nav-card-icon">visibility</span>
        <span class="nav-card-title">Usage: Visual UI</span>
      </div>
      <p class="nav-card-description">Try the visual interface with <code>micro -v</code></p>
    </div>
  </a>

  <a href="faq/" class="nav-card-link">
    <div class="nav-card">
      <div class="nav-card-header">
        <span class="material-icons nav-card-icon">help</span>
        <span class="nav-card-title">FAQ</span>
      </div>
      <p class="nav-card-description">Common questions and answers</p>
    </div>
  </a>

  <a href="advanced/configuration/" class="nav-card-link">
    <div class="nav-card">
      <div class="nav-card-header">
        <span class="material-icons nav-card-icon">settings</span>
        <span class="nav-card-title">Configuration</span>
      </div>
      <p class="nav-card-description">Setup and customize your agent</p>
    </div>
  </a>

  <a href="advanced/cookbook/" class="nav-card-link">
    <div class="nav-card">
      <div class="nav-card-header">
        <span class="material-icons nav-card-icon">fitness_center</span>
        <span class="nav-card-title">Power up</span>
      </div>
      <p class="nav-card-description">Start hacking the agent!</p>
    </div>
  </a>
</div>

{% include-markdown "_footer.md" %}
