<div align="center">
<img src="assets/mini-swe-agent-banner.svg" alt="mini-swe-agent banner" style="height: 7em"/>

<h1 style="margin-bottom: 1ex;">The 100 line AI agent that's actually useful</h1>

<a href="https://join.slack.com/t/swe-bench/shared_invite/zt-36pj9bu5s-o3_yXPZbaH2wVnxnss1EkQ">
    <img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white" alt="Slack">
</a>
<a href="https://github.com/SWE-agent/mini-swe-agent">
    <img alt="GitHub Release" src="https://img.shields.io/github/v/release/swe-agent/mini-swe-agent?style=for-the-badge&logo=github&label=GitHub&labelColor=black&color=green" alt="GitHub Release">
</a>
<a href="https://pypi.org/project/mini-swe-agent/">
    <img src="https://img.shields.io/pypi/v/mini-swe-agent?style=for-the-badge&logo=python&logoColor=white&labelColor=black&color=deeppink" alt="PyPI - Version">
</a>

</div>
In 2024, [SWE-bench](https://swebench.com) & [SWE-agent](https://swe-agent.com) helped kickstart the coding agent revolution.

We now ask: **What if the agent was 100x smaller, and still worked nearly as well?**

`mini` is for

- **Researchers** who want to **[benchmark](https://swe-bench.com), [fine-tune](https://swesmith.com/) or RL** without assumptions, bloat, or surprises
- **Developers** who like their tools like their scripts: **short, sharp, and readable**
- **Engineers** who want something **trivial to sandbox & to deploy anywhere**

Here's some details:

- **Minimal**: Just [100 lines of python](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py) (+100 total for [env](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments/local.py),
[model](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/models/litellm_model.py), [script](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/run/hello_world.py)) — no fancy dependencies!
- **Powerful:** Resolves >74% of GitHub issues in the [SWE-bench verified benchmark](https://www.swebench.com/) ([leaderboard](https://swe-bench.com/)).
- **Convenient:** Comes with UIs that turn this into your daily dev swiss army knife!
- **Deployable:** In addition to local envs, you can use **docker**, **podman**, **singularity**, **apptainer**, and more
- **Cutting edge:** Built by the Princeton & Stanford team behind [SWE-bench](https://swebench.com) and [SWE-agent](https://swe-agent.com).

??? note "Why use mini-SWE-agent for research?"

    [SWE-agent](https://swe-agent.com/latest/) jump-started the development of AI agents in 2024. Back then, we placed a lot of emphasis on tools and special interfaces for the agent. However, one year later, a lot of this is not needed at all to build a useful agent!

    In fact, mini-SWE-agent:

    - **Does not have any tools other than bash** — it doesn't even use the tool-calling interface of the LMs.
      This means that you can run it with literally any model.
      When running in sandboxed environments you also don't need to take care of installing a single package — all it needs is bash.
    - **Has a completely linear history** — every step of the agent just appends to the messages and that's it.
      So there's no difference between the trajectory and the messages that you pass on to the LM.
      Great for debugging & fine-tuning.
    - **Executes actions with `subprocess.run`** — every action is completely independent (as opposed to keeping a stateful shell session running). This makes it trivial to execute the actions in sandboxes (literally just switch out `subprocess.run` with `docker exec`) and to scale up effortlessly.
      Seriously, this is [a big deal](faq.md#why-no-shell-session), trust me.

    This makes it perfect as a baseline system and for a system that puts the language model (rather than the agent scaffold) in the middle of our attention.
    You can see the result on the [SWE-bench (bash only)](https://www.swebench.com/) leaderboard, that evaluates the performance of different LMs with `mini`.

??? note "Why use mini-SWE-agent as a tool?"

    Some agents are overfitted research artifacts. Others are UI-heavy frontend monsters.

    `mini` wants to be a hackable tool, not a black box.

    - **Simple** enough to understand at a glance
    - **Convenient** enough to use in daily workflows
    - **Flexible** to extend

    Unlike other agents (including our own [swe-agent](https://swe-agent.com/latest/)), it is radically simpler, because it:

    - **Does not have any tools other than bash** — it doesn't even use the tool-calling interface of the LMs.
      Instead of implementing custom tools for every specific thing the agent might want to do, the focus is fully on the LM utilizing the shell to its full potential.
      Want it to do something specific like opening a PR?
      Just tell the LM to figure it out rather than spending time to implement it in the agent.
    - **Executes actions with `subprocess.run`** — every action is completely independent (as opposed to keeping a stateful shell session running).
      This is [a big deal](https://mini-swe-agent.com/latest/faq/#why-no-shell-session) for the stability of the agent, trust me.
    - **Has a completely linear history** — every step of the agent just appends to the messages that are passed to the LM in the next step and that's it.
      This is great for debugging and understanding what the LM is prompted with.

??? note "Should I use mini-SWE-agent or swe-agent?"

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
<a href="usage/mini"><strong>Simple UI</strong></a> (<code>mini</code>)
</td>
<td>
<a href="usage/mini_v"><strong>Visual UI</strong></a> (<code>mini -v</code>)
</td>
</tr>
<tr>
<td width="50%">
  <div class="gif-container" data-glightbox-disabled>
    <img src="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/png/mini.png?raw=true"
         data-gif="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/mini.gif?raw=true"
         alt="mini" data-glightbox="false" />
  </div>
</td>
<td>
  <div class="gif-container" data-glightbox-disabled>
    <img src="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/png/mini2.png?raw=true"
         data-gif="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/mini2.gif?raw=true"
         alt="miniv" data-glightbox="false" />
  </div>
</td>
</tr>
<tr>
<td>
<a href="usage/swebench/"><strong>Batch inference</strong></a>
</td>
<td>
<a href="usage/inspector/"><strong>Trajectory browser</strong></a>
</td>
</tr>
<tr>
<td>
<div class="gif-container" data-glightbox-disabled>
  <img src="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/png/swebench.png?raw=true"
       data-gif="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/swebench.gif?raw=true"
       alt="swebench" data-glightbox="false" />
</div>
</td>
<td>
<div class="gif-container" data-glightbox-disabled>
  <img src="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/png/inspector.png?raw=true"
       data-gif="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/inspector.gif?raw=true"
       alt="inspector" data-glightbox="false" />
</div>
</td>
</tr>
<tr>
<td>
<a href="advanced/cookbook/"><strong>Python bindings</strong></a>
</td>
<td>
<a href="https://mini-swe-agent.com"><strong>More in the docs</strong></a>
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
<li><a href="quickstart">Quick start</a></li>
<li><a href="usage/mini"><code>mini</code></a></li>
<li><a href="faq">FAQ</a></li>
<li><a href="advanced/global_configuration">Global configuration</a></li>
<li><a href="advanced/yaml_configuration">Yaml configuration</a></li>
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
      <p class="nav-card-description">Get started with mini-SWE-agent</p>
    </div>
  </a>

  <a href="usage/mini/" class="nav-card-link">
    <div class="nav-card">
      <div class="nav-card-header">
        <span class="material-icons nav-card-icon">flash_on</span>
        <span class="nav-card-title">Usage: Simple UI</span>
      </div>
      <p class="nav-card-description">Learn to use the <code>mini</code> command</p>
    </div>
  </a>

  <a href="usage/mini_v/" class="nav-card-link">
    <div class="nav-card">
      <div class="nav-card-header">
        <span class="material-icons nav-card-icon">visibility</span>
        <span class="nav-card-title">Usage: Visual UI</span>
      </div>
      <p class="nav-card-description">Try the visual interface with <code>mini -v</code></p>
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

  <a href="advanced/yaml_configuration/" class="nav-card-link">
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

## 📣 News

* Nov 19: [Gemini 3 Pro reaches 74% on SWE-bench verified with mini-swe-agent!](https://x.com/KLieret/status/1991164693839270372)
* Aug 19: [New blogpost: Randomly switching between GPT-5 and Sonnet 4 boosts performance](https://www.swebench.com/SWE-bench/blog/2025/08/19/mini-roulette/)

## 📣 New features

Please check the [github release notes](https://github.com/SWE-agent/mini-swe-agent/releases) for the latest updates.

## 📣 Documentation updates

* Jul 27: More notes on [local models](models/local_models.md)

{% include-markdown "_footer.md" %}
