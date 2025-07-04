<div align="center">

<img src="docs/assets/micro-swe-agent-banner.svg" alt="micro-swe-agent banner" style="height: 12em"/>
<h1>The 100 line AI agent that solves GitHub issues & more</h1>

</div>

`micro-SWE-agent` provides

- An AI agent implemented in [100 lines of python](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/agents/default.py)!
Okay, maybe add another 100 lines total for the [environment](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/environments/local.py),
[model config](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/models/litellm_model.py),
and [run script](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/run/hello_world.py) -- but that's it, and no fancy packages used!
This still packs a punch though: We resolve XX% of GitHub issues in the [SWE-bench verified benchmark](https://www.swebench.com/).
- A few slightly extended variants that make turn this into your daily dev swiss army knive!
- Lots of additional [power-ups](#powering-up) to choose from to turn this into your personal tool!

Use `micro-SWE-agent` to

- 🔥 Run a powerful tool instantly without installation: `pip install pipx && pipx run micro-swe-agent`
- ⚙️ Take full control & quickly prototype your own agent ideas
- 🏋 Leverage a lean assumptions-free system made for reinforcement learning and finetuning
- 🐳 Trivially deploy to sandboxed environments

The project builds on our experience building [SWE-agent](https://swe-agent.com), one of the earliest and most successful software engineering agents for research.

## 🔥 Run as a tool without permanent installation <a target="fire"/>

```bash
export MSWEA_MODEL_NAME="claude-sonnet-4-20250514"  # your favorite model
export ANTHROPIC_API_KEY="xxx"  # API key for your model
pip install pipx && pipx run --spec git+ssh://git@github.com/SWE-agent/micro-swe-agent micro-swe-agent
```

## Install & experiment

```bash
git clone https://github.com/SWE-agent/micro-swe-agent.git
cd micro-swe-agent
pip install -e .
```

Now you can run the default local agent as

```bash
micro
```

But you can also just run the executables as

```bash
python microswea/run/local.py
```

## Configuration

You can put your LM API keys in a `.env` at the repository root or make sure they're set in your shell.
You can also set your default configs like so:

```bash
MSWEA_LOCAL_CONFIG_PATH="/path/to/your/own/config"  # override the default config for `micro`
```

## Powering up <a target="powerup"/>

We provide several different entry points to the agent,
for example [hello world](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/run/hello_world.py),
or the [default when calling `micro`](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/run/local.py).

Want to cook up your custom version and the config is not enough?
Just follow the recipe below:

1. What's the control flow you need? Pick an [agent class](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/agents) (e.g., [simplest example](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/agents/default.py), [with human in the loop](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/agents/interactive.py))
2. How should actions be executed? Pick an [environment class](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/environments) (e.g., [local](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/environments/local.py), or [docker](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/environments/docker.py))
3. How is the LM queried? Pick a [model class](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/models) (e.g., [litellm](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/models/litellm_model.py))
4. How to invoke the agent? Bind them all together in a [run script](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/run), possibly reading from a [config](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/config) (e.g., [hello world](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/run/hello_world.py), or [`micro` entry point](https://github.com/SWE-agent/micro-swe-agent/blob/main/microswea/run/local.py))

We aim to keep all of these components very simple, but offer lots of choice between them -- enough to cover a broad range of
things that you might want to do.

You can override the default entry point by setting the `MSWEA_DEFAULT_RUN` environment variable to the import path of your run script.

## Contributing

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
- Components should be relatively self-contained, but if there are utilities that might be shared, add a `utils` folder (like [this one](https://github.com/SWE-agent/micro-swe-agent/tree/main/microswea/models/utils)). But keep it simple!
- If your component is a bit more specific, add it into an `extra` folder (like [this one](https://github.com/SWE-agent/micro-swe-agent/tree/main/microswea/run/extra))
- Our target audience is anyone who doesn't shy away from modifying a bit of code (especially a run script) to get what they want
- Therefore, not everything needs to be configurable with the config files, but it should be easy to use with a run script
- Many LMs write very verbose code -- please clean it up! Same goes for the tests. They should still be concise and readable.
- Please install `pre-commit` (`pip install pre-commit && pre-commit install`) and run it before committing. This will enforce our style guide.

</details>
