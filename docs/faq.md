# FAQ

## General

!!! question "Does micro-SWE-agent work on my system?"

    micro-SWE-agent should work on any system that has a bash shell or uses a container runtime to emulate one.

??? question "Should I use SWE-agent or micro-SWE-agent?"

    You should use swe-agent if

    - You need specific tools or want to experiment with different tools
    - You want to experiment with different history processors
    - You want very powerful yaml configuration without touching code

    You should use micro-swe-agent if

    - You want a quick command line tool that works locally
    - You want an agent with a very simple control flow
    - You want even faster, simpler & more stable sandboxing & benchmark evaluations

    What you get with both

    - Excellent performance on SWE-Bench
    - A trajectory browser

!!! question "What models do you support?"


??? question "What are the limitations of micro-SWE-agent?"

    micro-SWE-agent can be extended trivially in various ways, the following assumes the default setup.
    As reflected in the high SWE-bench scores, none of the following limitations are a problem in practice.

    - No tools other than bash
    - Actions are parsed from triple-backtick blocks (rather than assuming a function calling/tool calling format)
    - By default, actions are executed as `subprocess.run`, i.e., every action is independent of the previous ones
      (meaning that the agent cannot change directories or export environment variables; however environment variables
      can be set per-action)

    If you want more flexibility with these items, you can use [SWE-agent](https://swe-agent.com/) instead.

??? question "Where is global configuration stored?"

    The global configuration is stored in the `.env` file in the config directory.
    The location is printed when you run `micro --help`.

    The `.env` file is a simple key-value file that is read by the `dotenv` library.


## Models

!!! question "What models do you support?"

    Currently, micro-SWE-agent supports all models that are supported by [litellm](https://github.com/BerriAI/litellm)
    and we're open to extend the `models/` directory with more models should `litellm` not support them.

!!! question "How do I set the API key for a model?"

    The API key can be stored either as an environment variable (note that enviroinment variables are not persistent
    unless you set them in your `~/.bashrc` or similar), or as a permanent key in the config file.

    To temporarily set the API key as an environment variable, you can use the following command:

    ```bash
    export OPENAI_API_KEY=sk-test123
    ```

    To permanently set the API key in the config file, you can use the following command:

    ```bash
    micro set-key OPENAI_API_KEY sk-test123
    ```

    Alternatively, you can directly edit the `.env` file in the config directory
    (the location is printed when you run `micro --help`).

!!! question "How can I set the default model?"

    The default model is stored in the config/environment as `MSWEA_MODEL_NAME`.
    To permanently change it:

    ```bash
    micro set-key MSWEA_MODEL_NAME claude-sonnet-4-20250514
    ```

    Alternatively, you can directly edit the `.env` file in the config directory
    (the location is printed when you run `micro --help`).

{% include-markdown "_footer.md" %}