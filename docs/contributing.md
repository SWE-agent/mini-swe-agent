# ❤️ Contributing

We happily accept contributions!

## Areas of help

- Feedback on the `mini` and `mini -v` interfaces at [this github issue](https://github.com/swe-agent/mini-swe-agent/issues/161) or in our [Slack channel](https://join.slack.com/t/swe-bench/shared_invite/zt-36pj9bu5s-o3_yXPZbaH2wVnxnss1EkQ).
- Documentation, examples, tutorials, etc. In particular, we're looking for
    - examples of how this library is used in the wild
    - additional examples for the [cookbook](advanced/cookbook.md)
- Support for more models (anything where `litellm` doesn't work out of the box)
- Support for more environments & deployments (e.g., run it as a github action, etc.)
- Take a look at the [issues](https://github.com/SWE-agent/mini-swe-agent/issues) and look for issues marked `good-first-issue` or `help-wanted` (please read the guidelines below first)

## Design & Architecture

- `mini-swe-agent` aims to stay minimalistic, hackable, and of high quality code.
- To extend features, we prefer to add a new version of the one of the four components (see [cookbook](advanced/cookbook.md)), rather than making the existing components more complex.
- Components should be relatively self-contained, but if there are utilities that might be shared, add a `utils` folder (like [this one](https://github.com/SWE-agent/mini-swe-agent/tree/main/src/minisweagent/models/utils)). But keep it simple!
- If your component is a bit more specific, add it into an `extra` folder (like [this one](https://github.com/SWE-agent/mini-swe-agent/tree/main/src/minisweagent/run/extra))
- Our target audience is anyone who doesn't shy away from modifying a bit of code (especially a run script) to get what they want.
- Therefore, not everything needs to be configurable with the config files, but it should be easy to create a run script that makes use of it.
- Many LMs write very verbose code -- please clean it up! Same goes for the tests. They should still be concise and readable.
- Please install `pre-commit` (`pip install pre-commit && pre-commit install`) and run it before committing. This will enforce our style guide.

## Development setup

Make sure to follow the dev setup instructions in [quickstart.md](quickstart.md).

After that you can run `pytest` with `pytest -n auto` (this parallelizes the tests across all cores for speedup).

{% include-markdown "_footer.md" %}
