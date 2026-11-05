# Python bindings

!!! abstract "Overview"

    This page shows the most basic example of how to use mini-SWE-agent as a Python library.
    For more advanced usage, subclassing, and mix & match of components, see [subclassing and more](../advanced/cookbook.md).

## Hello world

Load the bundled `mini` configuration to provide the required agent prompt templates and the matching model configuration.
Set up your model's API key as described in [model quickstart](../models/quickstart.md).

```python
import logging

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.environments.local import LocalEnvironment

logging.basicConfig(level=logging.DEBUG)
task = "Write a hello world program"
model_name = "anthropic/claude-sonnet-4-5-20250929"
config = get_config_from_spec("mini")

agent = DefaultAgent(
    get_model(input_model_name=model_name, config=config["model"]),
    LocalEnvironment(),
    **config["agent"],
)

# Run the agent
agent.run(task)
```

{% include-markdown "../_footer.md" %}
