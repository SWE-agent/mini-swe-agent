# LLMTR Model

LLMTR is an OpenAI-compatible gateway at [`https://llmtr.com/v1`](https://llmtr.com/).
Model IDs are namespaced `vendor/model` values such as `minimax/minimax-m3.1-flash`.
IDs under `llmtr/` are hosted in Turkey; other vendor/model IDs route to their vendors.

Set `LLMTR_API_KEY` (see https://llmtr.com/dashboard/keys). Select this class with `--model-class llmtr`.
If the API does not return a `usage.cost` field, set `cost_tracking: ignore_errors` in the model config
or export `MSWEA_COST_TRACKING=ignore_errors`.

```bash
export LLMTR_API_KEY="your-api-key"
mini --model-class llmtr --model minimax/minimax-m3.1-flash
```

!!! note "LLMTR Model class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/models/llmtr_model.py)

    ??? note "Full source code"

        ```python
        --8<-- "src/minisweagent/models/llmtr_model.py"
        ```

::: minisweagent.models.llmtr_model

{% include-markdown "../../_footer.md" %}
