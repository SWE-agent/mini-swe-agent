# Study planner input

Run the interactive terminal input flow from the repository root:

```bash
python -m study_planner.ui.app
```

Enter each task's name, optional course/category, deadline, estimated workload in
minutes, and priority (`1` low, `2` medium, `3` high), followed by one or more
availability windows. Dates and times use `YYYY-MM-DD HH:MM`.

The validated JSON is saved to `study_planner/data/planner_input.json` by
default. Choose another destination with `--output`:

```bash
python -m study_planner.ui.app --output /tmp/planner.json
```

The saved `tasks` and `availability` lists are accepted by
`study_planner.input.load_planner_input` and can be passed to the agent pipeline.
