# SWE-bench


## FAQ

> How can I run on a different dataset?

As long as it follows the SWE-bench format, you can use `--subset /path/to/your/dataset` to run on a custom dataset.
The dataset needs to be loadable as `datasets.load_dataset(path, split=split)`.

> Some progress runners are stuck at 'initializing task' for a very long time

They might be pulling docker containers -- the run sshould start immediately the next time.