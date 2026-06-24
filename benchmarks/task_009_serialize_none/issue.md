# Fix recursive_merge None handling

`upstream/src/minisweagent/utils/serialize.py` implements `recursive_merge` for config merging.

Run `python -m pytest upstream/tests/utils/test_serialize.py -v` — at least one test should fail.

Your job:

1. Read the failing test(s) and `recursive_merge`
2. Fix the bug so `None` values merge correctly (distinct from `UNSET` skipping)
3. Run pytest again to verify
4. Run `git diff` and submit when done
