# Fix recursive_merge UNSET handling

`upstream/src/minisweagent/utils/serialize.py` implements `recursive_merge()` for merging nested config dictionaries. The sentinel `UNSET` marks keys that should be omitted from the merged result.

Run `python -m pytest upstream/tests/utils/test_serialize.py -v` — tests around nested `UNSET` values should fail.

Your job:

1. Read the failing test(s) and the `recursive_merge` implementation
2. Fix the bug so nested dictionaries with `UNSET` values are merged correctly
3. Run pytest again to verify
4. Run `git diff` and submit when done

Notes:

- `UNSET` is a module-level sentinel object, not `None`
- Later dictionaries take precedence over earlier ones
- Nested dicts should be merged recursively
