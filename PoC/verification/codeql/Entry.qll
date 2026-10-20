/** The agent entry, as `Class.method` (run_codeql.py rewrites this file in its temporary copy of the pack). */
predicate entry(string cls, string method) { cls = "ToyAgent" and method = "run" }
