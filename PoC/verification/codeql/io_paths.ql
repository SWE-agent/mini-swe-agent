/**
 * @name I/O value flow paths to uses and log calls
 * @description Every path from a User Input / LLM Response / Tool Response value to a use (LLM input, tool input,
 *              final output) or to a log call. run_codeql.py draws them; the message is "<category>|<end kind>".
 * @kind path-problem
 * @problem.severity recommendation
 * @id verification/io-paths
 */

import python
import semmle.python.dataflow.new.DataFlow
import IoFlow
import Io::PathGraph

from Io::PathNode src, Io::PathNode end, string cat, string kind
where
  Io::flowPath(src, end) and
  source(src.getNode(), cat) and
  (useSink(end.getNode(), kind) or logArg(end.getNode(), _) and kind = "log")
select end.getNode(), src, end, cat + "|" + kind
