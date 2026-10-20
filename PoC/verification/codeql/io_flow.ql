/**
 * @name I/O value flows and their logging
 * @description Cross-check for flow_audit.py: where do User Input, LLM Response and Tool Response values flow
 *              (LLM input, tool input, final output), and does a log call receive the value on the way.
 * @kind table
 * @id verification/io-flow
 */

import python
import semmle.python.dataflow.new.DataFlow
import IoFlow

from DataFlow::Node src, string cat, string use, string useLoc, string logged
where
  source(src, cat) and
  (
    exists(DataFlow::Node snk | useSink(snk, use) and Io::flow(src, snk) and useLoc = loc(snk)) and
    if exists(DataFlow::Node l | logArg(l, _) and Io::flow(src, l))
    then logged = "logged: " + concat(DataFlow::Node l | logArg(l, _) and Io::flow(src, l) | loc(l), ", ")
    else logged = "NOT logged"
    or
    not exists(DataFlow::Node snk | useSink(snk, _) and Io::flow(src, snk)) and
    use = "(no use)" and
    useLoc = "" and
    logged = ""
  )
select cat, src.toString(), loc(src), use, useLoc, logged
