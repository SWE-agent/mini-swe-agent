/** Shared definitions for io_flow.ql (table) and io_paths.ql (paths): sources, uses, log calls, taint config. */

import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.dataflow.new.TaintTracking
import semmle.python.ApiGraphs
import Entry

/** The agent entry (flow_audit.py --entry), set in Entry.qll. */
predicate isEntry(Function f) {
  exists(string cls, string method | entry(cls, method) |
    f.getName() = method and f.getScope().(Class).getName() = cls
  )
}

/** A call whose result is an I/O value, with its category (same rules as flow_audit.py / io_rules.py). */
predicate ioCall(DataFlow::CallCfgNode c, string cat) {
  // R1: <client>.messages.create(...) (anthropic), <client>.chat.completions.create(...) (openai)
  exists(Attribute f |
    f = c.getFunction().asExpr() and
    f.getName() = "create" and
    f.getObject().(Attribute).getName() = ["messages", "completions"]
  ) and
  cat = "LLM"
  or
  // T1 / T2: process and code execution
  c = API::moduleImport("subprocess").getMember(["run", "Popen", "call", "check_output"]).getACall() and
  cat = "TOOL"
  or
  c = API::builtin(["exec", "eval"]).getACall() and cat = "TOOL"
  or
  // T4: dynamic dispatch, registry[name](...)
  c.getFunction().asExpr() instanceof Subscript and cat = "TOOL"
  or
  // I1: user input
  c = API::builtin("input").getACall() and cat = "INPUT"
}

/** Any expression passed to `call`: positional, keyword, *args, **kwargs. */
predicate argOf(DataFlow::Node n, Call call) {
  n.asExpr() = [call.getAPositionalArg(), call.getANamedArg().(Keyword).getValue(), call.getStarargs(), call.getKwargs()]
}

predicate source(DataFlow::Node n, string cat) {
  ioCall(n, cat)
  or
  exists(Function f | isEntry(f) | n.(DataFlow::ParameterNode).getParameter() = f.getAnArg() and
    not n.(DataFlow::ParameterNode).getParameter().getName() = "self") and
  cat = "INPUT"
}

/** Where an I/O value is used: argument of an LLM call / tool execution, or the entry's return value. */
predicate useSink(DataFlow::Node n, string kind) {
  exists(DataFlow::CallCfgNode c, string cat | ioCall(c, cat) and cat != "INPUT" |
    argOf(n, c.asExpr()) and kind = cat + " input"
  )
  or
  exists(Function f, Return r | isEntry(f) and r.getScope() = f and n.asExpr() = r.getValue()) and
  kind = "final output"
}

/** An argument of a logging call: agent_observatory_kit functions or "[LLM_" / "[TOOL_" tagged log lines. */
predicate logArg(DataFlow::Node n, Call call) {
  argOf(n, call) and
  (
    call.getFunc().(Attribute).getName() =
      ["llm_query", "llm_response", "llm_error", "tool_start", "tool_end", "tool_error"]
    or
    call.getAPositionalArg().(StringLiteral).getText().regexpMatch("\\[(LLM|TOOL|INPUT)_.*")
  )
}

module IoConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node n) { source(n, _) }

  predicate isSink(DataFlow::Node n) { useSink(n, _) or logArg(n, _) }

  /**
   * Reading an attribute of a tainted object yields tainted data (`response.content`, `run(...).stdout`).
   * CodeQL does not do this for unmodelled library objects; flow_audit.py does, so the two are compared alike.
   */
  predicate isAdditionalFlowStep(DataFlow::Node n1, DataFlow::Node n2) {
    exists(DataFlow::AttrRead r | r.getObject() = n1 and r = n2)
  }
}

module Io = TaintTracking::Global<IoConfig>;

string loc(DataFlow::Node n) {
  result = n.getLocation().getFile().getShortName() + ":" + n.getLocation().getStartLine()
}
