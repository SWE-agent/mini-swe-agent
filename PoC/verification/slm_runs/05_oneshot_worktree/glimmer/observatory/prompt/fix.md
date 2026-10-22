## This step: fix what the checker found

The checker (`{kit}/flow_verify.py`) followed the user input, the LLM reply and the tool output from the anchors in `{anchors}` and reported what is below: anchors that do not match the code, or values that reach a use without a logger call recording them whole (일부만 기록: only a part of the value is logged; 누락: it is not logged at all).

{problems}

Fix the logger calls or the anchors, whichever is wrong. Then run the checker again until every line reads 검증 완료:

    uv run {kit}/flow_verify.py . {anchors} {kit}/verify --only {files}
