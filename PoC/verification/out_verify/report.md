# 래퍼 검증 리포트

기준점: `PoC\verification\toy_agent\observatory\diagram\verify-anchors.json`

흐름 8개: **검증 완료 8** · **일부만 기록 0** · **누락 0** · 사각지대 0곳

## 사각지대 (래퍼 밖에서 값이 외부로 나가는 호출)

없음. 세 값이 외부로 나가는 호출은 모두 기준점의 래퍼 안에 있습니다.

## User Input

### ✅ 검증 완료: [User Input] task (entry parameter) → [LLM 입력] self.client.messages.create

- 생김: `agent.py:42` `def run(self, task: str, max_steps: int = 10) -> str:`
- 기록: `llm.py:14` `observatory_log.llm_query(step, model=self.model, messages=messages)`
- 기록: `agent.py:43` `start = observatory_log.tool_start("user_input", {"task": task})`
- 기록: `agent.py:44` `observatory_log.tool_end(start, "user_input", result=task)`
- 쓰임 (LLM 입력): `llm.py:16` `response = self.client.messages.create(`

### ✅ 검증 완료: [User Input] input → [최종 출력] return of ToyAgent.run

- 생김: `agent.py:68` `answer = input(f"{question} ")`
- 기록: `agent.py:69` `observatory_log.tool_end(start, "user_input", result=answer)`
- 쓰임 (최종 출력): `agent.py:49` `return reply["content"] if self.confirm(reply["content"]) else self.ask_user("What next?")`


## LLM Response

### ✅ 검증 완료: [LLM Response] self.client.messages.create → [LLM 입력] self.client.messages.create

- 생김: `llm.py:16` `response = self.client.messages.create(`
- 기록: `llm.py:14` `observatory_log.llm_query(step, model=self.model, messages=messages)`
- 기록: `llm.py:27` `observatory_log.llm_response(`
- 쓰임 (LLM 입력): `llm.py:16` `response = self.client.messages.create(`

### ✅ 검증 완료: [LLM Response] self.client.messages.create → [사용자에게 표시] input

- 생김: `llm.py:16` `response = self.client.messages.create(`
- 기록: `llm.py:27` `observatory_log.llm_response(`
- 기록: `agent.py:67` `start = observatory_log.tool_start("user_input", {"question": question})`
- 쓰임 (사용자에게 표시): `agent.py:68` `answer = input(f"{question} ")`

### ✅ 검증 완료: [LLM Response] self.client.messages.create → [외부 호출] exec

- 생김: `llm.py:16` `response = self.client.messages.create(`
- 기록: `tools.py:23` `start = observatory_log.tool_start(name, args)`
- 기록: `llm.py:27` `observatory_log.llm_response(`
- 쓰임 (외부 호출): `tools.py:14` `exec(code, scope)`

### ✅ 검증 완료: [LLM Response] self.client.messages.create → [외부 호출] subprocess.run

- 생김: `llm.py:16` `response = self.client.messages.create(`
- 기록: `tools.py:23` `start = observatory_log.tool_start(name, args)`
- 기록: `llm.py:27` `observatory_log.llm_response(`
- 쓰임 (외부 호출): `tools.py:9` `return subprocess.run(command, shell=True, capture_output=True, text=True).stdout`

### ✅ 검증 완료: [LLM Response] self.client.messages.create → [최종 출력] return of ToyAgent.run

- 생김: `llm.py:16` `response = self.client.messages.create(`
- 기록: `llm.py:27` `observatory_log.llm_response(`
- 쓰임 (최종 출력): `agent.py:49` `return reply["content"] if self.confirm(reply["content"]) else self.ask_user("What next?")`


## Tool Response

### ✅ 검증 완료: [Tool Response] self.tools[name] → [LLM 입력] self.client.messages.create

- 생김: `tools.py:25` `out = self.tools[name](**args)`
- 기록: `llm.py:14` `observatory_log.llm_query(step, model=self.model, messages=messages)`
- 기록: `tools.py:29` `observatory_log.tool_end(start, name, result=out)`
- 쓰임 (LLM 입력): `llm.py:16` `response = self.client.messages.create(`
