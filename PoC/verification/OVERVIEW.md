# Agent Observatory 키트와 로깅 누락 정적 검증 (flow_verify)

## 1. 목적

LLM 에이전트(mini-swe-agent, browser-use, smolagents 등)에 로그를 자동으로 심고, 심어진 로그가 에이전트의 핵심 데이터 세 가지를 **빠짐없이, 통째로** 기록하는지 실행 없이 검증하는 것이 목표입니다.

| 값 | 의미 |
|---|---|
| **User Input** | 사용자가 준 task, 실행 중 사용자가 입력한 답 |
| **LLM Response** | 모델 호출이 돌려준 응답 |
| **Tool Response** | tool·action 실행이 돌려준 결과 |

작업은 두 부분으로 나뉩니다.

1. **Agent Observatory 키트** (`observatory/`): 에이전트 저장소에 넣고 명령 하나를 실행하면, Claude Code 세션이 그 에이전트에 로거를 심고 검증용 기준점(`verify-anchors.json`)과 구조도를 만듭니다. 로그는 대시보드에서 실시간으로 볼 수 있습니다.
2. **정적 검증기** (`PoC/verification/flow_verify.py`): 기준점에서 출발해 세 값이 코드 안에서 흘러가는 경로를 따라가며, 값이 쓰이기 전에 로그에 통째로 기록되는지 판정합니다. LLM을 쓰지 않는 결정적 분석입니다.

```
에이전트 저장소
  │  observatory/ 폴더를 넣고: uv run observatory/run.py
  ▼
[키트] Claude Code 세션 (비대화형)
  ├─ 로거 복사, LLM 호출·tool 실행·사용자 입력에 로그 호출 삽입
  ├─ 결정적 모델로 실행해 로그 확인            → observatory/runs/*.jsonl  → 대시보드
  ├─ 기준점 작성                               → observatory/diagram/verify-anchors.json
  └─ 구조도 spec 작성 → Archify가 HTML 생성      → observatory/diagram/agent-architecture.html
  ▼
[검증기] flow_verify.py  (소스 코드 + verify-anchors.json)
  └─ 흐름별 판정 (검증 완료 / 일부만 기록 / 누락)  → report.html, report.json
```

검증기는 하나의 파일이고, 에이전트마다 다른 것은 입력(소스 루트, 기준점 파일, 분석 범위)뿐입니다. 코드 안에 특정 에이전트용 분기는 없습니다.

---

## 2. 키트 (`observatory/`)

### 2.1 구성

| 파일 | 역할 | 에이전트마다 |
|---|---|---|
| `observatory_log.py` | 로거. 표준 라이브러리만 사용. 에이전트 패키지에 그대로 복사해서 호출만 심음 | 같음 |
| `dashboard.py`, `static/index.html` | 대시보드 서버와 화면. 에이전트를 모르고 로그 파일만 읽음 | 같음 |
| `run.py` | 전체 실행 스크립트 (아래 2.3) | 같음 |
| `prompt/task.md` | Claude Code 세션에 주는 요청문 (아래 2.4) | 같음 |
| `selftest.py` | 로거·대시보드 자체 테스트 | 같음 |
| `diagram/example-architecture.json` | 구조도 작성 시 참고할 다른 에이전트의 예시 | 같음 |
| `diagram/verify-anchors.json` | 검증 기준점 | **세션이 생성** |
| `diagram/agent-architecture.json` → `.html` | 대시보드 왼쪽 구조도 | **세션이 생성** |
| `runs/*.jsonl` | 실행 로그 | **실행 시 생성** |

### 2.2 로거 (`observatory_log.py`)

`OBSERVATORY_RUNS` 환경변수가 디렉터리를 가리킬 때만 그 안의 파일에 JSON 한 줄씩 기록합니다. 변수가 없으면 아무것도 하지 않아, 평소 실행에는 영향이 없습니다.

| 함수 | 호출 시점 |
|---|---|
| `llm_query(step, model, messages, **fields)` | 모델 호출 직전 |
| `llm_response(step, actions, **fields)` | 응답을 액션으로 파싱한 직후 |
| `llm_error(step, error, **fields)` | 모델 호출이 예외를 던졌을 때 |
| `start = tool_start(action, params, step)` | tool·action 실행 직전 |
| `tool_end(start, action, result, step, **fields)` | 실행이 끝났을 때. `result`는 출력 전체 |
| `tool_error(start, action, error, step, **fields)` | 실행이 예외를 던졌을 때 |

- 사용자 입력은 별도 이벤트 없이 action 이름 `"user_input"`인 `tool_start`/`tool_end`로 기록합니다. 대시보드는 이를 USER로 구분해 보여 줍니다.
- 이름에 `KEY`, `TOKEN`, `SECRET`, `PASSWORD`가 들어간 환경변수 값은 기록에서 지웁니다.
- SDK 응답 같은 pydantic 객체는 `model_dump()`로 구조를 유지해 저장합니다. 문자열 하나는 20,000자에서 자릅니다.

### 2.3 실행 흐름 (`run.py`)

```
uv run observatory/run.py [--archify-dir DIR] [--budget USD] [--model sonnet]
  1. selftest.py로 키트 자체 점검
  2. Claude Code 세션 1개 실행 (prompt/task.md를 stdin으로)
       claude -p --no-session-persistence --setting-sources project --permission-mode acceptEdits
       허용 도구: Read, Edit, Write, Grep, Glob, Bash(uv run:*), Bash(git diff:*), Archify validate
       OBSERVATORY_RUNS=observatory/runs 로 설정 → 세션 안의 실행 로그가 여기 쌓임
  3. 결과 점검
       - 저장소 안 로거 사본이 원본과 다르면 원본으로 복원
       - 로거를 호출하는 파일, runs/의 이벤트 수 출력
       - 구조도 spec의 파일·줄 번호가 실제 코드와 맞는지
  4. Archify (있을 때만)
       spec의 revision을 현재 작업 트리 스냅샷 커밋으로 고정 → validate
       → 실패하면 오류를 Claude 세션에 넘겨 수정 (최대 2회) → deliver로 HTML 생성
```

| 옵션 | 동작 |
|---|---|
| `--no-claude` | 세션 없이 점검·구조도만 |
| `--archify-only` | 커밋 후 revision 재고정과 HTML 재생성만 |
| `--serve`, `--serve-only` | 대시보드 실행 |
| `--budget` | 세션당 최대 비용(USD, 기본 3) |

### 2.4 요청문 (`prompt/task.md`) 전문

Claude Code 세션에 그대로 들어가는 프롬프트입니다. `{kit}`은 `observatory`, `{archify_dir}`은 Archify 경로로 치환되고, Archify가 없으면 `<!-- diagram -->` 아래(Part 2)는 빠집니다.

````markdown
This repo is an LLM agent. `{kit}/` contains the Agent Observatory: a dashboard (`{kit}/dashboard.py`) that shows a per-step history of an agent's LLM calls and tool executions, fed by a small logger module (`{kit}/observatory_log.py`). The dashboard and the logger are fixed and agent-agnostic: do not modify them. Your job is to connect this agent to the logger.

This session is non-interactive: nobody can answer questions, so never ask; carry out every part below to the end. If the agent already calls the logger, check that wiring against the steps below, fix only what is missing or wrong, and still write every file the steps ask for.

## Part 1: wire the logger into the agent

1. Copy `{kit}/observatory_log.py` unchanged into the agent's main Python package (the package that contains the agent loop) and import it from there. Read its docstring first: it lists the six functions and when to call each.
2. Find where the agent calls the model and where it executes a tool or action, and add the calls:
   - `llm_query` right before every model call, `llm_response` right after the reply was parsed into actions, `llm_error` when the call raises (then the agent's existing error handling continues unchanged).
   - `tool_start` right before one tool or action runs, `tool_end` after it returned, `tool_error` when it raised (then re-raise unchanged).
   - Report the user's task too: right where the agent receives it (its run function), and every place it reads user input, as the tool action `"user_input"` (`tool_start` with the prompt, `tool_end` with what the user typed).
   - Log whole values, never parts of them: a checker verifies that the user input, the model reply and the tool output each reach a logger call as the variable itself. Pass the variable (`messages=messages`, `response=response`, `result=output`), not a count (`len(messages)`), a slice (`output[:500]`), one field (`response.stop_reason`) or a value built from it (`[a["command"] for a in actions]`). Such summaries may be added as extra fields next to the whole value. For example:
     ```python
     observatory_log.llm_query(step, model=self.model, messages=messages)
     response = self.client.messages.create(model=self.model, messages=messages)
     reply = self.parse(response)
     observatory_log.llm_response(step, actions=[c["name"] for c in reply["calls"]], reply=reply, response=response)
     ...
     start = observatory_log.tool_start(name, args)
     out = self.tools[name](**args)
     observatory_log.tool_end(start, name, result=out)
     ```
   - Other useful keyword fields: the model name, a purpose, cost, token counts.
3. Rules:
   - Every code path that calls the model or executes an action must be covered, including subclasses and alternative agent classes that bypass the base-class path. If several paths run actions, wire the tool calls at the one place they all go through.
   - If the agent signals a normal outcome with an exception (for example finishing or submitting a task), report it with `tool_end` and a short result such as "submitted", not with `tool_error`.
   - No behavior change besides the logging. The agent's normal user-facing output must not change. Follow AGENTS.md if the repo has one (minimal, concise code, no extra comments).
   - Never put secrets the agent is configured to hide (API keys, sensitive data) into the fields.
4. Check your work: run the agent, or its smallest entry point, with a scripted or deterministic model so that no API key or network is needed. Start it with plain `uv run python <script>`: the environment variable `OBSERVATORY_RUNS` is already set to `{kit}/runs` for you, so do not put it in front of the command. Then read the JSON-lines file created in `{kit}/runs/` and confirm that every step shows its llm_query, llm_response and tool events in order. Leave that file there.
5. Write `{kit}/diagram/verify-anchors.json`, the anchors a static checker starts from to verify that your logging records the user input, the LLM response and the tool output before they leave the agent. All paths are relative to the repo root and all lines are the code as it is after your changes:
   ```json
   {
     "logger": "observatory_log",
     "entry": {"function": "Class.method", "path": "pkg/agent.py", "line": 1, "inputs": ["task"]},
     "sources": [
       {"kind": "llm_response", "path": "pkg/agent.py", "line": 1, "call": "self.model.query", "label": "model call"},
       {"kind": "tool_response", "path": "pkg/agent.py", "line": 1, "call": "self.env.execute", "label": "tool call"},
       {"kind": "user_input", "path": "pkg/agent.py", "line": 1, "call": "input", "label": "prompt"}
     ]
   }
   ```
   - `entry`: the agent's run function and its parameters that carry the user's task.
   - `sources`: every call whose return value is a model reply, a tool or action output, or text the user typed. `call` is the called expression exactly as written in the source (`ast.unparse` of the call's function) on that `line`.

<!-- diagram -->
## Part 2: the architecture diagram (left panel of the dashboard)

Write `{kit}/diagram/agent-architecture.json`, the Archify architecture spec of this agent's loop, which is shown next to the history. A separate tool generates the HTML from it, so never create or edit HTML.

- Read Archify's instructions at `{archify_dir}/SKILL.md` and the references it points to. Follow `{kit}/diagram/example-architecture.json`, the diagram of another agent, for the style and structure.
- Show the real flow of this agent: the task coming in, the agent loop, the LLM call, tool execution and the result going out, with this repo's real classes and functions as component labels.
- Every component that corresponds to code carries `sources`: the file `path` and `line` (and `end_line`) of that code in this repo as it is now, including your logger calls. Give the components that carry your logger calls a `tag` naming the events, for example "llm_query llm_response llm_error".
- Keep the `meta.repository` block; the kit sets its url and revision itself.
- Validate with `node {archify_dir}/bin/archify.mjs validate architecture {kit}/diagram/agent-architecture.json --quality showcase` until it passes. Add `--repo-root .` if the tool asks for it. Errors about the repository evidence (the revision of your uncommitted changes) can be ignored: the kit pins the revision itself afterwards and validates again.

Do not commit.
````

**요청문에서 검증과 직접 연결되는 부분**

| 항목 | 내용 | 이유 |
|---|---|---|
| 비대화형 명시 | 질문하지 말고 끝까지 진행. 이미 연결돼 있으면 점검·보완하고 파일은 반드시 작성 | 이 문장이 없을 때 세션이 "무엇을 할지" 질문만 남기고 종료한 사례가 두 번 있었음 |
| 사용자 task 기록 | run 함수에서 받은 task와 모든 사용자 입력을 `user_input`으로 기록 | mini는 처음 받은 task를 기록하지 않아 누락으로 판정됐음 |
| 전체 값 로깅 | 변수 자체를 넘기고 개수·슬라이스·필드·가공값은 금지. 요약은 옆 필드로만 | `len(messages)`, `output[:500]`, 명령어 목록만 기록하는 패턴이 "일부만 기록"의 원인이었음 |
| 기준점 작성 (Part 1의 5번) | entry와 sources를 파일·줄·호출식으로 | 로그를 심은 세션이 출발점을 가장 잘 앎. Archify가 없어도 생성되도록 Part 1에 둠 |

### 2.5 대시보드

`uv run observatory/dashboard.py --runs <runs 폴더> [--port N] [--agent "<실행 명령>"]`

- 왼쪽: Archify 구조도. runs 폴더 옆의 `diagram/agent-architecture.html`을 읽습니다.
- 오른쪽 Call history: step별 LLM, TOOL, USER 행. `messages`, `response` 같은 큰 필드는 접힌 상태로 표시되고, 누르면 전체 JSON이 펼쳐집니다.
- Raw events: 이벤트 원문.
- `--agent`를 주면 헤더에 task 입력창이 생겨, 입력한 task로 에이전트를 직접 실행할 수 있습니다.

---

## 3. 키트가 검증기에 넘기는 것: 기준점 (`verify-anchors.json`)

검증기는 이 파일에서 출발점을 얻습니다. 기준점에 없는 호출은 출발점이 되지 않습니다.

| 키 | 내용 |
|---|---|
| `logger` | 로그 모듈 이름 (`observatory_log`). 이 모듈의 함수 호출을 로그로 봄 |
| `entry` | 진입 함수, 파일, 줄, 그 인자 중 사용자 task를 받는 것(`inputs`). 인자는 User Input 출발점, 이 함수의 return은 최종 출력 |
| `sources` | 세 값이 생기는 호출. `kind`(`user_input` / `llm_response` / `tool_response`), `path`, `line`, `call`(소스에 적힌 호출식 그대로) |

`call`이 해당 줄에서 찾아지지 않으면 검증기는 즉시 멈춥니다. 기준점이 코드와 어긋났다는 뜻입니다.

mini-swe-agent에서 세션이 생성한 예:
```json
{
  "logger": "observatory_log",
  "entry": {"function": "DefaultAgent.run", "path": "src/minisweagent/agents/default.py", "line": 88, "inputs": ["task"]},
  "sources": [
    {"kind": "llm_response", "path": "src/minisweagent/agents/default.py", "line": 161, "call": "self.model.query"},
    {"kind": "tool_response", "path": "src/minisweagent/agents/default.py", "line": 193, "call": "self.env.execute"},
    {"kind": "user_input", "path": "src/minisweagent/agents/interactive.py", "line": 195, "call": "prompt_session.prompt"},
    {"kind": "user_input", "path": "src/minisweagent/agents/interactive.py", "line": 94, "call": "input"}
  ]
}
```

---

## 4. 검증기 (`flow_verify.py`)

### 4.1 실행

```bash
uv run PoC/verification/flow_verify.py <소스 루트> <verify-anchors.json> <출력 폴더> --only <분석할 파일들>
```

출력: `report.html`(흐름별 카드와 경로, 칸을 누르면 VS Code에서 해당 줄이 열림), `report.json`.
표준 라이브러리만 사용하며, 에이전트 코드를 실행하지 않습니다.

### 4.2 한 줄 요약

> 코드를 AST로 읽어 "값이 A에서 B로 간다"는 흐름 그래프를 만들고, 기준점의 각 출발점에서 값이 쓰이는 곳(도착점)까지 **로그에 통째로 기록되는 칸을 거치지 않고 갈 수 있는 길이 있는지** 찾는다.

### 4.3 그래프 만들기

**노드: 값이 머무는 자리**

| 노드 | 이름 형식 | 예 |
|---|---|---|
| 지역 변수, 파라미터 | `함수:변수` | `LLM.chat:response` |
| 함수 반환값 | `함수:<return>` | `LLM.parse:<return>` |
| 객체 필드 | `클래스.필드` | `ToyAgent.messages` |
| 출발점, 도착점, 로그 | `[종류] 호출식 @ 파일:줄` | `[LOG] observatory_log.tool_end @ tools.py:29` |

**간선: AST 종류별 규칙**

| AST | 코드 | 간선 |
|---|---|---|
| 대입 | `x = y` | `y → x` |
| return | `return x` | `x → 함수:<return>` |
| 필드 대입 | `self.messages = m` | `m → 클래스.messages` |
| 프로젝트 함수 호출 | `self.parse(response)` | 인자 → 파라미터, 호출식의 값 = 그 함수의 `<return>` |
| 변경 메서드 | `lst.append(x)` | `x → lst` |
| 반복, comprehension | `for c in calls`, `[f(c) for c in calls]` | `calls → c` |
| 필드·인덱스 읽기 | `reply["content"]`, `r.stop_reason` | 값 = `reply`, `r` (가공으로 봄) |
| 데이터 없는 builtin | `len(x)`, `isinstance(x, ...)` | 간선 없음 (흐름이 끊김) |
| 외부 호출 | `subprocess.run(cmd)`, `client.messages.create(...)` | 인자 → **도착점** |
| 로그 호출 | `observatory_log.tool_end(result=out)` | 인자 → **로그 노드** |
| 기준점 호출 | 기준점과 줄·호출식이 일치하는 호출 | 반환값 = **출발점** 노드 |

**호출 대상 판단**은 이름이 아니라 출처로 합니다.
- import를 파일 경로로 해석합니다. 상대 import(`from . import x`)와 패키지 경로(`from pkg.agent.views import X`)를 분석 대상 파일과 맞춰 프로젝트 코드인지 외부 라이브러리인지 정합니다.
- `self.client = anthropic.Anthropic()` 같은 대입으로 필드 타입을 먼저 수집해, `self.client.messages.create`가 외부 호출임을 압니다.
- 같은 클래스나 부모 클래스의 메서드는 프로젝트 함수입니다.
- 타입을 알 수 없는 객체(생성자 인자로 받은 `self.model` 등)의 메서드 호출은 가공으로 봅니다. 단, 기준점 호출이면 도착점으로 봅니다.

**도착점(값이 쓰이는 곳)**
- 외부 호출: import한 라이브러리 함수나 그것으로 만든 객체의 메서드, 부작용 있는 builtin(`exec`, `input`, `print`, `open` 등)
- 최종 출력: entry 함수의 반환값

### 4.4 로그가 값을 "통째로" 받는지

로그 호출의 각 인자를 봅니다.

| 넘긴 것 | 판정 |
|---|---|
| 이름(`out`), `self.필드`, 그것들의 list/tuple/dict | 통째로 기록 |
| `out[:200]`, `response.stop_reason`, `reply["content"]`, `[c["name"] for c in ...]` | 일부만 기록 |
| `len(messages)` | 기록 아님 |

로그를 헬퍼 함수 안에서 남기는 경우(`self._log_response(response=msg)` → 헬퍼 안에서 `observatory_log.llm_response(response=response)`)에는, 값을 그대로 넘긴 호출한 쪽 변수도 통째로 기록된 것으로 봅니다.

### 4.5 판정

출발점 하나마다 화살표를 따라 닿는 도착점을 모두 찾고, (출발점, 도착점) 쌍마다 판정합니다.

1. 통째로 기록된 노드를 **막힌 칸**으로 둡니다.
2. 막힌 칸을 피해서 출발점에서 도착점까지 갈 수 있는지 찾습니다.

| 판정 | 조건 |
|---|---|
| **검증 완료** | 피해 가는 길이 없음. 모든 경로가 통째 기록을 지남 |
| **일부만 기록** | 피해 가는 길이 있고, 그 길에서는 값의 일부만 로그에 들어감 |
| **누락** | 피해 가는 길이 있고, 그 길에서는 어떤 로그도 값을 받지 않음 |

리포트는 판정이 검증 완료가 아니면 **기록 없이 지나는 경로**를 보여 주고, 각 칸에 붙은 로그가 통째(✓)인지 일부(△)인지 표시합니다.

### 4.6 예시 (toy agent, Tool Response)

```
[Tool Response] self.tools[name](**args)      tools.py:25   출발점
  → Registry.run:out
       └→ [LOG] tool_end(result=out)          tools.py:29   out을 통째로 받음 → out은 막힌 칸
  → Registry.run:<return> → ToyAgent.step:results → LLM.results_message → ToyAgent.messages
  → LLM.chat:messages → [LLM 입력] client.messages.create   도착점
```
모든 경로가 `out`을 지나므로 검증 완료입니다. `result=out[:200]`이면 일부만 기록, 해당 줄이 없으면 누락이 됩니다.

---

## 5. 적용 결과

같은 키트(같은 요청문)와 같은 검증기로 네 에이전트에 적용했습니다.

| 에이전트 | 로그 삽입 | 분석 범위 | 흐름 | 검증 완료 | 일부만 기록 | 누락 |
|---|---|---|---|---|---|---|
| toy agent | 시험용으로 직접 작성 | 3개 파일 | 8 | 8 | 0 | 0 |
| mini-swe-agent | 키트 세션이 기존 로그 보완 | `agents/` 2개 파일 | 29 | 29 | 0 | 0 |
| browser-use | 키트 세션이 새로 삽입 | `agent/service.py` | 105 | 105 | 0 | 0 |
| smolagents | 키트 세션이 새로 삽입 | 패키지 전체 | 55 | 34 | 0 | 21 (오탐) |

### 5.1 mini-swe-agent: 요청문 개선 전후

같은 검증기로 비교했습니다. 바뀐 것은 요청문에 "전체 값 로깅, task 기록" 규칙을 넣고 다시 돌린 것뿐입니다.

| | 검증 완료 | 일부만 기록 | 누락 |
|---|---|---|---|
| 개선 전 (기존 로그) | 1 | 20 | 8 |
| 개선 후 (세션이 보완) | 29 | 0 | 0 |

세션이 고친 내용:
- `llm_query(messages=len(self.messages))` → `messages=self.messages`
- `llm_response`에 응답 전체(`response=message`) 추가
- `tool_end(result=output.get("output")[:500])` → `result=output`
- `run()`에서 처음 받은 task를 `user_input`으로 기록

### 5.2 smolagents의 누락 21개 (오탐)

- 21개 모두 실행 중에 사용자에게 묻는 `UserInputTool`의 `input()` 답변에서 출발합니다. 처음 받는 task가 아닙니다.
- 경로가 tool 출력 변환 함수 `handle_agent_output_types`의 이미지·오디오 분기(`AgentImage`, `AgentAudio`)를 지나 `PIL.Image.open`, `torch.tensor`, `sf.read` 등에 닿습니다.
- 실제로는 `UserInputTool.output_type = "string"`이라 항상 `AgentText`로만 변환되고, 결과는 `execute_tool_call`의 `tool_end(result=...)`에서 통째로 기록됩니다.
- 검증기가 `isinstance` 분기를 구분하지 않아 실행되지 않는 경로도 가능하다고 본 것입니다.

---

## 6. 개발 중 드러난 문제와 해결

| 문제 | 해결 |
|---|---|
| 이미 로그가 있는 저장소에서 세션이 질문만 남기고 종료 | 요청문에 비대화형·점검 후 보완·파일 필수 작성 명시 |
| 로그가 개수·일부만 기록 | 요청문에 전체 값 로깅 규칙과 예시 코드 추가. 로거 docstring의 "short outcome"도 "whole output"으로 수정 |
| 기준점을 사람이 써야 함 | 요청문 Part 1에 기준점 작성 단계 추가 |
| `from pkg import observatory_log`, `from . import observatory_log`를 로거로 인식 못 함 | import를 파일 경로로 해석 (일반 규칙) |
| 생성자 인자로 받은 객체(`self.model.query`)를 도착점으로 못 봄 | 기준점 호출이면 타입을 몰라도 도착점으로 봄 (일반 규칙) |
| 헬퍼 함수 안에서 남긴 로그를 인정 못 함 | 값을 그대로 넘긴 호출한 쪽 변수도 기록된 것으로 봄 (일반 규칙) |
| "사각지대"(래퍼 밖 외부 호출) 표시가 잡음만 늘림 | 로그 기록 여부와 무관하므로 제거 |

검증기 수정은 모두 특정 에이전트용 분기가 아닌 일반 규칙이며, 수정할 때마다 네 에이전트를 다시 돌려 기존 판정이 유지되는지 확인했습니다.

---

## 7. 한계

- **실행 순서 무시**: 같은 변수는 한 노드라서, 값이 쓰인 뒤에 기록해도 기록으로 봅니다.
- **분기 무시**: `if`/`isinstance` 조건을 해석하지 않아 실행되지 않는 경로도 경로로 봅니다. smolagents의 오탐 21개가 이 경우입니다.
- **필드 단위**: `self.state`는 클래스당 노드 하나라 안의 필드를 구분하지 않습니다. browser-use에서 step 번호(`self.state.n_steps`)를 넘긴 로그가 칸 표시 △로 나타나지만, 판정에는 영향이 없습니다.
- **예외 흐름 미추적**: `raise X(값)` → `except`로 가는 값은 따라가지 않습니다.
- **기준점 의존**: 기준점에 없는 출발점은 검사하지 않습니다. 예를 들어 mini의 비용 한도 입력(`interactive.py:95`)은 세션이 기준점에서 빠뜨렸습니다.
- **로그 내용은 보지 않음**: 로거가 20,000자에서 자르는 것은 판정에 반영되지 않습니다.
- **분석 범위**: `--only`로 준 파일만 프로젝트 코드로 봅니다. 범위가 좁으면 같은 패키지의 다른 모듈 호출이 외부 호출(도착점)로 잡힙니다. 현재 네 에이전트의 범위가 통일되어 있지 않습니다.

## 8. 남은 과제

1. 분석 범위 규칙 통일 (예: 에이전트 패키지 전체, 로거 제외) 후 네 에이전트 재측정
2. 검증을 `run.py`의 마지막 단계로 넣어 명령 하나로 리포트까지 생성
3. Windows에서 Claude 세션 안의 Archify `node` 실행이 권한에 막히는 문제 수정 (browser-use, smolagents는 구조도 레이아웃을 수동으로 마무리함)
4. 타입 분기(`isinstance`) 해석으로 오탐 감소 여부 검토
5. 런타임 추적(`flow_trace.py`, canary 토큰)으로 정적 판정을 실제 실행과 대조하는 범위를 toy 외 에이전트로 확대

---

## 부록: 파일 위치

| 파일 | 위치 |
|---|---|
| 키트 | `mini-swe-agent/observatory/` (다른 에이전트에는 같은 폴더를 복사) |
| 검증기 | `mini-swe-agent/PoC/verification/flow_verify.py` |
| 런타임 추적 (정답지) | `mini-swe-agent/PoC/verification/flow_trace.py` |
| toy agent | `mini-swe-agent/PoC/verification/toy_agent/` |
| 리포트 | `mini-swe-agent/PoC/verification/out_verify*/report.html` |
