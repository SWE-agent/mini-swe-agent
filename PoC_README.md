# PoC: mini-swe-agent 호출 이력 대시보드

mini-swe-agent가 **LLM에 무엇을 물었고, 어떤 bash 명령을 실행했는지** step 단위로 뽑아서 웹에서 보여주는 PoC입니다. browser-use, smolagents용으로 만든 Agent Observatory와 디자인과 방식이 같습니다.

- **왼쪽:** Archify로 그린 mini-swe-agent agent loop 구조도. LLM 호출과 tool(bash) 실행이 일어나는 위치가 표시돼 있고, 노드의 소스 링크를 누르면 해당 코드가 열립니다.
- **오른쪽:** 실행 이력. step마다 Thought → LLM 호출(모델이 고른 bash 명령) → 명령 실행(종료 코드, 소요 시간, 제출 여부) → Observation 순서로 쌓입니다.

browser-use, smolagents와 같은 방법으로, **agent loop 안의 LLM 호출 지점과 tool 실행 지점에 로그를 심고** 대시보드가 그 로그를 읽습니다.

## 1. 실행 방법

browser-use 대시보드와 같은 방법으로 실행합니다.

준비물: Python 3.10+, [uv](https://docs.astral.sh/uv/), Anthropic API 키

```bash
git clone https://github.com/NiceGuy1313/mini-swe-agent.git
cd mini-swe-agent

# 1. API 키 설정 (처음 한 번만)
cp .env.example .env
# .env를 열어서 ANTHROPIC_API_KEY= 뒤에 실제 키를 넣으세요. .env는 git에 안 올라갑니다.

# 2. 의존성 설치
uv sync

# 3. 서버 실행
uv run uvicorn examples.agent_dashboard.app:app --port 8000
```

브라우저에서 http://127.0.0.1:8000 을 엽니다.

- `.env`는 서버가 시작할 때만 읽습니다. 키를 바꿨으면 서버를 재시작하세요.
- 다른 대시보드와 동시에 띄울 때는 포트를 나눠 쓰세요(예: browser-use `8000`, smolagents `8001`, mini-swe-agent `8002`).

> ⚠️ **agent가 실행하는 bash 명령은 이 컴퓨터에서 실제로 실행됩니다.** 실행마다 `examples/agent_dashboard/runs/<run_id>-workspace` 폴더를 새로 만들어 그 안에서 시작하지만, 샌드박스는 아닙니다. 위험한 작업은 시키지 마세요.

## 2. 사용 방법

1. 오른쪽 위 입력칸에 **task**, **최대 step 수**, **모델**(기본 `claude-sonnet-5`)을 넣고 **Run**을 누릅니다.
   - 모델 칸에는 Claude 모델 이름을 그대로 쓰면 됩니다. 다른 제공자는 LiteLLM 형식(`openai/gpt-5` 등)으로 쓰고, 해당 키를 `.env`에 넣으세요.
2. 오른쪽 **Call history**에 step별 이력이 실시간으로 쌓입니다.
   - `Thought`: 모델이 쓴 생각
   - `LLM` 행: LLM 호출 1회와 모델이 고른 bash 명령. 비용과 소요 시간
   - `TOOL` 행: 실제 실행된 명령, 종료 코드(0이 아니면 빨간색), 소요 시간. 작업 제출은 `submitted`로 표시
   - `STEP` 행: 응답 형식 오류(`FormatError`) 등 tool 밖에서 난 문제와 모델에게 돌려준 피드백
   - `Observation`: 다음 step에 모델이 받는 명령 실행 결과 (펼쳐서 보기)
   - `SUBMITTED` 배지: 작업을 제출한 step
3. 상단 카운터: steps / LLM 호출 수 / tool 실행 수 / 비용 합계 / 에러 수
4. **Raw log** 탭에서 가공하지 않은 이벤트 전체를 볼 수 있습니다.
5. **Export JSON**으로 전체 이력을 받습니다. 실행이 끝날 때마다 `examples/agent_dashboard/runs/<run_id>.json`에도 자동 저장됩니다(git에는 안 올라감).

실행 도중이나 끝난 뒤에 페이지를 새로 열어도 이력이 처음부터 다시 표시됩니다. 한 번에 하나의 실행만 할 수 있습니다.

## 3. 이력 JSON 형식

```jsonc
{
  "id": "…", "task": "…", "model": "claude-sonnet-5", "status": "finished",
  "events": [
    { "seq": 1, "kind": "llm_query",    "step": 1, "data": { "model": "anthropic/claude-sonnet-5", "messages": 2 } },
    { "seq": 2, "kind": "llm_response", "step": 1, "data": { "commands": ["cat fib.py"], "cost": 0.0021, "duration_seconds": 2.4 } },
    { "seq": 3, "kind": "step",         "step": 1, "data": { "type": "thought", "content": "THOUGHT: …" } },
    { "seq": 4, "kind": "tool_start",   "step": 1, "data": { "action": "bash", "command": "cat fib.py" } },
    { "seq": 5, "kind": "tool_end",     "step": 1, "data": { "action": "bash", "returncode": 0, "duration_seconds": 0.01 } },
    { "seq": 6, "kind": "step",         "step": 1, "data": { "type": "observation", "content": "<returncode>0</returncode> …" } }
  ]
}
```

`kind`는 `run`, `llm_query`, `llm_response`, `llm_error`, `tool_start`, `tool_end`, `tool_error`, `step` 중 하나입니다. browser-use, smolagents 버전과 같은 형식이라 세 agent의 이력을 같은 도구로 분석할 수 있습니다. `step` 이벤트의 `type`은 `thought`, `observation`, `interrupt`(형식 오류 등), `exit`(제출, 제한 도달) 중 하나입니다.

## 4. 동작 원리와 코드 위치

browser-use, smolagents와 같은 방식으로 mini-swe-agent 코어(`agents/default.py`)에 경계 로그(DEBUG)를 심었습니다. 기존 `agent` 로거를 써서 `mini` CLI 화면에는 아무것도 추가로 찍히지 않습니다.

| 로그 | 심은 위치 | 남는 정보 |
|---|---|---|
| `[LLM_QUERY]` | `DefaultAgent.query`: `model.query` 직전 | step, 모델, 메시지 수 |
| `[LLM_RESPONSE]` | 모델이 파싱된 액션을 돌려준 직후 | step, bash 명령 목록, 비용 |
| `[LLM_ERROR]` | `model.query` 실패(응답 형식 오류 `FormatError` 포함) | step, 에러 종류 |
| `[TOOL_EXECUTION_START/END/ERROR]` | `DefaultAgent.execute_action`(새로 추가): `env.execute` 앞뒤 | 명령, 종료 코드/에러 종류, 소요 시간 |

- **tool 실행은 한 지점으로 모았습니다.** 원래 CLI용 `InteractiveAgent`는 `env.execute`를 직접 불러서, `DefaultAgent`에만 로그를 심으면 CLI 실행이 빠졌습니다. 그래서 `execute_action` 하나를 만들고 두 agent가 모두 여기를 거치게 했습니다.
- **작업 제출은 `[TOOL_EXECUTION_ERROR] error_type=Submitted`로 남습니다.** mini-swe-agent는 완료 신호를 `Submitted` 예외로 보내기 때문입니다. 화면에서는 성공한 제출로 표시합니다.
- **step 결과는 `add_messages`에서 받습니다.** mini-swe-agent에는 step 콜백이 없어서, 대시보드의 `DashboardAgent`(`DefaultAgent` 상속)가 모든 메시지가 지나는 `add_messages`에서 Thought, observation, 형식 오류, 종료 메시지를 받습니다.

| 파일 | 역할 |
|---|---|
| `src/minisweagent/agents/default.py` | LLM/tool 경계 로그, `execute_action` |
| `src/minisweagent/agents/interactive.py` | CLI agent도 `execute_action`을 거치게 변경 (1줄) |
| `examples/agent_dashboard/app.py` | FastAPI 서버. 로그 → 이력 변환(`DashboardLogHandler`), `DashboardAgent`, SSE, export |
| `examples/agent_dashboard/static/index.html` | 화면 |
| `examples/agent_dashboard/agent-architecture.json` | 구조도 원본 spec (Archify 형식) |
| `examples/agent_dashboard/static/agent-architecture.html` | Archify 3.0.1로 생성한 구조도 (생성 결과) |
| `tests/agents/test_boundary_logging.py` | 코어 경계 로그 테스트 (`DefaultAgent`, `InteractiveAgent`) |
| `examples/agent_dashboard/test_app.py` | 로그 → 이력 변환 테스트 |

테스트(네트워크/키 불필요, mock 없음): `uv run pytest tests/agents/test_boundary_logging.py examples/agent_dashboard/test_app.py`

## 5. 아직 안 되는 것 / 다음 작업

- **구조도를 고치려면 Archify가 필요합니다.** spec(`agent-architecture.json`)을 수정한 뒤 Archify로 검증하고 HTML을 다시 생성하세요. 명령은 `examples/agent_dashboard/README.md`에 있습니다.
- **대시보드는 `LocalEnvironment`만 씁니다.** Docker 등 다른 환경도 `execute_action`을 거치므로 로그는 똑같이 남지만, 대시보드에서 고르게 하려면 `execute_agent`를 고쳐야 합니다.
- **LiteLLM에 가격 정보가 없는 모델은 비용이 $0으로 나옵니다.** 이 경우 비용 제한이 걸리지 않으니 step 제한으로 실행 길이를 조절하세요.
- **응답 형식 오류가 난 호출의 비용은 `[LLM_RESPONSE]`에 안 남습니다.** 실행 전체 비용에는 포함됩니다.
