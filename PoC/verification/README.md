# 래퍼 누락 검증

LLM이 에이전트에 심은 로그(래퍼)가 **User Input, LLM Response, Tool Response** 세 값을 빠짐없이 기록하는지 정적으로 검증합니다. 구조도 단계(Archify + LLM)가 넘겨준 기준점에서 출발해 세 값이 흘러가는 경로만 따라가고, 흐름별 판정을 리포트로 냅니다. 검증 자체에는 LLM이 들어가지 않습니다.

```
구조도 단계 (Archify + LLM) ──▶ verify-anchors.json ──▶ flow_verify.py ──▶ report.md / report.html
```

## 구성

| 파일 | 역할 |
|---|---|
| `flow_verify.py` | 검증기. 표준 라이브러리만 사용 |
| `toy_agent/` | 기록 상태를 아는 시험용 에이전트 (`agent.py`, `llm.py`, `tools.py`, 로거 `observatory_log.py`) |
| `toy_agent/observatory/diagram/verify-anchors.json` | toy 에이전트의 기준점 |
| `flow_trace.py` | 런타임 추적. toy 에이전트를 실제로 돌려서 정적 판정이 맞는지 확인하는 정답지 |

## 실행

```bash
uv run PoC/verification/flow_verify.py PoC/verification/toy_agent \
    PoC/verification/toy_agent/observatory/diagram/verify-anchors.json PoC/verification/out_verify \
    --only agent.py llm.py tools.py
open PoC/verification/out_verify/report.html
```

결과(`out_verify/`, git에 안 올라감): `report.html`(카드와 경로 체인, 칸을 누르면 VS Code에서 그 줄이 열림), `report.md`, `report.json`.

## 기준점 파일 (`verify-anchors.json`)

| 키 | 내용 |
|---|---|
| `entry` | 진입 함수와, 그 인자 중 사용자 입력(`inputs`) |
| `sources` | 세 값이 생기는 호출: `kind`(`user_input` / `llm_response` / `tool_response`), `path`, `line`, `call`(호출식 그대로) |
| `wrappers` | 래퍼 구간(`path`, `line`~`end_line`)과 남기는 이벤트 |
| `logger` | 로그 모듈 이름 (기본 `observatory_log`) |

`call`이 코드에서 안 찾아지면 바로 멈춥니다(기준점이 코드와 어긋났다는 뜻).

## 동작

1. **코드 읽기**: AST로 함수, 클래스, import를 정리하고, 객체가 어디서 왔는지(`self.client = anthropic.Anthropic()`이면 외부 객체) 기록합니다.
2. **흐름 그래프**: 대입, 인자 → 매개변수, 반환값, `self.필드`, 컨테이너, 반복문, comprehension을 따라 "값이 A에서 B로 간다" 엣지를 만듭니다. 호출은 이름이 아니라 출처로 나눕니다.
   - 프로젝트 함수: 인자와 반환값을 연결
   - **외부 호출**(import한 라이브러리, 그걸로 만든 객체의 메서드, `exec`/`input`/`print` 같은 부작용 있는 builtin): **사용 지점**
   - 로거 호출: 인자가 값 자체(변수, `self.필드`, 그것들의 list/tuple/dict)면 **통째로 기록**, 일부(`out[:200]`, `response.stop_reason`, `[c["name"] for c in ...]`)면 **일부만 기록**
   - 값 계산(`str`, `sorted`, `.lower()`, `.get()`): 가공. `len`, `isinstance`: 흐름 끊김
3. **판정**: 출발점과 사용 지점(외부 호출, 진입 함수의 반환)의 쌍마다, 통째로 기록된 노드를 막고 길을 찾습니다.

   | 판정 | 의미 |
   |---|---|
   | 검증 완료 | 막으면 도착 못 함: 모든 경로가 기록을 지남 |
   | 일부만 기록 | 기록 없이 도착하는 경로가 있고, 그 경로엔 일부만 받는 로그뿐 |
   | 누락 | 기록 없이 도착하는 경로가 있고, 로그가 전혀 없음 |

   **사각지대**: 사용 지점이 어떤 래퍼 구간 안에도, 래퍼 구간에서 불리는 함수 안에도 없음. 구조도 단계가 못 본 출구입니다.
4. **리포트**: 흐름별로 생김 · 기록 · 쓰임 위치와 기록 없이 지나는 경로.

## toy 에이전트 결과

흐름 8개: 검증 완료 5, 일부만 기록 3, 누락 0, 사각지대 0.

- 일부만 기록: `llm.py:27`의 `llm_response`가 tool 이름만 남겨서 LLM 응답 → 다음 LLM 입력, 최종 출력. `tools.py:29`의 `tool_end`가 `out[:200]`만 남겨서 Tool 응답 → LLM 입력.
- 시험: `os.popen(reply["content"])`를 부르는 메서드를 심으면 사각지대로, `agent.py:43-44`의 task 로그를 지우면 task → LLM 입력이 누락으로 나옵니다.

## 런타임 추적 (`flow_trace.py`)

```bash
cd PoC/verification && uv run flow_trace.py toy && open out_flow/flow.html
```

가짜 모델 응답, tool 출력, 사용자 입력에 `CANARY_<LLM|TOOL|USER>_<n>` 토큰을 심고 toy 에이전트를 실행해서(API 키 불필요), `sys.monitoring`(Python 3.12)으로 토큰이 어디로 가는지와 `observatory_log`에 남는지를 봅니다. 위 정적 판정과 같은 두 곳(`llm_response`, `tool_end`)이 나옵니다. `--hidden`이면 system prompt에 붙인 작업 폴더 경로가 "어떤 래퍼도 못 본 입력"으로 나옵니다.

## 한계

- 예외에 실려 가는 값(`raise X(msg)` → `except`)은 따라가지 않습니다.
- 그래프에 실행 순서가 없어서, 값이 쓰인 뒤에 기록되는 경우도 기록으로 봅니다.
- 기준점에서 빠진 출발점의 값은 추적하지 않습니다.
- 타입을 모르는 매개변수(인자로 받은 외부 객체)의 메서드 호출은 가공으로 봐서 사용 지점을 놓칠 수 있습니다.
- 경로 중간 칸의 줄 번호는 그 변수나 필드가 처음 나온 곳입니다.
