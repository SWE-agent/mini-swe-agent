# Agent Observatory 키트

에이전트(mini-swe-agent, smolagents, browser-use, …) 저장소 루트에 이 `observatory/` 폴더를 넣고 명령 하나를 실행하면, Claude Code가 그 에이전트를 로거에 연결하고 정적 검증기가 빠진 곳이 없는지 확인합니다. dashboard는 실행 기록과 함께, 코드에서 읽어 낸 에이전트 루프를 왼쪽에 그립니다. dashboard, 로거, 검증기는 어떤 에이전트에서도 똑같습니다.

## 구성

| 파일 | 역할 | 에이전트마다 |
|---|---|---|
| `observatory_log.py` | 로거. 에이전트 코드 옆에 복사해 호출만 심는다. 표준 라이브러리만 사용 | 같음 |
| `flow_verify.py` | 정적 검증기. 기준점에서 값의 흐름을 따라가 로그가 통째로 기록하는지 판정하고, 에이전트 루프 그림(`verify/loop.mmd`)을 그린다 | 같음 |
| `dashboard.py`, `static/index.html` | dashboard 서버와 화면. 왼쪽은 `verify/loop.mmd`의 에이전트 루프, 오른쪽은 실행 기록 | 같음 |
| `run.py`, `prompt/*.md`, `selftest.py` | 실행 스크립트, 단계별 Claude 요청, 키트 자체 테스트 | 같음 |
| `candidates.py` | `--scope`로 고른 파일에서 기준점이 될 수 있는 호출(구현이 보이지 않는 호출)을 모두 뽑아 1단계에 넘긴다 | 같음 |
| `diagram/verify-anchors.json` | 검증기의 기준점 (진입 함수, LLM 응답과 사용자 입력이 들어오는 호출) | **Claude가 작성** |
| 에이전트 코드 안의 로거 호출 | LLM 호출, tool 실행, 사용자 입력 위치에 심는 호출 | **Claude가 추가** |

## 동작 방식

```
에이전트 ──observatory_log 호출──▶ observatory/runs/*.jsonl ──▶ dashboard.py ──SSE──▶ 브라우저
(이벤트마다 호출 위치 at=파일:줄)  (OBSERVATORY_RUNS를 켰을 때만 기록)        (왼쪽: verify/loop.mmd의 루프)
```

`OBSERVATORY_RUNS` 환경변수가 없으면 로거는 아무것도 하지 않아서, 평소 실행은 전과 똑같습니다.

`run.py`는 단계마다 Claude Code 세션을 하나씩 엽니다. 검증기는 각 단계에 로그를 넣을 위치를 알려 주고, 마지막에 결과를 독립적으로 판정합니다. 판정 결과는 세션에 돌려주지 않습니다: 모델이 검증기 출력에 맞춰 코드를 바꾸면 검증의 의미가 없어지기 때문입니다.

| 단계 | Claude가 하는 일 | 키트가 하는 일 |
|---|---|---|
| 1 | 기준점 작성 | 로거를 진입 함수 옆에 복사, 로그를 넣기 전의 값 흐름 기록 |
| 2, 3, 4 | LLM, tool, 사용자 입력 로그 추가 | 검증기가 찾은 호출 위치를 각 단계에 전달 |
| - | | 판정 (아래), 루프 그림 작성 |

판정은 다섯 가지를 봅니다. 하나라도 걸리면 `run.py`는 실패로 끝나고, 고칠지는 사람이 리포트를 보고 정합니다.

- 코드가 컴파일되고 쓰는 이름이 모두 정의되어 있는가 (실행 오류)
- 모든 값 흐름이 로그에 통째로 기록되는가 (`verify/report.html`)
- 로그를 넣기 전에 있던 값 흐름이 사라지지 않았는가 (사라졌다면 에이전트 동작이 바뀐 것)
- 에이전트 코드에서 지워진 줄이 없는가 (들여쓰기 변화, import 줄에 이름 추가, `return 식`을 변수에 받아 반환하는 것은 허용)
- 검증기 파일이 바뀌지 않았는가

dashboard 왼쪽 그림은 검증기가 코드에서 결정적으로 그립니다(모델은 그리지 않습니다). 진입 함수, 로그가 있는 함수, 밖으로 나가는 호출(LLM, tool, 사용자 입력), 상태만 남기고 그 사이의 보조 함수는 접습니다. 로그가 있는 함수는 박스의 이름 아래에 로그 호출마다 `[LOG] <이벤트> @ <파일:줄>`을 한 줄씩 적습니다. 박스를 누르면 VS Code에서 그 코드가 열리고, 기록의 행을 누르면 그 이벤트를 남긴 박스가 강조됩니다.

## 실행 전에 준비할 것

| 준비물 | 방법 |
|---|---|
| `uv`, `git` | 설치되어 있어야 합니다. |
| **Claude Code** | `claude` CLI가 PATH에 있고 로그인되어 있어야 합니다 (아니면 `CLAUDE_BIN`). 이 계정으로 실행되고 비용이 듭니다. |
| 에이전트의 API 키 | 로깅 연결에는 필요 없습니다. 에이전트를 실제 모델로 돌릴 때 그 에이전트가 쓰는 키(`ANTHROPIC_API_KEY` 등)를 평소대로 설정합니다. |
| 인터넷 (dashboard) | 왼쪽 그림은 Mermaid를 jsDelivr에서 불러와 그립니다. |

## 실행

```bash
# 에이전트 저장소 루트에서
uv run observatory/run.py            # Claude가 단계별로 로깅 연결, 검증, 결과 점검
uv run observatory/run.py --serve    # 끝나고 dashboard 실행 (기본 http://127.0.0.1:8000)

# dashboard를 띄운 상태에서, 다른 터미널에서 에이전트를 평소처럼 실행
OBSERVATORY_RUNS=$PWD/observatory/runs <에이전트 실행 명령>
```

세션들은 합쳐서 수십 분 걸릴 수 있습니다. 끝나면 `git diff`로 Claude가 바꾼 것을 검토하세요. 커밋은 키트가 하지 않습니다.

## 옵션

| 옵션 | 설명 |
|---|---|
| `--serve`, `--serve-only`, `--port` | dashboard를 실행(기본 8000). `--serve-only`는 다시 띄울 때 |
| `--no-claude` | Claude 세션 없이 판정과 루프 그림만 다시 (값 흐름과 지워진 줄 비교는 세션이 있을 때만) |
| `--model`, `--budget` | Claude 모델(기본 `sonnet`)과 세션당 최대 비용 USD(기본 3, 0이면 제한 없음) |
| `--scope <파일...>` | 에이전트를 이루는 파일. 1단계 세션은 이 파일만 읽고, 그 안에서 구현이 보이지 않는 호출 목록에서 기준점을 고른다 |
| `--test "<명령>"` | 끝나고 저장소에서 실행할 테스트 명령 (예: `"uv run pytest -q"`) |

## 알아 둘 점

- Claude Code는 `--no-session-persistence`와 `--setting-sources project`로 실행되고, 파일 편집과 `uv run`, `git diff`만 허용됩니다. 커밋은 하지 않습니다. 요청문은 에이전트 폴더 밖의 파일과 검증기(`flow_verify.py`, `verify/`)를 읽거나 쓰지 말라고 합니다.
- 로거는 이름에 `KEY`, `TOKEN`, `SECRET`, `PASSWORD`가 들어간 환경변수 값을 기록에서 지웁니다. 하지만 에이전트가 설정 안에 따로 가진 비밀값(예: browser-use의 `sensitive_data`)은 모르므로 결과를 검토하세요.
- 에이전트가 정상 종료를 예외로 알리는 경우(mini-swe-agent의 `Submitted` 등)는 `tool_error`가 아니라 `tool_end`의 결과로 보고하게 요청합니다.
- 키트 자체 테스트: `uv run observatory/selftest.py`
