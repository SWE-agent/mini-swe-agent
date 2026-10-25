# Agent Observatory 키트

에이전트(mini-swe-agent, smolagents, browser-use, …) 저장소 루트에 이 `observatory/` 폴더를 넣고 명령 하나를 실행하면, Claude Code가 그 에이전트를 로거에 연결하고 왼쪽 구조도를 그려 줍니다. dashboard와 로거는 어떤 에이전트에서도 똑같습니다.

## 구성

| 파일 | 역할 | 에이전트마다 |
|---|---|---|
| `observatory_log.py` | 로거. 에이전트 패키지에 복사해 호출만 심는다. 표준 라이브러리만 사용 | 같음 |
| `dashboard.py`, `static/index.html` | dashboard 서버와 화면. 에이전트를 전혀 모르고 로그 파일만 읽는다 | 같음 |
| `diagram/agent-architecture.json` → `.html` | 왼쪽 구조도. Claude가 spec을 쓰고 Archify가 HTML을 만든다 | **에이전트별로 생성** |
| 에이전트 코드 안의 로거 호출 | LLM 호출과 tool 실행 위치에 심는 호출 | **에이전트별로 Claude가 찾아서 추가** |
| `run.py`, `prompt/*.md`, `flow_verify.py`, `selftest.py` | 실행 스크립트, 단계별 Claude 요청, 정적 검증기, 키트 자체 테스트 | 같음 |

## 동작 방식

```
에이전트 ──observatory_log 호출──▶ observatory/runs/*.jsonl ──▶ dashboard.py ──SSE──▶ 브라우저
(LLM 호출·tool 실행 지점)         (OBSERVATORY_RUNS를 켰을 때만 기록)     (왼쪽: Archify 구조도)
```

`OBSERVATORY_RUNS` 환경변수가 없으면 로거는 아무것도 하지 않아서, 평소 실행은 전과 똑같습니다.

## 실행 전에 준비할 것

| 준비물 | 방법 |
|---|---|
| `uv`, `node`(18 이상), `git` | 설치되어 있어야 합니다. |
| 에이전트 저장소 | GitHub에서 clone한 저장소. `origin` 리모트가 있어야 Archify 검증이 통과합니다. |
| **Archify 저장소 (v3.0.1)** | 에이전트 폴더와 **같은 상위 폴더**에 `archify/`로 둡니다. 다른 곳이면 `--archify-dir <경로>/archify` 또는 `ARCHIFY_DIR`. 없으면 구조도 없이 로깅 연결만 합니다. |
| **Claude Code** | `claude` CLI가 PATH에 있고 로그인되어 있어야 합니다 (아니면 `CLAUDE_BIN`). 이 계정으로 실행되고 비용이 듭니다. |
| 에이전트의 API 키 | 로깅 연결에는 필요 없습니다. 에이전트를 실제 모델로 돌릴 때 그 에이전트가 쓰는 키(`ANTHROPIC_API_KEY` 등)를 평소대로 설정합니다. |

## 실행

```bash
# 에이전트 저장소 루트에서
uv run observatory/run.py            # Claude가 로깅 연결 + 구조도 작성, 결과 점검
uv run observatory/run.py --serve    # 끝나고 dashboard 실행 (기본 http://127.0.0.1:8000)

# dashboard를 띄운 상태에서, 다른 터미널에서 에이전트를 평소처럼 실행
OBSERVATORY_RUNS=$PWD/observatory/runs <에이전트 실행 명령>
```

로깅 세션은 몇 분 걸립니다. 끝나면 `git diff`로 Claude가 바꾼 것을 검토하세요. 커밋은 키트가 하지 않습니다. 커밋·push한 뒤 `uv run observatory/run.py --archify-only`를 실행하면 구조도의 GitHub 링크가 그 커밋을 가리킵니다.

## 옵션

| 옵션 | 설명 |
|---|---|
| `--serve`, `--serve-only`, `--port` | dashboard를 실행(기본 8000). `--serve-only`는 다시 띄울 때 |
| `--archify-dir` | Archify의 `archify/` 디렉터리 (기본 `../archify/archify`) |
| `--no-claude` | Claude 세션 없이 점검과 구조도만 |
| `--archify-only` | 구조도 spec을 현재 HEAD에 맞추고 HTML만 다시 생성 |
| `--model`, `--budget` | Claude 모델(기본 `sonnet`)과 세션당 최대 비용 USD(기본 3) |
| `--test "<명령>"` | 끝나고 저장소에서 실행할 테스트 명령 (예: `"uv run pytest -q"`) |

## 알아 둘 점

- Claude Code는 `--no-session-persistence`와 `--setting-sources project`로 실행되고, 파일 편집과 `uv run`, `git diff`, 그리고 Archify의 `validate`만 허용됩니다. 커밋은 하지 않습니다.
- 로거는 이름에 `KEY`, `TOKEN`, `SECRET`, `PASSWORD`가 들어간 환경변수 값을 기록에서 지웁니다. 하지만 에이전트가 설정 안에 따로 가진 비밀값(예: browser-use의 `sensitive_data`)은 모르므로, 프롬프트가 필드에 넣지 말라고 요청합니다. 결과를 검토하세요.
- 에이전트가 정상 종료를 예외로 알리는 경우(mini-swe-agent의 `Submitted` 등)는 `tool_error`가 아니라 `tool_end`의 결과로 보고하게 요청합니다.
- 이 구조에서는 dashboard가 에이전트를 직접 실행하지 않습니다(예전 에이전트별 dashboard의 Run 버튼은 없음). 에이전트는 평소처럼 실행하고 dashboard는 기록을 보여 줍니다.
- 구조도의 GitHub 링크는 커밋 전에는 임시 git 객체(HEAD와 브랜치는 그대로)를 가리켜서, 커밋·push 후 `--archify-only`를 해야 열립니다.
- 키트 자체 테스트: `uv run observatory/selftest.py`
