# I/O 계측 검증 도구

LLM 에이전트에 심은 로그(래퍼)가 **LLM 호출, Tool 실행, 사용자 입력을 빠짐없이 덮는지** 확인하는 도구입니다.

1. **정적 검증**: 코드를 분석해서 I/O 지점을 찾고, 진입점에서 그 지점까지 가는 모든 호출 경로가 래퍼를 지나는지 판정합니다. 결과는 브라우저 뷰어(call graph, CFG, 검증 목록)로 봅니다.
2. **누락 보완**: 뷰어에서 고른 항목을 요청서로 만들어, Agent Observatory 키트와 같은 방식으로 Claude Code에 래퍼 삽입을 요청합니다.
3. **실행 확인**: 에이전트를 실제로 돌려서 키트 dashboard에 이벤트가 빠짐없이 남는지 봅니다.

정적 검증은 규칙에 걸리는 호출만 볼 수 있고, 실행 확인은 실제로 실행된 경로만 볼 수 있습니다. 둘을 같이 써야 서로의 빈틈이 메워집니다.

## 구성

| 파일 | 역할 |
|---|---|
| `io_audit.py` | 메인. I/O 지점 탐지, call graph 가지치기, 래퍼 판정, 뷰어(`index.html`) 생성 |
| `io_rules.py` | 추출(E1~E3), Tool 실행(T1~T4), 사용자 입력(I1) 규칙 |
| `llm_rules.py` | LLM 호출 규칙(R1 SDK, R2 중간 라이브러리, R3 HTTP) |
| `audit_rules.py` | 동적 호출의 이름 연결, 역추적 (`io_audit.py`가 사용) |
| `audit.py` | AST, call graph, CFG 기본 엔진 (`io_audit.py`가 사용. 단독 실행도 가능) |
| `viewer.html` | 뷰어 템플릿. `io_audit.py`가 데이터를 채워 `index.html`로 저장 |
| `request_wrappers.py` | 래퍼 요청서를 키트의 `run.py` 방식으로 Claude Code에 보냄 |
| `run_mini_observed.py` | mini-swe-agent를 Claude로 한 번 실행해서 dashboard 기록을 남김 |
| `toy_agent/` | 정답을 알고 있는 시험용 에이전트. 도구가 맞게 판정하는지 확인할 때 씀 |

키트는 리포지토리 루트의 `observatory/`에 설치되어 있고, 로거는 `src/minisweagent/observatory_log.py`에 복사되어 있습니다.

## 준비

| 준비물 | 방법 |
|---|---|
| [uv](https://docs.astral.sh/uv/) | 파이썬 패키지는 스크립트 머리에 적혀 있어서 `uv run`이 알아서 설치합니다 |
| Graphviz `dot` | `winget install Graphviz.Graphviz` (설치 후 새 터미널) 또는 `conda install -c conda-forge graphviz`. `dot -V`로 확인 |
| `.env` | 실제 실행 때만 필요. 리포지토리 루트에 `ANTHROPIC_API_KEY=...` |

**Windows**: 키트가 파일을 플랫폼 기본 인코딩으로 읽어서 한글 Windows(cp949)에서 깨집니다. 터미널에서 먼저 설정하세요.

```powershell
$env:PYTHONUTF8 = "1"
```

## 1. 정적 검증

리포지토리 루트에서 실행합니다.

```powershell
uv run --python 3.12 PoC/verification/io_audit.py src/minisweagent PoC/verification/out_io --entry DefaultAgent.run --exclude observatory_log
start PoC/verification/out_io/index.html
```

- 분석 범위는 **패키지 전체**로 줍니다. LLM 호출은 `models/`, 명령 실행은 `environments/`에 있어서 `agents/`만 주면 끝까지 이어지지 않습니다.
- `--exclude observatory_log`는 로거 자체를 분석에서 뺍니다.
- 끝에 `checks: {'covered': .., 'bypass': .., 'outside': ..}`가 나오면 완료입니다.

| 옵션 | 설명 |
|---|---|
| `--entry` | 에이전트 루프의 시작 함수 (예: `DefaultAgent.run`, `MultiStepAgent.run`). 판정에 필요 |
| `--only` | 지정한 파일만 분석 (`src_root` 기준 경로) |
| `--exclude` | 이름에 이 글자가 들어간 함수를 뺌 |
| `--max-links` | 그래프에 남길 함수가 거쳐도 되는 이름 연결 수 (기본 1. 0이면 동적 호출 경로가 끊김) |
| `--e2-depth` | `re`/`json`/`yaml` 파싱을 LLM 호출 아래 몇 단계까지 추출로 셀지 (기본 2) |

### 결과 파일 (`out_io/`, git에 올라가지 않음)

| 파일 | 내용 |
|---|---|
| `index.html` | 뷰어 |
| `callgraph_io.svg` | I/O와 관련된 함수만 남긴 call graph |
| `cfg/<함수>.svg` | 남은 함수의 CFG |
| `io_checks.tsv` | I/O 지점별 판정, 래퍼, 우회 경로 |
| `io_sites.tsv` | 규칙에 걸린 모든 지점과 실제로 셌는지 여부 |
| `io_functions.tsv` | 남긴 함수별 범주, 진입점 도달 여부 |

### 뷰어 보는 법

- **검증 목록**(오른쪽): I/O 지점마다 자동 판정이 붙어 있습니다.

  | 판정 | 의미 | 할 일 |
  |---|---|---|
  | 우회 경로 | 진입점에서 래퍼를 하나도 거치지 않고 도달하는 길이 있음. 그 길을 같이 표시 | 먼저 확인. 진짜 누락이면 "수정 필요", 아니면 "오탐" |
  | 래퍼 통과 | 모든 길이 래퍼를 거침 | 래퍼를 눌러 CFG에서 정상·예외 경로 모두 로그가 있는지 확인 |
  | 진입점 밖 | 에이전트 루프에서 도달하지 않음 | 셋업·정리·테스트처럼 루프 밖 동작인지 확인 |

  항목마다 검토 표시(확인함/수정 필요/오탐/범위 밖)를 고를 수 있고, 브라우저에 저장됩니다. "검토 결과 내보내기"로 TSV를 받을 수 있습니다.
- **call graph**(왼쪽): 노드를 누르면 CFG, 호출 관계, VS Code 링크가 나옵니다. 상단 **범례** 버튼에 색과 선의 의미가 있습니다.
  - 채운 상자: I/O 호출을 직접 포함한 함수 (색 = 범주)
  - 점선 상자: I/O에 도달 가능한 길목
  - 타원: 실제 I/O 호출 지점
  - 파란 점선: 이름 연결, 청록 점선: 등록 함수 연결 (둘 다 정적 추정)

### 판정 기준

- **I/O 지점**: 규칙(`llm_rules.py`, `io_rules.py`)에 걸린 호출. 규칙은 "오탐은 허용, 누락은 불허" 원칙이고, 사람이 검토해서 고치는 파일입니다.
- **래퍼**: 범주에 맞는 로그를 찍는 함수. 문자열 태그(`[LLM_`, `[TOOL_`, `[INPUT_`)나 키트 함수(`llm_query`, `tool_start` 등. 사용자 입력은 `tool_start("user_input", ...)`), 또는 그런 로그 보조 함수를 직접 부르는 함수.
- **판정**: 진입점에서 래퍼를 피해서 I/O 지점에 가는 길이 있으면 우회, 없으면 통과.

## 2. 누락 보완 (래퍼 요청)

1. 뷰어의 검증 목록에서 항목을 체크합니다("보이는 항목 선택"으로 필터에 걸린 것을 한 번에 고를 수 있음).
2. **"선택 N개로 래퍼 요청 만들기"** → 요청서 확인·수정 → **`wrap_request.md`로 저장**
3. 리포지토리 루트에서 보냅니다.

   ```powershell
   uv run --python 3.12 PoC/verification/request_wrappers.py wrap_request.md
   ```

   키트의 `observatory/run.py`가 Claude Code를 실행하는 함수를 그대로 씁니다(기억 없는 새 세션, 파일 편집·`uv run`·`git diff`만 허용, 커밋 안 함, 예산 상한). 끝나면 키트의 연결 점검도 돌립니다.

   | 옵션 | 기본값 |
   |---|---|
   | `--model` | `sonnet` (키트와 같음. 저렴하게는 `haiku`) |
   | `--budget` | `3` (USD) |

4. `git diff`로 바뀐 코드를 검토하고, 1번 정적 검증을 다시 돌려서 항목이 "래퍼 통과"가 됐는지 봅니다.

요청서를 Claude Code에 직접 붙여 넣어도 됩니다. **Claude Code가 실행되고 비용이 듭니다.**

## 3. 실행 확인 (dashboard)

터미널 두 개를 씁니다. 둘 다 리포지토리 루트에서 `$env:PYTHONUTF8 = "1"`을 먼저 설정합니다.

```powershell
# 터미널 1: dashboard (http://127.0.0.1:8000, 끌 때는 Ctrl+C)
uv run --no-project --python 3.12 observatory/dashboard.py

# 터미널 2: mini-swe-agent 실행
$env:OBSERVATORY_RUNS = "$PWD\observatory\runs"
uv run --with mcp --env-file .env python PoC/verification/run_mini_observed.py "할 일" anthropic/claude-haiku-5-5 8
```

- 인자: 작업, 모델(기본 `anthropic/claude-haiku-5-5`), 최대 step(기본 8)
- `OBSERVATORY_RUNS`가 없으면 아무것도 기록하지 않고 평소처럼 실행됩니다. `mini` CLI로 실행할 때도 이 변수만 설정하면 기록됩니다.
- `--with mcp`: 리포지토리 환경의 litellm이 `mcp` 패키지 없이는 import되지 않아서 실행할 때만 추가합니다.
- **에이전트의 명령은 이 PC에서 실제로 실행됩니다**(`PoC/verification/mini_workspaces/<시각>/`에서 시작하지만 샌드박스는 아님). 안전한 작업만 시키세요.

## 4. 토이 에이전트로 도구 시험하기

`toy_agent/`는 기대 판정이 `agent.py` 맨 위에 적혀 있는 작은 에이전트입니다(Claude Haiku, 키트 로거 계측). 규칙이나 판정을 고친 뒤 결과가 기대와 같은지 확인할 때 씁니다.

```powershell
# 정적 검증 (리포지토리 루트)
uv run --python 3.12 PoC/verification/io_audit.py PoC/verification/toy_agent PoC/verification/out_toy --entry ToyAgent.run --only agent.py llm.py tools.py
```

```powershell
# 실행 (toy_agent 폴더에서, dashboard는 toy_agent/observatory/dashboard.py)
cd PoC/verification/toy_agent
$env:OBSERVATORY_RUNS = "$PWD\observatory\runs"
uv run --no-project --python 3.12 --with anthropic python run_scripted.py                     # 가짜 모델, 비용 없음
uv run --no-project --python 3.12 --with anthropic --env-file ..\..\..\.env python -c "from agent import ToyAgent; print(ToyAgent().run('현재 폴더의 .py 파일 개수를 알려줘', max_steps=6))"
```

누락을 시험하려면 래퍼를 거치지 않는 호출(예: `subprocess.run`을 직접 부르는 메서드)을 하나 심고 정적 검증을 돌려서 "우회 경로"로 나오는지 봅니다.

## 다른 프로젝트에 적용하기

1. 이 폴더의 파일 7개(`io_audit.py`, `io_rules.py`, `llm_rules.py`, `audit_rules.py`, `audit.py`, `viewer.html`, `request_wrappers.py`)를 대상 리포지토리의 아무 폴더에 복사합니다.
2. 래퍼 요청과 dashboard까지 쓰려면 키트를 그 리포지토리 루트에 `observatory/`로 설치합니다.
3. 패키지 경로와 진입점만 바꿔서 실행합니다.

```powershell
uv run --python 3.12 verification/io_audit.py src/smolagents verification/out_io --entry MultiStepAgent.run
uv run --python 3.12 PoC/verification/io_audit.py browser_use PoC/verification/out_io --entry Agent.run
```

## 알려진 한계

- **규칙에 안 걸리면 안 보입니다.** 예: smolagents의 로컬 코드 실행기는 자체 인터프리터라 `exec`를 쓰지 않아 어떤 규칙에도 걸리지 않습니다. 실행 확인(dashboard)으로 보완합니다.
- **문자열만 보고 거는 규칙은 오탐이 납니다.** 예: mini-swe-agent의 `get_content_string`(메시지를 화면용 문자열로 바꾸는 함수)이 `'tool_use'` 문자열 때문에 추출로 잡힙니다.
- **동적 호출은 추정입니다.** `self.model.query()` 같은 호출은 같은 이름의 모든 메서드로, `registry[name](...)`은 같은 클래스에 등록된 함수로 넓게 연결합니다. 실제로는 없는 연결이 생길 수 있습니다.
- **래퍼 통과는 call graph 수준 판정입니다.** 래퍼 함수 안에서 정상·예외 경로 모두 로그가 찍히는지는 CFG로 직접 확인해야 합니다.
- **CFG 그림의 한계**: py2cfg 위에서 그려서 bare `raise`가 `finally`로 가는 간선이 빠지고, `match` 문은 분기가 나뉘지 않습니다.
