# SLM wiring runs

Copied from the session scratchpad. Each run: `changes.patch` (what the model changed, against the clean agent), the files the model added, `observatory/diagram/verify-anchors.json`, `observatory/verify/` (report.html, report.json, graph.json, loop.mmd) and `run.log`.

## 01_oneshot_nvidia_budget3

one session does everything (old task.md), NVIDIA, Claude Code budget 3 USD

- `nemotron`
- `gpt-oss`
- `glm`

## 02_oneshot_nvidia

same, no budget limit

- `nemotron`
- `gpt-oss`
- `glm`
- `deepseek`

## 03_oneshot_checked_nvidia

one wiring session + checker + fix sessions + Archify diagram session

- `claude`: [report](claude/observatory/verify/report.html)
- `laguna`
- `nemotron`: [report](nemotron/observatory/verify/report.html)
- `glimmer`

## 04_oneshot_checked_deepinfra

same, DeepInfra

- `nemotron`: [report](nemotron/observatory/verify/report.html)
- `glimmer`: [report](glimmer/observatory/verify/report.html)
- `gpt-oss`

## 05_oneshot_worktree

same, in mini-swe-agent worktrees, Archify 3.0.1

- `claude`: [report](claude/observatory/verify/report.html)
- `glimmer`: [report](glimmer/observatory/verify/report.html)
- `nemotron`
- `gpt-oss`
- `laguna`

## 06_staged_worktree_aborted

first staged kit in worktrees, stopped (a session wrote into the real repo)

- `claude`
- `glimmer`
- `nemotron`
- `gpt-oss`
- `laguna`

## 07_staged_fixloop_toy

staged kit with fix sessions fed by the checker

- `glimmer`: [report](glimmer/observatory/verify/report.html)
- `nemotron`: [report](nemotron/observatory/verify/report.html)
- `gpt-oss`: [report](gpt-oss/observatory/verify/report.html)

## 08_staged_judge_toy

staged kit, checker judges independently (no fix sessions)

- `glimmer`: [report](glimmer/observatory/verify/report.html)
- `nemotron`: [report](nemotron/observatory/verify/report.html)
- `gpt-oss`: [report](gpt-oss/observatory/verify/report.html)

## 09_mini_first

mini-swe-agent, stopped in step 4

- `glimmer`: [report](glimmer/observatory/verify/report.html)
- `nemotron`: [report](nemotron/observatory/verify/report.html)

## 10_toy_anchor_rule

toy after the anchor rule (wrapper vs SDK)

- `glimmer`: [report](glimmer/observatory/verify/report.html)
- `nemotron`: [report](nemotron/observatory/verify/report.html)

## 11_toy_no_rewrite_rule

toy after 'only add lines', stopped

- `glimmer`: [report](glimmer/observatory/verify/report.html)
- `nemotron`: [report](nemotron/observatory/verify/report.html)

## 12_toy_mini

anchor normalization, user_input(), toy and mini

- `toy-glimmer`: [report](toy-glimmer/observatory/verify/report.html)
- `toy-nemotron`: [report](toy-nemotron/observatory/verify/report.html)
- `mini-glimmer`: [report](mini-glimmer/observatory/verify/report.html)
- `mini-nemotron`: [report](mini-nemotron/observatory/verify/report.html)

## 13_smolagents_browser_use

4 steps, no scripted run, smolagents and browser-use

- `smol-glimmer`: [report](smol-glimmer/observatory/verify/report.html)
- `smol-nemotron`: [report](smol-nemotron/observatory/verify/report.html)
- `bu-glimmer`: [report](bu-glimmer/observatory/verify/report.html)
- `bu-nemotron`: [report](bu-nemotron/observatory/verify/report.html)

## 14_claude_current_kit

Claude (sonnet) with the current kit and verifier: toy, mini, smolagents

- `toy-claude`: [report](toy-claude/observatory/verify/report.html)
- `mini-claude`: [report](mini-claude/observatory/verify/report.html)
- `smol-claude`: [report](smol-claude/observatory/verify/report.html)

## 15_qwen_nemotron_current_kit

Qwen3.8-27B (toy, mini, smolagents) and nemotron 4 times on smolagents, same kit as 14

- `toy-qwen`: [report](toy-qwen/observatory/verify/report.html)
- `mini-qwen`: [report](mini-qwen/observatory/verify/report.html)
- `smol-qwen`: [report](smol-qwen/observatory/verify/report.html)
- `smol-nemotron`: [report](smol-nemotron/observatory/verify/report.html)
- `smol-nemotron2`: [report](smol-nemotron2/observatory/verify/report.html)
- `smol-nemotron3`
- `smol-nemotron4`: [report](smol-nemotron4/observatory/verify/report.html)
