#!/usr/bin/env bash
# Quick health + progress snapshot for the verified run
OUT=/data/jiahao/code/mini-swe-agent/experiments/qwen3_baseline_verified
LOG=/data/jiahao/code/mini-swe-agent/experiments/logs/verified.log
echo "### $(date -Iseconds)"
echo "-- tmux --"
tmux ls 2>&1 | grep -q qwen_baseline && echo "qwen_baseline: ALIVE" || echo "qwen_baseline: DEAD"
echo "-- vLLM PID 2172205 --"
ps -p 2172205 -o pid,etime,pcpu,rss --no-headers 2>&1 || echo "vLLM: DEAD"
echo "-- GPUs 0,3 --"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 0,3
echo "-- load --"
uptime | awk -F 'load average: ' '{print $2}'
echo "-- minisweagent containers --"
docker ps --filter "name=minisweagent" --format '{{.Names}} {{.Status}}' | head -10
docker ps --filter "name=minisweagent" -q | wc -l | xargs echo "count:"
echo "-- preds progress --"
python3 - <<'PY'
import json, os, collections, yaml, glob
OUT='/data/jiahao/code/mini-swe-agent/experiments/qwen3_baseline_verified'
preds=os.path.join(OUT,'preds.json')
if os.path.exists(preds):
    d=json.load(open(preds))
    print(f'completed: {len(d)}/500')
    empty=sum(1 for v in d.values() if not (v.get('model_patch') or '').strip())
    print(f'empty_patches: {empty}/{len(d)}')
else:
    print('preds.json: missing')
ys=sorted(glob.glob(os.path.join(OUT,'exit_statuses_*.yaml')))
if ys:
    y=yaml.safe_load(open(ys[-1]))
    by=y.get('instances_by_exit_status',{})
    print('exit statuses:', {k: len(v) for k,v in by.items()})
PY
echo "-- last 6 log lines --"
tail -6 "$LOG" 2>/dev/null | sed 's/^/  /'
echo "==="
