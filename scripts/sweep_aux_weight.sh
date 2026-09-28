#!/usr/bin/env bash
# aux 权重扫描：每个配置完整训练一次并评估，产出 trade-off 曲线。
#
# 关注两个指标：
#   路由分化  末层 Code/Math 理科核权重 与 Arts 的差（越大越"分工明确"）
#   验证集PPL 双大核相对 baseline 的变化（越负越好）
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

WEIGHTS=${1:-"0.01 0.02 0.05"}
mkdir -p reports/sweep

for w in $WEIGHTS; do
  tag="aux${w}"
  ck="checkpoints/sweep_${tag}.pt"
  echo "=================== aux_weight=${w} ==================="
  python train_dual_big_resurrect.py --out "$ck" --log-every 1000 2>&1 \
    | grep -E "step .*loss" | tail -2
  python eval_ppl.py --weights "$ck" --split val --samples-per-domain 100 \
    --batch-size 8 --out "reports/sweep/${tag}.json" --skip-infer 2>&1 \
    | grep -E "^  (Code|Math|Arts) "
  python -m dbl.diag_routing --weights "run=$ck" --data val=dual_contrast_val.jsonl \
    --n 60 --batch-size 8 --out "reports/sweep/${tag}_routing.json" 2>&1 \
    | grep -E "^  (27|16) " || true
done
