#!/usr/bin/env bash
# =============================================================================
# 自动跑完 32 个实验（跳过已完成的）
# - 4 models x 4 conditions x 2 datasets = 32 runs
# - 已跳过: GSM8K + Qwen2.5-1.5B + direct (results/gsm8k/qwen2.5-1.jsonl)
#
# 用法:
#   ./run_all_experiments.sh           # 实际运行
#   ./run_all_experiments.sh --dry-run # 仅打印会跑/会跳过的实验
# =============================================================================

set -e
cd "$(dirname "$0")"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

MAX_SAMPLES=100
SEED=42
TEMPERATURE=0
DEVICE=auto
DTYPE=bfloat16

# 模型列表（按 workflow 推荐顺序）
MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

CONDITIONS=(direct short medium long)

# GSM8K model_tag: Qwen/Qwen2.5-1.5B-Instruct -> qwen2.5-1.5b-instruct
model_tag_gsm8k() {
  echo "$1" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr ' ' '_'
}

# DROP model_tag: 额外把 . 换成 _ (qwen2.5 -> qwen2_5)
model_tag_drop() {
  model_tag_gsm8k "$1" | tr '.' '_'
}

# 检查是否应跳过该实验
should_skip() {
  local dataset="$1"
  local model="$2"
  local cond="$3"
  local tag

  if [[ "$dataset" == "gsm8k" ]]; then
    tag=$(model_tag_gsm8k "$model")
    local out="results/gsm8k/${tag}_${cond}_test_n${MAX_SAMPLES}.jsonl"
    # 已完成的文件可能是 qwen2.5-1.jsonl（旧命名）
    local alt="results/gsm8k/qwen2.5-1.jsonl"
    if [[ -f "$out" ]]; then
      return 0  # skip
    fi
    if [[ "$model" == "Qwen/Qwen2.5-1.5B-Instruct" && "$cond" == "direct" && -f "$alt" ]]; then
      return 0  # skip (已完成)
    fi
  else
    tag=$(model_tag_drop "$model")
    local out="results/drop/${tag}_${cond}_validation_n${MAX_SAMPLES}.jsonl"
    if [[ -f "$out" ]]; then
      return 0  # skip
    fi
  fi
  return 1  # don't skip
}

run_count=0
skip_count=0

echo "=============================================="
echo "  Person 3 实验批量运行 (max_samples=${MAX_SAMPLES})"
echo "=============================================="

mkdir -p results/gsm8k results/drop

# ---------- GSM8K (16 runs) ----------
echo ""
echo ">>> GSM8K (4 models x 4 conditions)"
for model in "${MODELS[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    if should_skip "gsm8k" "$model" "$cond"; then
      echo "[SKIP] gsm8k $model $cond"
      ((skip_count++)) || true
      continue
    fi
    echo "[RUN] gsm8k $model $cond"
    if $DRY_RUN; then
      echo "  python gsm8k/eval_gsm8k_step.py --model $model --condition $cond ..."
      ((run_count++)) || true
      continue
    fi
    python gsm8k/eval_gsm8k_step.py \
      --model "$model" \
      --condition "$cond" \
      --split test \
      --max-samples "$MAX_SAMPLES" \
      --seed "$SEED" \
      --temperature "$TEMPERATURE" \
      --device "$DEVICE" \
      --dtype "$DTYPE"
    ((run_count++)) || true
  done
done

# ---------- DROP (16 runs) ----------
echo ""
echo ">>> DROP (4 models x 4 conditions)"
for model in "${MODELS[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    if should_skip "drop" "$model" "$cond"; then
      echo "[SKIP] drop $model $cond"
      ((skip_count++)) || true
      continue
    fi
    echo "[RUN] drop $model $cond"
    if $DRY_RUN; then
      echo "  python drop/eval_drop.py --model $model --condition $cond ..."
      ((run_count++)) || true
      continue
    fi
    python drop/eval_drop.py \
      --model "$model" \
      --condition "$cond" \
      --split validation \
      --max-samples "$MAX_SAMPLES" \
      --seed "$SEED" \
      --temperature "$TEMPERATURE" \
      --device "$DEVICE" \
      --dtype "$DTYPE"
    ((run_count++)) || true
  done
done

echo ""
echo "=============================================="
echo "  完成: 新跑 ${run_count} 个, 跳过 ${skip_count} 个"
echo "=============================================="
