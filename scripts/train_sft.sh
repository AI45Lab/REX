#!/usr/bin/env bash
# Standard full-parameter SFT S_std with ms-swift Megatron (F2 -> F3,
# A2 -> A3-direct, R -> R-SFT).
#
# Usage: bash scripts/train_sft.sh CONFIG MODEL OUT
#
#   CONFIG  SFT recipe, e.g. configs/sft/s_std-bio.yaml
#   MODEL   HF checkpoint to fine-tune
#   OUT     new node directory (e.g. <run_dir>/F3-250M):
#             OUT/*.safetensors, config.json, ...  HF checkpoint at max_tokens (the endpoint)
#             OUT/run.json                         its record
#             OUT/online_eval/tokens-<012d>/       resident eval results (recipe eval section)
#             OUT/milestones/tokens-<012d>/        HF checkpoint + run.json per earlier save
#                                                  milestone, only if the recipe lists any
#             OUT/swift_data/train.jsonl           converted training data
#             OUT/logs/node_0.log, plus swift's args.json, logging.jsonl, runs/, images/
#
# Environment:
#   REX_SWIFT_VENV  venv used to run swift (python, torch, lm-eval)
#   SWIFT_ROOT      optional override with a clean, pinned ms-swift checkout
#
# Runs on one node with 8 GPUs. The tokenizer and chat template come from MODEL.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -eq 3 ]] || { rex_usage >&2; exit 2; }
CONFIG="$(realpath "$1")" MODEL="$(realpath "$2")" OUT="$(realpath -m "$3")"
rex_require_file "$CONFIG"; rex_require_dir "$MODEL"; rex_require_new_output "$OUT"
if [[ -n "${SWIFT_ROOT:-}" ]]; then
    [[ -f "$SWIFT_ROOT/swift/__init__.py" ]] || rex_die "ms-swift checkout not found: $SWIFT_ROOT"
fi
[[ -n "${REX_SWIFT_VENV:-}" ]] || rex_die "REX_SWIFT_VENV must point to the swift venv."
[[ -f "$REX_SWIFT_VENV/bin/activate" ]] || rex_die "Swift venv not found: $REX_SWIFT_VENV"

cd "$REX_REPO_DIR"
# shellcheck disable=SC1091
source "$REX_SWIFT_VENV/bin/activate"
rex_setup_cache
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REX_REPO_DIR}${SWIFT_ROOT:+:${SWIFT_ROOT}}"
rex_require_eight_gpus
ulimit -n 65536

mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/node_0.log") 2>&1
python -m sft.train --config "$CONFIG" --model "$MODEL" --output "$OUT"
