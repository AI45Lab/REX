#!/usr/bin/env bash
# Train an expanded checkpoint with REX (F0 -> F1).
#
# Usage: bash scripts/train_rex.sh CONFIG F0 OUT
#
#   CONFIG  REX recipe, e.g. configs/intern/wmdp-bio/rex.yaml
#   F0      expanded input checkpoint (from scripts/expand_experts.sh)
#   OUT     new directory for F1 (HF checkpoint + run.json)
#
# Runs on one node with 8 GPUs. The tokenizer is read from F0.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -eq 3 ]] || { rex_usage >&2; exit 2; }
CONFIG="$(realpath "$1")" F0="$(realpath "$2")" OUT="$(realpath -m "$3")"
rex_require_file "$CONFIG"; rex_require_dir "$F0"; rex_require_new_output "$OUT"

rex_activate
rex_require_eight_gpus
export WANDB_DIR="${WANDB_DIR:-$OUT}"
rex_accelerate methods.rex.trainer --config "$CONFIG" --init "$F0" --output "$OUT"
