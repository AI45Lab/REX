#!/usr/bin/env bash
# Train RMU on a standard checkpoint (A1 -> A2, or B -> matched R).
#
# Usage: bash scripts/train_rmu.sh CONFIG PARENT OUT [--save-forget-token-milestones N ...]
#
#   CONFIG  RMU recipe, e.g. configs/intern/wmdp-bio/rmu.yaml; A2 and its matched
#           R use the same recipe and differ only in PARENT
#   PARENT  standard input checkpoint: A1 (from scripts/excise_experts.sh) gives
#           A2, the base model B gives R; it is also the retain reference and
#           the tokenizer source
#   OUT     new directory for the final checkpoint (HF checkpoint + run.json)
#
#   --save-forget-token-milestones N ...
#           also save checkpoints once N forget tokens have been seen, under
#           OUT/milestones/forget-tokens-<N>/ (each with its own run.json)
#
# Runs on one node with 8 GPUs.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -ge 3 ]] || { rex_usage >&2; exit 2; }
CONFIG="$(realpath "$1")" PARENT="$(realpath "$2")" OUT="$(realpath -m "$3")"
shift 3
MILESTONES=()
if [[ $# -gt 0 ]]; then
    [[ "$1" == --save-forget-token-milestones && $# -ge 2 ]] || { rex_usage >&2; exit 2; }
    shift
    for milestone in "$@"; do
        [[ "$milestone" =~ ^[1-9][0-9]*$ ]] || rex_die "Milestone must be a positive integer: $milestone"
    done
    MILESTONES=(--save-forget-token-milestones "$@")
fi
rex_require_file "$CONFIG"; rex_require_dir "$PARENT"; rex_require_new_output "$OUT"

rex_activate
rex_require_eight_gpus
export WANDB_DIR="${WANDB_DIR:-$OUT}"
rex_accelerate methods.rmu.trainer --config "$CONFIG" --parent "$PARENT" --output "$OUT" \
    "${MILESTONES[@]}"
