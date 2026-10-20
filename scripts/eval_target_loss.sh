#!/usr/bin/env bash
# Measure a checkpoint's mean token log-probability on the full target corpus.
#
# Usage: bash scripts/eval_target_loss.sh MODEL CONFIG NODE_DIR
#
#   MODEL     HF checkpoint, evaluated as is; its tokenizer is used
#   CONFIG    training recipe whose role: harmful datasets form the target
#             corpus, tokenized and chunked as in training, e.g.
#             configs/intern/wmdp-bio/rmu.yaml
#   NODE_DIR  node directory the result belongs to (may be MODEL itself);
#             writes the new directory
#             NODE_DIR/diagnostics/target_loss/<CONFIG file stem>/ with
#             target_loss.json (corpus totals), documents.csv (per-document
#             mean token NLL/log-probability) and run.json
#
# One process, the model spread over all visible GPUs (device_map=auto), BF16,
# batch size 1.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -eq 3 ]] || { rex_usage >&2; exit 2; }
MODEL="$(realpath "$1")" CONFIG="$(realpath "$2")" NODE_DIR="$(realpath -m "$3")"
rex_require_dir "$MODEL"; rex_require_file "$CONFIG"
OUT="$NODE_DIR/diagnostics/target_loss/$(basename -- "$CONFIG" .yaml)"
rex_require_new_output "$OUT"

rex_activate
python -m eval.target_loss --model "$MODEL" --config "$CONFIG" --output "$OUT"
