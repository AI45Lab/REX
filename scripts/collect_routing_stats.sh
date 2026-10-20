#!/usr/bin/env bash
# Count router dispatches per layer and expert of a MoE model over a corpus.
#
# Usage: bash scripts/collect_routing_stats.sh MODEL CORPUS OUT_DIR [OPTIONS]
#
#   MODEL    MoE checkpoint, normally the base model B
#   CORPUS   training corpus: a .jsonl/.json/.parquet/.csv/.txt file, a
#            directory of files of one type, or a save_to_disk dataset
#   OUT_DIR  new directory; receives dispatch_freq.csv (input of
#            scripts/expand_experts.sh), counts/ (raw per-GPU counts and logs)
#            and run.json (settings, documents, valid tokens per layer)
#
# Options:
#   --text-column C                      text column (default: text)
#   --long-document-policy chunk|truncate
#                                        chunk (default) splits long documents
#                                        into --max-length windows so every token
#                                        is counted; truncate keeps only the
#                                        first --max-length tokens. Use the same
#                                        policy for target and control.
#   --max-length N                       tokens per model sequence (default: 2048)
#   --batch-size N                       documents per batch (default: 4)
#   --max-samples N                      use only the first N corpus rows
#
# Runs one data-parallel worker per visible GPU, each holding a full model
# replica in bfloat16 and a contiguous corpus shard; their counts are summed
# before frequencies are computed.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -ge 3 ]] || { rex_usage >&2; exit 2; }
MODEL="$(realpath "$1")" CORPUS="$(realpath "$2")" OUT="$(realpath -m "$3")"
shift 3
OPTIONS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --text-column)
            [[ $# -ge 2 ]] || rex_die "$1 requires a value."
            OPTIONS+=("$1" "$2"); shift 2 ;;
        --long-document-policy)
            [[ $# -ge 2 && ( "$2" == chunk || "$2" == truncate ) ]] || rex_die "$1 must be chunk or truncate."
            OPTIONS+=("$1" "$2"); shift 2 ;;
        --max-length|--batch-size|--max-samples)
            [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || rex_die "$1 requires a positive integer."
            OPTIONS+=("$1" "$2"); shift 2 ;;
        *)
            rex_usage >&2; exit 2 ;;
    esac
done
rex_require_dir "$MODEL"; [[ -e "$CORPUS" ]] || rex_die "Corpus does not exist: $CORPUS"
rex_require_new_output "$OUT"

# Loading a large corpus directory materializes an Arrow cache of comparable
# size; keep it on the node's temporary disk.
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/rex-hf/datasets"
rex_activate
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=20 MKL_NUM_THREADS=20
python -m surgery.routing_stats --model "$MODEL" --corpus "$CORPUS" --output "$OUT" "${OPTIONS[@]}"
