#!/usr/bin/env bash
# Build F0 and its excised form A0 from the base model, evaluating both.
#
# Usage: bash pipelines/build_f0.sh EXPAND_CONFIG B TARGET_CORPUS CONTROL_CORPUS RUN_DIR --control-max-samples N|all [--protocols P1,P2]
#
#   EXPAND_CONFIG   expansion recipe, e.g. configs/intern/wmdp-bio/expand.yaml
#   B               standard base checkpoint
#   TARGET_CORPUS   user-provided target corpus (JSONL with a content column)
#   CONTROL_CORPUS  user-provided control corpus
#   --control-max-samples  required: positive row limit, or all
#   --target-max-samples   optional: positive row limit, or all (default)
#   RUN_DIR         run directory (created if missing); RUN_DIR/stats/target,
#                   RUN_DIR/stats/control, RUN_DIR/F0 and RUN_DIR/A0 must not exist
#   --protocols     comma-separated evaluation protocols of eval/protocols.yaml
#                   run on every node (default: lm-eval-vllm-v1)
#
# Steps, in order:
#   1. scripts/collect_routing_stats.sh B on TARGET_CORPUS (every row)
#      -> RUN_DIR/stats/target, and on CONTROL_CORPUS (explicit row limit)
#      -> RUN_DIR/stats/control; both with text column "content", chunk / max
#      length 2048 / batch size 4, the routing_stats settings recorded in
#      this pipeline; use collect_routing_stats.sh directly for other settings
#   2. scripts/expand_experts.sh -> RUN_DIR/F0; scripts/eval_lm.sh on F0
#   3. scripts/excise_experts.sh F0 -> RUN_DIR/A0; scripts/eval_lm.sh on A0;
#      scripts/drop_weights.sh A0 (A0 = E(F0) keeps only records and eval)
# Every node's evaluation lands in <node>/eval/<protocol>/<task>/. On failure
# the pipeline stops and every node is left as it is.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/lib/env.sh"
rex_handle_help "$@"
[[ $# -ge 5 ]] || { rex_usage >&2; exit 2; }
CONFIG="$(realpath "$1")" B="$(realpath "$2")" TARGET_CORPUS="$(realpath "$3")"
CONTROL_CORPUS="$(realpath "$4")" RUN_DIR="$(realpath -m "$5")"
shift 5
PROTOCOLS=lm-eval-vllm-v1
CONTROL_MAX_SAMPLES="" TARGET_MAX_SAMPLES=all
while [[ $# -gt 0 ]]; do
    case "$1" in
        --protocols)
            [[ $# -ge 2 && -n "$2" ]] || rex_die "$1 requires a value."
            PROTOCOLS="$2"; shift 2 ;;
        --control-max-samples|--target-max-samples)
            [[ $# -ge 2 && ( "$2" == all || "$2" =~ ^[1-9][0-9]*$ ) ]] || rex_die "$1 requires a positive integer or all."
            if [[ "$1" == --control-max-samples ]]; then CONTROL_MAX_SAMPLES="$2"; else TARGET_MAX_SAMPLES="$2"; fi
            shift 2 ;;
        *)
            rex_usage >&2; rex_die "Unknown option: $1" ;;
    esac
done
[[ -n "$CONTROL_MAX_SAMPLES" ]] || rex_die "Specify --control-max-samples N|all; no data sampling recipe is assumed."
IFS=, read -r -a PROTOCOL_LIST <<< "$PROTOCOLS"
STATS="$RUN_DIR/stats" F0="$RUN_DIR/F0" A0="$RUN_DIR/A0"
rex_require_file "$CONFIG"; rex_require_dir "$B"
rex_require_new_output "$STATS/target"; rex_require_new_output "$STATS/control"
rex_require_new_output "$F0"; rex_require_new_output "$A0"
mkdir -p "$STATS"

SCRIPTS="$REX_REPO_DIR/scripts"
evaluate() {
    local protocol
    for protocol in "${PROTOCOL_LIST[@]}"; do
        bash "$SCRIPTS/eval_lm.sh" "$1" "$1" --protocol "$protocol"
    done
}

COLLECTION=(--text-column content --long-document-policy chunk --max-length 2048 --batch-size 4)
TARGET_LIMIT=() CONTROL_LIMIT=()
[[ "$TARGET_MAX_SAMPLES" == all ]] || TARGET_LIMIT=(--max-samples "$TARGET_MAX_SAMPLES")
[[ "$CONTROL_MAX_SAMPLES" == all ]] || CONTROL_LIMIT=(--max-samples "$CONTROL_MAX_SAMPLES")
bash "$SCRIPTS/collect_routing_stats.sh" "$B" "$TARGET_CORPUS" "$STATS/target" "${COLLECTION[@]}" "${TARGET_LIMIT[@]}"
bash "$SCRIPTS/collect_routing_stats.sh" "$B" "$CONTROL_CORPUS" "$STATS/control" "${COLLECTION[@]}" "${CONTROL_LIMIT[@]}"

bash "$SCRIPTS/expand_experts.sh" "$CONFIG" "$B" \
    "$STATS/target/dispatch_freq.csv" "$STATS/control/dispatch_freq.csv" "$F0"
evaluate "$F0"

bash "$SCRIPTS/excise_experts.sh" "$F0" "$A0"
evaluate "$A0"
bash "$SCRIPTS/drop_weights.sh" "$A0"
echo "F0 written to $F0; A0 evaluated and kept record-only at $A0"
