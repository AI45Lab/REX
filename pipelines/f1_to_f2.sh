#!/usr/bin/env bash
# Build F2 from F1 inside a run directory: excise, train RMU, recombine,
# evaluating every node.
#
# Usage: bash pipelines/f1_to_f2.sh RMU_CONFIG RUN_DIR [--protocols P1,P2]
#
#   RMU_CONFIG  RMU recipe, e.g. configs/intern/wmdp-bio/rmu.yaml
#   RUN_DIR     run directory holding RUN_DIR/F1 (from scripts/train_rex.sh);
#               RUN_DIR/A1, RUN_DIR/A2 and RUN_DIR/F2 must not exist
#   --protocols comma-separated evaluation protocols of eval/protocols.yaml
#               run on every node (default: lm-eval-vllm-v1)
#
# Steps, in order:
#   1. scripts/eval_lm.sh on F1 (full form; tasks already present are skipped)
#   2. scripts/excise_experts.sh F1 -> RUN_DIR/A1; scripts/eval_lm.sh on A1
#   3. scripts/train_rmu.sh RMU_CONFIG A1 -> RUN_DIR/A2 (A1 is A2's parent and
#      retain reference); its rmu_metrics.jsonl and wandb/ stay in A2
#   4. scripts/drop_weights.sh A1
#   5. scripts/eval_lm.sh on A2
#   6. scripts/recombine_experts.sh F1 A2 -> RUN_DIR/F2, using the same F1
#      that produced A1
#   7. scripts/drop_weights.sh A2; scripts/eval_lm.sh on F2 (full form; F2's
#      excised form is A2, whose evaluation is A2's)
# A1 keeps its weights until A2 is trained and A2 until F2 is written; each
# loses them only after its own evaluation. A1 and A2 end record-only
# (run.json, eval/, training records); F2's run.json embeds both records. On
# failure the pipeline stops and every node is left as it is.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/lib/env.sh"
rex_handle_help "$@"
[[ $# -ge 2 ]] || { rex_usage >&2; exit 2; }
CONFIG="$(realpath "$1")" RUN_DIR="$(realpath "$2")"
shift 2
PROTOCOLS=lm-eval-vllm-v1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --protocols)
            [[ $# -ge 2 && -n "$2" ]] || rex_die "$1 requires a value."
            PROTOCOLS="$2"; shift 2 ;;
        *)
            rex_usage >&2; rex_die "Unknown option: $1" ;;
    esac
done
IFS=, read -r -a PROTOCOL_LIST <<< "$PROTOCOLS"
F1="$RUN_DIR/F1" A1="$RUN_DIR/A1" A2="$RUN_DIR/A2" F2="$RUN_DIR/F2"
rex_require_file "$CONFIG"; rex_require_dir "$F1"; rex_require_file "$F1/expert_expansion_manifest.json"
rex_require_new_output "$A1"; rex_require_new_output "$A2"; rex_require_new_output "$F2"

SCRIPTS="$REX_REPO_DIR/scripts"
evaluate() {
    local protocol
    for protocol in "${PROTOCOL_LIST[@]}"; do
        bash "$SCRIPTS/eval_lm.sh" "$1" "$1" --protocol "$protocol"
    done
}

evaluate "$F1"
bash "$SCRIPTS/excise_experts.sh" "$F1" "$A1"
evaluate "$A1"
WANDB_DIR="$A2" bash "$SCRIPTS/train_rmu.sh" "$CONFIG" "$A1" "$A2"
bash "$SCRIPTS/drop_weights.sh" "$A1"
evaluate "$A2"
bash "$SCRIPTS/recombine_experts.sh" "$F1" "$A2" "$F2"
bash "$SCRIPTS/drop_weights.sh" "$A2"
evaluate "$F2"
echo "F2 written to $F2; A1 and A2 evaluated and kept record-only"
