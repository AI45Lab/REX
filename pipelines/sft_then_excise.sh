#!/usr/bin/env bash
# Fine-tune a node with S_std and evaluate the result; for an expanded start,
# also excise the removable bank and evaluate the excised node.
#
# Usage: bash pipelines/sft_then_excise.sh SFT_CONFIG START_NODE RUN_DIR NAME [--protocols P1,P2]
#
#   SFT_CONFIG  SFT recipe, e.g. configs/sft/s_std-bio.yaml
#   START_NODE  node with weights to fine-tune: F2 (gives F3), A2 (A3-direct)
#               or R (R-SFT)
#   RUN_DIR     run directory (created if missing) receiving the new nodes
#   NAME        name of the fine-tuned node, e.g. F3-250M, A3-direct-250M,
#               R-SFT-250M. An expanded START_NODE (one carrying
#               expert_expansion_manifest.json) needs a NAME starting with F,
#               and its excised node is NAME with that F replaced by A
#               (F3-250M -> A3-250M); a standard START_NODE needs a NAME not
#               starting with F and gets no excised node. The new nodes must
#               not exist yet.
#   --protocols comma-separated evaluation protocols of eval/protocols.yaml
#               run on every node (default: lm-eval-vllm-v1)
#
# Steps, in order:
#   1. scripts/train_sft.sh SFT_CONFIG START_NODE -> RUN_DIR/NAME (endpoint
#      weights at the node root, resident milestone eval in NAME/online_eval/)
#   2. scripts/eval_lm.sh on RUN_DIR/NAME (its real form; full for F3)
#   3. expanded start only: scripts/excise_experts.sh RUN_DIR/NAME ->
#      RUN_DIR/<excised name>; scripts/eval_lm.sh on it; scripts/drop_weights.sh
#      on it (the excised node keeps only records and eval)
# On failure the pipeline stops and every node is left as it is.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/lib/env.sh"
rex_handle_help "$@"
[[ $# -ge 4 ]] || { rex_usage >&2; exit 2; }
CONFIG="$(realpath "$1")" START="$(realpath "$2")" RUN_DIR="$(realpath -m "$3")" NAME="$4"
shift 4
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
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || rex_die "NAME must be one directory name: $NAME"
rex_require_file "$CONFIG"; rex_require_dir "$START"
NODE="$RUN_DIR/$NAME" EXCISED=""
if [[ -f "$START/expert_expansion_manifest.json" ]]; then
    [[ "$NAME" == F* ]] || rex_die "An expanded start node needs a NAME starting with F: $NAME"
    EXCISED="$RUN_DIR/A${NAME:1}"
    rex_require_new_output "$EXCISED"
else
    [[ "$NAME" != F* ]] || rex_die "A standard start node needs a NAME not starting with F: $NAME"
fi
rex_require_new_output "$NODE"
mkdir -p "$RUN_DIR"

SCRIPTS="$REX_REPO_DIR/scripts"
evaluate() {
    local protocol
    for protocol in "${PROTOCOL_LIST[@]}"; do
        bash "$SCRIPTS/eval_lm.sh" "$1" "$1" --protocol "$protocol"
    done
}

bash "$SCRIPTS/train_sft.sh" "$CONFIG" "$START" "$NODE"
evaluate "$NODE"
if [[ -n "$EXCISED" ]]; then
    bash "$SCRIPTS/excise_experts.sh" "$NODE" "$EXCISED"
    evaluate "$EXCISED"
    bash "$SCRIPTS/drop_weights.sh" "$EXCISED"
    echo "$NAME written to $NODE; $(basename -- "$EXCISED") evaluated and kept record-only"
else
    echo "$NAME written to $NODE"
fi
