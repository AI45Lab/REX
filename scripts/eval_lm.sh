#!/usr/bin/env bash
# Evaluate a checkpoint, in its real form, under an evaluation protocol.
#
# Usage: bash scripts/eval_lm.sh MODEL NODE_DIR --protocol P [--tasks SPEC1,SPEC2] [--processor-from DIR]
#
#   MODEL       HF checkpoint, evaluated as is (an expanded checkpoint is
#               evaluated with its removable bank; to evaluate the excised
#               form, run scripts/excise_experts.sh first and evaluate that)
#   NODE_DIR    node directory the results belong to; may be MODEL itself, or
#               a record-only node dir when MODEL is a temporary checkpoint
#               (e.g. an excised form); created if missing
#   --protocol  protocol name from eval/protocols.yaml (lm-eval-vllm-v1,
#               ifeval-chat-v1, ...): backend, model args, seeds, chat template
#   --tasks     comma-separated task spec names of the protocol (default: all,
#               e.g. wmdp_bio,wmdp_cyber,mmlu,arc_challenge-25shot,
#               hellaswag-10shot,gsm8k-5shot for lm-eval-vllm-v1)
#   --processor-from DIR
#               take multimodal processor configs MODEL lacks (some external
#               checkpoints ship without them) from DIR, normally the base
#               model B; MODEL itself is not modified, the added files are
#               recorded in each spec's run.json
#
# Specs that already have NODE_DIR/eval/P/<spec>/ are skipped; the others run
# in one lm-eval invocation and are split per spec into NODE_DIR/eval/P/<spec>/:
# results.json (lm-eval's results file restricted to the spec's tasks and
# groups), samples_<task>_<date>.jsonl and run.json (actual library versions and
# whether they match the protocol). Each spec directory is moved into place
# only when complete.
#
# Uses the vLLM venv (.venv-eval) with one data-parallel vLLM instance per
# visible GPU; datasets come from the offline HF cache. IFEval scoring needs
# the NLTK punkt_tab resource offline (set NLTK_DATA).
#
# vLLM occasionally fails its engine startup or stalls a replica. Set
# REX_EVAL_ATTEMPTS (default 1) to retry the whole evaluation and
# REX_EVAL_TIMEOUT (e.g. 3h; default none) to bound each attempt; specs
# completed by an earlier attempt are skipped.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -ge 2 ]] || { rex_usage >&2; exit 2; }
MODEL="$(realpath "$1")" NODE_DIR="$(realpath -m "$2")"
shift 2
rex_require_dir "$MODEL"

PROTOCOL="" OPTIONS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --protocol)
            [[ $# -ge 2 && -n "$2" ]] || rex_die "$1 requires a value."
            PROTOCOL="$2"; shift 2 ;;
        --tasks)
            [[ $# -ge 2 && -n "$2" ]] || rex_die "$1 requires a value."
            OPTIONS+=("$1" "$2"); shift 2 ;;
        --processor-from)
            [[ $# -ge 2 && -n "$2" ]] || rex_die "$1 requires a value."
            rex_require_dir "$2"
            OPTIONS+=("$1" "$(realpath "$2")"); shift 2 ;;
        *)
            rex_usage >&2; rex_die "Unknown option: $1" ;;
    esac
done
[[ -n "$PROTOCOL" ]] || { rex_usage >&2; rex_die "--protocol is required."; }

rex_activate_eval
GPUS="$(rex_gpu_count)"
[[ "$GPUS" -gt 0 ]] || rex_die "No visible CUDA devices."
ATTEMPTS="${REX_EVAL_ATTEMPTS:-1}" BOUND=()
[[ -n "${REX_EVAL_TIMEOUT:-}" ]] && BOUND=(timeout "$REX_EVAL_TIMEOUT")
for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
    "${BOUND[@]}" python -m eval.vllm_eval --model "$MODEL" --node-dir "$NODE_DIR" --protocol "$PROTOCOL" \
        --data-parallel-size "$GPUS" "${OPTIONS[@]}" && exit 0
    echo "eval_lm: attempt $attempt/$ATTEMPTS failed" >&2
    # A failed or timed-out attempt can leave engine and Ray processes holding the GPUs.
    pkill -u "$(id -u)" -f "eval.vllm_eval" || true
    pkill -u "$(id -u)" -f raylet || true
    sleep 20
done
exit 1
