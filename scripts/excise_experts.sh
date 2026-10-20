#!/usr/bin/env bash
# Physically remove the added expert bank from an expanded checkpoint (F -> A).
#
# Usage: bash scripts/excise_experts.sh MODEL OUT [--random-original-experts-seed N]
#
#   MODEL  expanded checkpoint carrying expert_expansion_manifest.json
#          (F0, F1, F2 or F3)
#   OUT    new directory for the standard checkpoint (HF checkpoint + run.json);
#          F0 gives A0, F1 gives A1, F3 gives A3
#
#   --random-original-experts-seed N
#          selectivity control: instead of the added bank, remove the same
#          number of original experts per layer, sampled with seed N; the added
#          bank is kept. The removed ids are recorded in run.json.
#
# Runs on CPU.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -eq 2 || $# -eq 4 ]] || { rex_usage >&2; exit 2; }
MODEL="$(realpath "$1")" OUT="$(realpath -m "$2")"
OPTIONS=()
if [[ $# -eq 4 ]]; then
    [[ "$3" == --random-original-experts-seed && "$4" =~ ^[0-9]+$ ]] || { rex_usage >&2; exit 2; }
    OPTIONS=(--random-original-experts-seed "$4")
fi
rex_require_dir "$MODEL"; rex_require_file "$MODEL/expert_expansion_manifest.json"
rex_require_new_output "$OUT"

rex_activate
python -m surgery.excise --model "$MODEL" --output "$OUT" "${OPTIONS[@]}"
