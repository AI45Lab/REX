#!/usr/bin/env bash
# Rebuild an expanded checkpoint from A2 and the removed bank of its F1 (F1 + A2 -> F2).
#
# Usage: bash scripts/recombine_experts.sh F1 A2 OUT
#
#   F1   expanded checkpoint whose excision gave A2's parent A1
#   A2   standard checkpoint from scripts/train_rmu.sh on that A1
#   OUT  new directory for F2 (HF checkpoint, F1's expert_expansion_manifest.json
#        and run.json)
#
# F2 takes every retained parameter from A2 and only the added experts and
# their router rows from F1. F1 is not checked against A2's lineage; use the
# F1 that produced A1 (pipelines/f1_to_f2.sh does). Runs on CPU.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -eq 3 ]] || { rex_usage >&2; exit 2; }
F1="$(realpath "$1")" A2="$(realpath "$2")" OUT="$(realpath -m "$3")"
rex_require_dir "$F1"; rex_require_file "$F1/expert_expansion_manifest.json"
rex_require_dir "$A2"; rex_require_new_output "$OUT"

rex_activate
python -m surgery.recombine --f1 "$F1" --a2 "$A2" --output "$OUT"
