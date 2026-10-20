#!/usr/bin/env bash
# Expand a base checkpoint into F0 by cloning the most target-specific experts (B -> F0).
#
# Usage: bash scripts/expand_experts.sh CONFIG BASE TARGET_CSV CONTROL_CSV OUT
#
#   CONFIG       expansion recipe, e.g. configs/intern/wmdp-bio/expand.yaml
#   BASE         standard input checkpoint B
#   TARGET_CSV   dispatch_freq.csv of BASE on the forget corpus
#   CONTROL_CSV  dispatch_freq.csv of BASE on the retain corpus
#                (both from scripts/collect_routing_stats.sh)
#   OUT          new directory for F0 (HF checkpoint, expert_expansion_manifest.json
#                and run.json)
#
# In every sparse layer, experts are ranked by
# target_freq - control_weight * control_freq; the top num_new_experts are
# cloned (with epsilon Gaussian noise) together with their router rows and
# appended as the removable bank, and the source experts are scaled by
# source_expert_scale. Runs on CPU.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -eq 5 ]] || { rex_usage >&2; exit 2; }
CONFIG="$(realpath "$1")" BASE="$(realpath "$2")" TARGET_CSV="$(realpath "$3")"
CONTROL_CSV="$(realpath "$4")" OUT="$(realpath -m "$5")"
rex_require_file "$CONFIG"; rex_require_dir "$BASE"
rex_require_file "$TARGET_CSV"; rex_require_file "$CONTROL_CSV"; rex_require_new_output "$OUT"

rex_activate
python -m surgery.expand --config "$CONFIG" --base "$BASE" \
    --target-frequency-file "$TARGET_CSV" --control-frequency-file "$CONTROL_CSV" --output "$OUT"
