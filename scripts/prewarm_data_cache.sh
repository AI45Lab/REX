#!/usr/bin/env bash
# Fill the XTuner tokenization and packing-metadata cache of a recipe.
#
# Usage: bash scripts/prewarm_data_cache.sh CONFIG MODEL
#
#   CONFIG  REX or RMU recipe, e.g. configs/intern/wmdp-bio/rex.yaml
#   MODEL   checkpoint whose tokenizer the training run will read (B or F0)
#
# Needs no GPU. Writes only into the recipe's xtuner.cache_dir (relative to the
# repository root); training with prewarm_metadata_cache: true does the same
# step itself, so running this first only moves the cost off the GPU node.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -eq 2 ]] || { rex_usage >&2; exit 2; }
CONFIG="$(realpath "$1")" MODEL="$(realpath "$2")"
rex_require_file "$CONFIG"; rex_require_dir "$MODEL"

rex_activate
python -m dataset.xtuner.loader "$CONFIG" "$MODEL"
