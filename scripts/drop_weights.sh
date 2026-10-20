#!/usr/bin/env bash
# Make a node record-only: delete its weight files, keep everything else.
#
# Usage: bash scripts/drop_weights.sh NODE_DIR
#
#   NODE_DIR  node directory carrying run.json (e.g. <run_dir>/A1); directories
#             without run.json are refused, as are the repository root and
#             anything under the repository's checkpoints/, data/ or internal/
#
# Deletes every *.safetensors file and model.safetensors.index.json below
# NODE_DIR, save milestones included; run.json, config and tokenizer files,
# eval/, online_eval/ and training records stay. Run it only after every
# evaluation of the node and every step that reads its weights has finished.
# Runs on CPU.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -eq 1 ]] || { rex_usage >&2; exit 2; }
NODE_DIR="$(realpath "$1")"
rex_require_dir "$NODE_DIR"; rex_require_file "$NODE_DIR/run.json"

rex_activate
python -m common.drop_weights "$NODE_DIR"
