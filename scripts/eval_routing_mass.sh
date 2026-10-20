#!/usr/bin/env bash
# Measure passive routing mass on the removable bank for F0 and F1.
#
# Usage: bash scripts/eval_routing_mass.sh F0 F1 FORGET RETAIN
#
#   F0      expanded checkpoint before REX (from scripts/expand_experts.sh);
#           its tokenizer is used for sampling
#   F1      the same F0 after REX (from scripts/train_rex.sh); the result is
#           written to the new directory F1/diagnostics/routing_mass/
#   FORGET  directory of training forget-corpus *.jsonl files ("text" or "content")
#   RETAIN  directory of training retain-corpus *.jsonl files
#
# Output: sampling/{inputs.jsonl,metadata.json}, per-token masses F0/rank-*.npz
# and F1/rank-*.npz, routing_mass.json (per-layer means and CDFs per corpus and
# checkpoint) and run.json.
#
# Up to 512 non-overlapping 2048-token windows per corpus (reservoir sampling,
# seed 42; all forget documents form the pool, retain uses the first 4096
# documents in sorted file order; plain text, no BOS, EOS appended), shared by
# F0 and F1. Each visible GPU runs one BF16/SDPA model copy on its share of the
# windows (batch 4); weights are never modified.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/env.sh"
rex_handle_help "$@"
[[ $# -eq 4 ]] || { rex_usage >&2; exit 2; }
F0="$(realpath "$1")" F1="$(realpath "$2")" FORGET="$(realpath "$3")" RETAIN="$(realpath "$4")"
rex_require_dir "$F0"; rex_require_dir "$F1"; rex_require_dir "$FORGET"; rex_require_dir "$RETAIN"
OUT="$F1/diagnostics/routing_mass"
rex_require_new_output "$OUT"

rex_activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
GPUS="$(rex_gpu_count)"
[[ "$GPUS" -gt 0 ]] || rex_die "No visible CUDA devices."
torchrun --standalone --nproc_per_node "$GPUS" -m eval.routing_mass \
    --f0 "$F0" --f1 "$F1" --forget "$FORGET" --retain "$RETAIN" --output "$OUT"
