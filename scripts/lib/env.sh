#!/usr/bin/env bash
# Environment shared by the public entrypoints in scripts/. Sourced, not run.
#
# Every entrypoint runs from the repository root inside an already allocated
# GPU environment. Override locations with:
#   REX_VENV       training venv          (default: <repo>/.venv)
#   REX_EVAL_VENV  vLLM evaluation venv   (default: <repo>/.venv-eval)
#   XDG_CACHE_HOME HF/kernel cache root   (default: <repo>/.cache)

REX_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

rex_die() {
    echo "$*" >&2
    exit 1
}

# Print the leading comment block of the calling script as its help text.
rex_usage() {
    sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"
}

rex_handle_help() {
    for arg in "$@"; do
        if [[ "$arg" == -h || "$arg" == --help ]]; then
            rex_usage
            exit 0
        fi
    done
}

rex_activate() {
    local venv="${REX_VENV:-${REX_REPO_DIR}/.venv}"
    [[ -f "$venv/bin/activate" ]] || rex_die "Training venv not found: $venv"
    cd "$REX_REPO_DIR"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    rex_setup_cache
}

# vLLM needs a different torch than the training venv.
rex_activate_eval() {
    local venv="${REX_EVAL_VENV:-${REX_REPO_DIR}/.venv-eval}"
    [[ -f "$venv/bin/activate" ]] || rex_die "Evaluation venv not found: $venv"
    cd "$REX_REPO_DIR"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    export PYTHONNOUSERSITE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
    rex_setup_cache
}

rex_setup_cache() {
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REX_REPO_DIR}/.cache}"
    export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    local kernel_cache="${REX_KERNEL_CACHE_ROOT:-${XDG_CACHE_HOME}/rex/kernel-cache}"
    export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${kernel_cache}/hf_modules}"
    export TILELANG_CACHE_DIR="${kernel_cache}/tilelang"
    export TILELANG_TMP_DIR="${TILELANG_CACHE_DIR}/tmp"
    export TRITON_CACHE_DIR="${kernel_cache}/triton"
    export TORCH_EXTENSIONS_DIR="${kernel_cache}/torch_extensions"
    export TORCHINDUCTOR_CACHE_DIR="${kernel_cache}/torchinductor"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_MODULES_CACHE" "$TILELANG_TMP_DIR" \
        "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR"
}

rex_gpu_count() {
    python -c 'import torch; print(torch.cuda.device_count())'
}

# Training is single-node 8-GPU by protocol.
rex_require_eight_gpus() {
    local count
    count="$(rex_gpu_count)"
    [[ "$count" == 8 ]] || rex_die "This step requires one node with 8 visible GPUs, found $count."
}

# Run one Python module under accelerate on all local GPUs.
rex_accelerate() {
    local threads=20
    export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
    accelerate launch \
        --num_processes "$(rex_gpu_count)" \
        --num_machines 1 \
        --main_process_port "${MASTER_PORT:-29500}" \
        --num_cpu_threads_per_process "$threads" \
        -m "$@"
}

# Inputs must exist and outputs must not overwrite them.
rex_require_dir() {
    [[ -d "$1" ]] || rex_die "Input directory does not exist: $1"
}

rex_require_file() {
    [[ -f "$1" ]] || rex_die "Input file does not exist: $1"
}

rex_require_new_output() {
    [[ ! -e "$1" ]] || rex_die "Output already exists: $1"
}
