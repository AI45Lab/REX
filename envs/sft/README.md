# SFT environment

Use Linux x86-64, Python 3.12, eight Hopper GPUs, and CUDA 12.8 with a C++
compiler and Ninja. Run from the repository root. The lock pins public
ms-swift and Emerging Optimizers commits; no private checkout is required.
Transformer Engine's CUDA 12 core is explicit: its published extension metadata
names CUDA 13, while its build selects the core from `torch.version.cuda`.
The dependency metadata in this environment matches the pinned CUDA 12.8 torch.

Some CUDA extensions import torch while building. Bootstrap the build
dependencies before syncing the environment:

```bash
uv venv --python 3.12 .venv-sft
uv pip install --python .venv-sft/bin/python --index-url https://pypi.org/simple \
    torch==2.10.0 setuptools wheel packaging ninja psutil
# Use the cuDNN/NCCL headers installed with torch, plus the CUDA toolkit headers.
export NVTE_BUILD_USE_NVIDIA_WHEELS=1
export NVTE_WITH_NCCL_EP=0
export CPATH="${CUDA_HOME:-/usr/local/cuda}/include${CPATH:+:$CPATH}"
UV_PROJECT_ENVIRONMENT="$PWD/.venv-sft" UV_CONCURRENT_BUILDS=1 \
    MAX_JOBS=4 NVCC_THREADS=2 FLASH_ATTN_CUDA_ARCHS=90 TORCH_CUDA_ARCH_LIST=9.0 \
    uv sync --project envs/sft --locked --inexact
```

The optional Transformer Engine NCCL-EP extension requires PyTorch headers
absent from torch 2.10 and is disabled here. Megatron's configured all-to-all
expert dispatch still uses ordinary NCCL collectives.

Qwen/Intern use the Hopper FlashAttention-3 implementation shipped alongside
FlashAttention 2.8.3. Build its `hopper` subdirectory at the exact upstream
commit (the tag is recorded here only for context):

```bash
MAX_JOBS=4 NVCC_THREADS=2 FLASH_ATTENTION_DISABLE_SM80=TRUE \
    uv pip install --python .venv-sft/bin/python --no-build-isolation \
    'flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@060c9188beec3a8b62b33a3bfa6d5d2d44975fab#subdirectory=hopper'
.venv-sft/bin/python -c 'import flash_attn_interface, swift, megatron.core, transformer_engine; print("SFT imports OK")'
export REX_SWIFT_VENV="$PWD/.venv-sft"
```

FlashAttention-3 is a separately built architecture-specific extension. A
later `uv sync` removes packages absent from its lock, so repeat that build
command after resyncing, or use `uv sync --inexact`. Set `CUDA_HOME` to the
CUDA toolkit directory if it is not discoverable. SFT uses the repository
plugin for tokenizer adaptation, resident evaluation and HF export.

Gemma uses the repository's text adapter and attention implementation.
Its model files still need the processor/tokenizer assets supplied with the
checkpoint. The CPU-only download/excision environment does not need these
training kernels or Megatron.

Compile on the compute node with sufficient host RAM. The commands limit
package/compiler concurrency and target Hopper; FlashAttention-2 uses
`FLASH_ATTN_CUDA_ARCHS` independently of `TORCH_CUDA_ARCH_LIST`. Reduce
`MAX_JOBS` if memory is constrained. If an upstream prebuilt-wheel download
fails, rerun with `FLASH_ATTENTION_FORCE_BUILD=TRUE` and
`CAUSAL_CONV1D_FORCE_BUILD=TRUE` to compile from the pinned public sources.
For slow connections, uv supports `UV_HTTP_TIMEOUT=300`; completed downloads
remain reusable after an interrupted install.
