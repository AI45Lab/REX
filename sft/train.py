"""Run one S_std trajectory: convert the recipe's data, then exec ms-swift Megatron.

Invoked by ``scripts/train_sft.sh`` inside the ms-swift environment. The swift
process loads ``sft/swift_plugin.py``, which receives the run settings as JSON
in ``REX_SFT_RUN``, writes one ``run.json`` into every saved checkpoint and
places the endpoint at the output (node) root.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from sft.swift_data import build as convert_data

REPO_DIR = Path(__file__).resolve().parents[1]

# Per model_type: swift template, thinking wrapper used in data conversion,
# Megatron attention backend.
MODEL_FAMILIES = {
    "qwen3_5_moe": {"template": "rex_qwen3_5_text", "thinking_format": "qwen3_5", "attention_backend": "flash_3"},
    # TE unfused is only the Swift/TE configuration; the plugin routes packed
    # training to memory-efficient CUDA attention, including 512-dim heads.
    "gemma4": {"template": "rex_gemma4_text", "thinking_format": "gemma4", "attention_backend": "unfused"},
}
# benign_sft drops samples longer than one pack; target_relearning (megatron
# pt, raw text) splits long documents instead.
PURPOSES = {
    "benign_sft": {"command": "sft", "loss": "square", "truncation": "delete"},
    "target_relearning": {"command": "pt", "loss": "token", "truncation": "split"},
}
DATA_WORKERS = 32


def swift_arguments(*, model: Path, output: Path, data: Path, recipe: dict[str, Any], family: dict[str, str],
                    purpose: dict[str, str], pack_length: int, global_batch: int, train_iters: int) -> list[str]:
    opt, seed = recipe["optimizer"], str(recipe["seed"])
    # --cross_entropy_fusion_impl te: TE's cross entropy works on the bf16 logits
    # in place; the native fused one materialises a float32 copy (~29 GiB at
    # pack 31250 x 251k vocab) and OOMs.
    # --muon_split_qkv false: Megatron splits linear_qkv as [q, k, v] per query
    # group, but Qwen3.5 gated attention also packs the output gate into q, so
    # the split shape is wrong; orthogonalise the fused matrix as one block.
    return [
        purpose["command"],
        "--model", str(model),
        # Inputs are local HF checkpoints; do not query ModelScope for a newer
        # revision of the training artifact during offline initialization.
        "--use_hf", "true", "--check_model", "false",
        "--template", family["template"],
        "--dataset", str(data),
        "--external_plugins", str(REPO_DIR / "sft" / "swift_plugin.py"),
        "--callbacks", "rex_milestone",
        # Stock Swift supports steps/epoch. Its default callback saves only the
        # endpoint; rex_milestone adds explicitly configured earlier saves.
        "--save_strategy", "steps", "--save_steps", str(train_iters),
        "--output_dir", str(output),
        "--add_version", "false",
        "--tuner_type", "full",
        "--torch_dtype", "bfloat16", "--bf16", "true",
        "--packing", "true", "--max_length", str(pack_length), "--truncation_strategy", purpose["truncation"],
        "--micro_batch_size", "1", "--global_batch_size", str(global_batch),
        "--train_iters", str(train_iters),
        "--optimizer", "muon", "--muon_momentum", str(opt["muon_momentum"]),
        "--muon_use_nesterov", str(opt["muon_use_nesterov"]).lower(),
        "--muon_scale_mode", opt["muon_scale_mode"], "--muon_extra_scale_factor", str(opt["muon_extra_scale_factor"]),
        "--muon_scalar_optimizer", opt["muon_scalar_optimizer"], "--muon_split_qkv", "false",
        "--lr", str(opt["lr"]), "--min_lr", str(opt["min_lr"]), "--lr_decay_style", opt["lr_decay_style"],
        "--lr_warmup_fraction", str(opt["lr_warmup_fraction"]),
        "--weight_decay", str(opt["weight_decay"]), "--clip_grad", str(opt["clip_grad"]),
        "--seed", seed, "--data_seed", seed,
        "--split_dataset_ratio", "0", "--eval_iters", "0",
        "--tensor_model_parallel_size", "1", "--pipeline_model_parallel_size", "1",
        "--expert_model_parallel_size", "8", "--expert_tensor_parallel_size", "1",
        "--sequence_parallel", "false", "--language_model_only", "true",
        "--moe_grouped_gemm", "true", "--moe_permute_fusion", "true", "--moe_aux_loss_coeff", "0",
        "--recompute_granularity", "full", "--recompute_method", "uniform", "--recompute_num_layers", "1",
        "--finetune", "true", "--cross_entropy_loss_fusion", "true", "--cross_entropy_fusion_impl", "te",
        "--gradient_accumulation_fusion", "false",
        "--save_safetensors", "true", "--no_save_optim", "true", "--no_save_rng", "true",
        "--dataset_num_proc", str(DATA_WORKERS),
        "--logging_steps", "1", "--report_to", "tensorboard",
        "--attention_backend", family["attention_backend"],
    ]


def swift_environment(*, swift_root: Path | None, seed: int, gpus: int, run: dict[str, Any], gemma: bool) -> dict[str, str]:
    # Resolve libraries from the active environment, including an explicitly
    # configured overlay venv. A checkout-local `env/` is not required.
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None or torch_spec.origin is None:
        raise SystemExit("PyTorch is missing from REX_SWIFT_VENV")
    site = str(Path(torch_spec.origin).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_DIR)] + ([str(swift_root)] if swift_root else []))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [
        f"{site}/nvidia/cudnn/lib", f"{site}/nvidia/nccl/lib", f"{site}/torch/lib", env.get("LD_LIBRARY_PATH")]))
    # Transformer Engine globs lib<name>.so* under CUDNN_PATH; the wheel ships only .so.9.
    env["CUDNN_PATH"] = f"{site}/nvidia/cudnn"
    env.update(CUDA_DEVICE_MAX_CONNECTIONS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="offline",
               PYTHONHASHSEED=str(seed), NPROC_PER_NODE=str(gpus), MASTER_PORT=env.get("MASTER_PORT", "29500"))
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if gemma:
        env["MALLOC_ARENA_MAX"] = "2"
    env["REX_SFT_RUN"] = json.dumps(run)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    swift_root = Path(os.environ["SWIFT_ROOT"]).resolve() if os.environ.get("SWIFT_ROOT") else None
    recipe = yaml.safe_load(args.config.read_text())
    model, output = args.model.resolve(), args.output.resolve()
    for required in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        if not (model / required).is_file():
            raise SystemExit(f"Model directory is missing {required}: {model}")
    data = (REPO_DIR / recipe["data"]).resolve()
    if not data.exists():
        raise SystemExit(f"SFT data path does not exist: {data}")

    model_type = json.loads((model / "config.json").read_text())["model_type"]
    if model_type not in MODEL_FAMILIES:
        raise SystemExit(f"Unsupported model_type for swift SFT: {model_type}")
    family = MODEL_FAMILIES[model_type]
    purpose = PURPOSES[recipe["training_purpose"]]
    pack_length = int(recipe["pack_length"][model_type])
    tokens_per_step, max_tokens, seed = int(recipe["tokens_per_step"]), int(recipe["max_tokens"]), int(recipe["seed"])
    if tokens_per_step % pack_length:
        raise SystemExit(f"tokens_per_step ({tokens_per_step}) must be a multiple of pack_length ({pack_length})")

    # Check both schedules before converting data or starting swift.
    from sft.online_eval import OnlineEvalConfig, eval_milestone_steps
    from sft.swift_checkpoint import milestone_steps

    milestone_steps(tuple(recipe["save_token_milestones"]), tokens_per_step, max_tokens)
    eval_output = output / "online_eval"
    eval_config = OnlineEvalConfig.from_recipe(recipe.get("eval"), output_root=eval_output)
    if eval_config is not None:
        eval_milestone_steps(eval_config.milestones, tokens_per_step, max_tokens)

    import torch

    gpus = torch.cuda.device_count()
    data_dir = output / "swift_data"
    print(f"[train_sft] converting {data} -> {data_dir} (thinking format {family['thinking_format']})", flush=True)
    summary = convert_data(data, data_dir, thinking_format=family["thinking_format"], seed=seed,
                           workers=DATA_WORKERS)
    print(f"[train_sft] {summary['records']} records, {summary['skipped_records']} skipped, "
          f"sha256 {summary['sha256']}", flush=True)

    global_batch, train_iters = tokens_per_step // pack_length, max_tokens // tokens_per_step
    run = {
        "model": str(model), "data": str(data), "config": recipe, "loss": purpose["loss"],
        "gemma_text": model_type == "gemma4", "eval_output_root": str(eval_output),
        "args": {"model_type": model_type, "template": family["template"],
                 "pack_length": pack_length, "global_batch_size": global_batch, "train_iters": train_iters},
        "data_summary": summary,
    }
    argv = swift_arguments(model=model, output=output, data=data_dir / "train.jsonl", recipe=recipe, family=family,
                           purpose=purpose, pack_length=pack_length, global_batch=global_batch,
                           train_iters=train_iters)
    env = swift_environment(swift_root=swift_root, seed=seed, gpus=gpus, run=run, gemma=model_type == "gemma4")
    print(f"[train_sft] model={model} ({model_type}, template {family['template']}) output={output}")
    print(f"[train_sft] purpose={recipe['training_purpose']} loss={purpose['loss']} seed={seed} "
          f"lr={recipe['optimizer']['lr']} min_lr={recipe['optimizer']['min_lr']}")
    print(f"[train_sft] pack_length={pack_length} global_batch={global_batch} train_iters={train_iters} "
          f"save_milestones={recipe['save_token_milestones']}")
    if eval_config is not None:
        print(f"[train_sft] resident_eval={list(eval_config.milestones)} tasks={list(eval_config.tasks)} "
              f"results={eval_output}", flush=True)
    sys.stdout.flush()
    os.chdir(output)
    os.execve(sys.executable, [sys.executable, "-m", "swift.cli._megatron.main", *argv], env)


if __name__ == "__main__":
    main()
