"""Shared, ordered trainer setup with method-specific preparation callbacks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from transformers import set_seed

from training.config import load_config


@dataclass
class TrainingContext:
    config_path: Path
    config: dict
    setup: Any
    accelerator: Any
    seed: int
    prepared: Any = None
    loaders: tuple = ()
    tokenizer: Any = None
    model: Any = None
    optimizer: Any = None
    scheduler: Any = None
    reference_model: Any = None
    method_state: Any = None

    def init_trackers(self, extra: dict, *, run_name: str | None = None) -> None:
        self.accelerator.init_trackers(
            project_name=str(self.config.get("wandb_project", "rex")),
            config={**self.config, **extra},
            init_kwargs={"wandb": {"name": run_name or self.setup.run_name}},
        )


def resolve_output_dir(config) -> tuple[str, Path]:
    """The output directory is given on the command line; its name is the run name."""
    output_dir = Path(str(config["output_dir"]))
    return output_dir.name, output_dir


def load_frozen_reference_model(
    config, checkpoint_dir, *, accelerator, model_loader, convert_expanded: bool = False
):
    reference_config = dict(config)
    reference_config["initial_checkpoint_dir"] = str(checkpoint_dir)
    _, model = model_loader(reference_config, accelerator)
    if convert_expanded:
        from moe.runtime import convert_packed_moe_to_expanded_runtime

        convert_packed_moe_to_expanded_runtime(
            model, initial_checkpoint_dir=checkpoint_dir
        )
    model.eval()
    model.requires_grad_(False)
    return model


def build_training_context(
    config_path: Path,
    *,
    paths: dict[str, str],
    prepare_fn,
    build_dataloaders,
    needs_reference,
    accelerator_factory,
    model_loader,
    configure_model: Callable,
    prepare_model: Callable,
    finish_setup: Callable | None = None,
    reference_expanded: Callable | None = None,
) -> TrainingContext:
    config_path = config_path.resolve()
    config = load_config(config_path)
    # Recipes carry no machine paths; inputs and outputs come from the command line.
    overlap = sorted(set(paths) & set(config))
    if overlap:
        raise ValueError(f"Path keys must not appear in the recipe config: {overlap}")
    config.update({key: str(Path(value).resolve()) for key, value in paths.items()})
    prepared = prepare_fn(
        config, config_path, world_size=int(os.environ.get("WORLD_SIZE", "1"))
    )
    setup = prepared[0] if isinstance(prepared, tuple) else prepared
    accelerator = accelerator_factory(
        config, gradient_accumulation_steps=setup.gradient_accumulation_steps
    )
    seed = int(config.get("seed", 42))
    set_seed(seed)
    ctx = TrainingContext(
        config_path, config, setup, accelerator, seed, prepared=prepared
    )
    ctx.loaders = build_dataloaders(
        config,
        accelerator=accelerator,
        micro_batch_size=setup.micro_batch_size,
        global_batch_size=setup.effective_global_batch_size,
        seed=seed,
    )
    ctx.tokenizer, ctx.model = model_loader(config, accelerator)
    configure_model(ctx)
    ctx.model, ctx.optimizer, ctx.scheduler = prepare_model(ctx)
    reference = needs_reference(ctx) if callable(needs_reference) else needs_reference
    if reference:
        ctx.reference_model = load_frozen_reference_model(
            config,
            reference,
            accelerator=accelerator,
            model_loader=model_loader,
            convert_expanded=bool(reference_expanded and reference_expanded(ctx)),
        )
        ctx.reference_model = accelerator.prepare(ctx.reference_model)
    if finish_setup is not None:
        finish_setup(ctx)
    return ctx
