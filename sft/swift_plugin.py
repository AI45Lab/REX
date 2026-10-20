"""ms-swift Megatron plugin for REX SFT trajectories (``--external_plugins``).

Loaded by the swift command that ``sft/train.py`` builds. ``sft/train.py``
passes the run settings (recipe, input paths, loss mode) as JSON in
``REX_SFT_RUN``, so the swift command line stays stock. It adds:

* ``loss == "square"``: XTuner's square loss reduction for benign SFT
  (``token`` keeps swift's per-token mean, used for target relearning).
* callback ``rex_milestone`` (with stock ``--save_strategy steps`` and
  ``--save_steps train_iters``): saves exactly at the token milestones and turns each
  save into a repository checkpoint via ``sft.swift_checkpoint``: the endpoint
  at the output (node) root, intermediate milestones under ``milestones/``.
* the recipe's ``eval`` section: independent resident full/masked evaluation,
  without saving or reloading model weights.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import mpu

from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map
from swift.megatron.trainers.trainer import MegatronTrainer
from swift.utils import get_logger

# Swift executes a script named sft.py. Resolve our package explicitly so its
# script directory cannot shadow it, without changing global import search order.
import importlib.util
_package_dir = Path(__file__).resolve().parent
_cached_sft = sys.modules.get("sft")
if _cached_sft is None or not hasattr(_cached_sft, "__path__"):
    _spec = importlib.util.spec_from_file_location(
        "sft", _package_dir / "__init__.py", submodule_search_locations=[str(_package_dir)])
    _package = importlib.util.module_from_spec(_spec)
    sys.modules["sft"] = _package
    _spec.loader.exec_module(_package)

from sft.swift_checkpoint import finalize_checkpoint, milestone_steps, square_loss_terms  # noqa: E402
from sft.online_eval import OnlineEvalConfig, SftOnlineEvaluator, eval_milestone_steps  # noqa: E402
from sft.gemma_text import install_gemma_text_adapter  # noqa: E402
from sft.qwen_text import install_qwen_text_adapter  # noqa: E402

RUN = json.loads(os.environ["REX_SFT_RUN"])
RECIPE = RUN["config"]

if RUN["gemma_text"]:
    install_gemma_text_adapter()
else:
    install_qwen_text_adapter()

logger = get_logger()

LOSS_MODE = RUN["loss"]
if LOSS_MODE not in {"token", "square"}:
    raise ValueError(f"loss must be token or square, got {LOSS_MODE!r}")
# Integer-denominator resolution for the square loss (see square_loss_terms).
SQUARE_LOSS_SCALE = 10_000.0

# Packed tokens (padding excluded) this rank has fed through the loss.
_local_tokens = 0
_original_loss_func = MegatronTrainer.loss_func


def _loss_func(self, output_tensor, *, labels, loss_scale=None, channels=None, packed_seq_params=None):
    global _local_tokens
    result = _original_loss_func(self, output_tensor, labels=labels, loss_scale=loss_scale, channels=channels,
                                 packed_seq_params=packed_seq_params)
    cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
    if cu_seqlens is not None:
        num_samples = getattr(packed_seq_params, "num_samples", None) or len(cu_seqlens) - 1
        _local_tokens += int(cu_seqlens[num_samples])
    else:
        _local_tokens += int((labels != -100).sum()) if labels is not None else 0
    if LOSS_MODE == "token":
        return result
    if self.args.context_parallel_size != 1 or self.args.enable_dft_loss:
        raise RuntimeError("square loss supports context_parallel_size=1 without DFT only")
    _, _, metrics = result  # keep swift's token-mean CE as the logged loss
    losses, loss_mask = output_tensor.float(), labels != -100
    if loss_scale is not None:
        # e.g. ignore_empty_think zeroes the empty think block; such tokens
        # then count as unsupervised for the per-sample weight too.
        losses = losses * loss_scale
        loss_mask = loss_mask & (loss_scale != 0)
    numerator, denominator = square_loss_terms(losses, loss_mask, cu_seqlens, SQUARE_LOSS_SCALE)
    return numerator, denominator, metrics


MegatronTrainer.loss_func = _loss_func


def _keep_sentencepiece_attributes() -> None:
    """Intern-S2's remote tokenizer stores an ``offset`` on each domain
    SentencePieceProcessor; stock pickling keeps only the model proto, so the
    ``datasets`` worker copies lose it and every encode raises AttributeError."""
    try:
        import sentencepiece as spm
    except ImportError:
        return
    cls = spm.SentencePieceProcessor
    if getattr(cls, "_rex_keeps_attributes", False):
        return
    proto_getstate, proto_setstate = cls.__getstate__, cls.__setstate__

    def getstate(self):
        # SWIG's native pointer is also stored in __dict__ as `this`. The
        # stock proto serializer reconstructs it; it cannot cross processes.
        extra = {k: v for k, v in vars(self).items() if not k.startswith("_") and k != "this"}
        return proto_getstate(self), extra

    def setstate(self, state):
        if isinstance(state, tuple):
            proto, extra = state
            proto_setstate(self, proto)
            vars(self).update(extra)
        else:
            proto_setstate(self, state)

    cls.__getstate__, cls.__setstate__ = getstate, setstate
    cls._rex_keeps_attributes = True


_keep_sentencepiece_attributes()


class MilestoneCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.initial_model = Path(RUN["model"])
        self.max_tokens = int(RECIPE["max_tokens"])
        self.tokens_per_step = self.args.global_batch_size * self.args.seq_length
        self.steps = milestone_steps(tuple(RECIPE["save_token_milestones"]), self.tokens_per_step, self.max_tokens)
        self.total_step = self.max_tokens // self.tokens_per_step
        if self.args.save_strategy != "steps" or self.args.save_steps != self.total_step:
            raise ValueError("rex_milestone requires --save_strategy steps --save_steps train_iters")
        if self.args.train_iters != self.total_step:
            raise ValueError(f"train_iters={self.args.train_iters} but the token budget needs {self.total_step}")
        self.eval_config = OnlineEvalConfig.from_recipe(RECIPE.get("eval"), output_root=Path(RUN["eval_output_root"]))
        self.eval_steps = (eval_milestone_steps(self.eval_config.milestones, self.tokens_per_step, self.max_tokens)
                           if self.eval_config else {})
        self.evaluator = None
        self.evaluated_steps = set()
        self.eval_results: list[str] = []

    @staticmethod
    def _is_master() -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    def on_step_end(self):
        self._maybe_evaluate()
        if self.state.iteration in self.steps:
            self.state.should_save = True

    def on_train_begin(self):
        if self.eval_config is not None:
            from training.lm_eval_runner import preserve_rng_state

            with preserve_rng_state():
                self.evaluator = SftOnlineEvaluator(self.trainer, self.eval_config, self.initial_model)

    def on_step_begin(self):
        # All callbacks (including PrintCallback's writer) have initialized by
        # here. Still before the first train forward/data consumption.
        if self.state.iteration == 0:
            self._maybe_evaluate()

    @staticmethod
    def _consumed_tokens():
        consumed = torch.tensor([_local_tokens], dtype=torch.int64, device=torch.cuda.current_device())
        dist.all_reduce(consumed, group=mpu.get_data_parallel_group(with_context_parallel=True))
        return int(consumed.item())

    def _maybe_evaluate(self):
        step = self.state.iteration
        if step not in self.eval_steps or step in self.evaluated_steps:
            return
        if self.evaluator is None:
            raise RuntimeError("SFT evaluator was not initialized at train begin.")
        nominal = self.eval_steps[step]
        self.evaluator.run(step=step, nominal_tokens=nominal, consumed_tokens=self._consumed_tokens())
        self.evaluated_steps.add(step)
        self.eval_results.append(str(self.eval_config.output_root / f"tokens-{nominal:012d}" / "result.json"))

    def on_save(self, output_dir):
        step = self.state.iteration
        if step not in self.steps:
            return
        # The last pipeline stage holds the counts; summing over data-parallel
        # (with context-parallel) ranks gives the global total.
        consumed = self._consumed_tokens()
        dist.barrier()
        placed_path = [None]
        if self._is_master():
            nominal = self.steps[step]
            results = {
                "step": step, "total_steps": self.total_step, "nominal_tokens": nominal,
                "consumed_tokens": consumed, "nominal_tokens_per_step": self.tokens_per_step,
                "online_eval_results": list(self.eval_results), "data": RUN["data_summary"],
            }
            # Swift saved into <args.output_dir>/checkpoint-<step>; args.output_dir is the node.
            placed = finalize_checkpoint(
                Path(output_dir), node_dir=Path(self.args.output_dir), initial_model=self.initial_model,
                data_path=Path(RUN["data"]), recipe=RECIPE, args=RUN["args"], results=results,
                nominal_tokens=nominal, final=nominal == self.max_tokens)
            placed_path[0] = str(placed)
            logger.info(f"rex_milestone: finalized {output_dir} at {nominal} nominal tokens as {placed}")
        dist.broadcast_object_list(placed_path, src=0)
        # Swift reports state from the last rank; every rank must point at the
        # final node/milestone location after the export has been moved.
        self.state.last_model_checkpoint = placed_path[0]
        if self.state.best_model_checkpoint == str(output_dir):
            self.state.best_model_checkpoint = placed_path[0]
        dist.barrier()


megatron_callbacks_map["rex_milestone"] = MilestoneCallback
