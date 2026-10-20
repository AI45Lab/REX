"""Resident Megatron log-likelihood backend for the REX SFT plugin.

Copyright 2026 Shanghai AI Laboratory. Licensed under Apache-2.0.

Every expert-parallel rank executes the same number and shape of forwards.
Questions are sharded across data-parallel ranks and gathered in request order.
The LM head scores continuation positions only, avoiding full-vocabulary logits
for all prompt tokens. Supports TP=PP=CP=1 and arbitrary expert parallelism.
This backend is maintained here; it requires no modifications to ms-swift.
"""
from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch

SUPPORTED_ONLINE_EVAL_TASKS = {'mmlu', 'wmdp_bio', 'wmdp_chem', 'wmdp_cyber'}
IGNORE_INDEX = -100
_SEQ_ALIGN = 128
_MAX_TOKENS_PER_FORWARD = 4096

def plan_dp_forwards(
    num_groups: int,
    dp_rank: int,
    dp_size: int,
    questions_per_forward: int,
) -> tuple[int, int, list[list[tuple[int, int]]]]:
    """Plan one DP rank's batched forwards over the question groups.

    Questions are owned strided: rank ``r`` owns groups ``r, r+dp_size, ...``.
    Every rank owns ``ceil(num_groups / dp_size)`` slots (tail-padded by
    repeating the last real question) and packs ``qpf`` questions per forward.
    Because the plan depends only on ``num_groups``/``dp_size``, all ranks run
    the same number of forwards at the same shape -- the EP all-to-all lockstep
    invariant.
    """
    if questions_per_forward <= 0:
        raise ValueError('questions_per_forward must be positive')
    if num_groups <= 0 or dp_size <= 0:
        return 0, 0, []
    groups_per_rank = math.ceil(num_groups / dp_size)
    qpf = min(questions_per_forward, groups_per_rank)
    num_forwards = math.ceil(groups_per_rank / qpf)
    forwards: list[list[tuple[int, int]]] = []
    for f in range(num_forwards):
        slots: list[tuple[int, int]] = []
        for j in range(qpf):
            slot = f * qpf + j
            real_slot = slot if slot < groups_per_rank else groups_per_rank - 1
            group_idx = dp_rank + real_slot * dp_size
            source_idx = group_idx if group_idx < num_groups else num_groups - 1
            slots.append((slot, source_idx))
        forwards.append(slots)
    return groups_per_rank, qpf, forwards


def group_choices_by_context(contexts: Sequence[str]) -> list[list[int]]:
    """Group the answer-choice requests of each MCQ question together.

    lm_eval emits one ``(context, continuation)`` request per answer choice;
    the choices of one question share a byte-identical context and arrive
    consecutively in document order.
    """
    groups: list[list[int]] = []
    for i, context in enumerate(contexts):
        if groups and contexts[groups[-1][0]] == context:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError('multiple must be positive')
    if value <= 0:
        return multiple
    return ((value + multiple - 1) // multiple) * multiple


def clamp_questions_per_forward(
    requested: int,
    *,
    max_choices: int,
    seq_length: int,
    max_tokens_per_forward: int,
) -> int:
    """Clamp questions/forward so the MoE routes a bounded token count."""
    if requested <= 0:
        raise ValueError('requested questions_per_forward must be positive')
    per_question_tokens = max(1, max_choices * seq_length)
    budget = max(1, max_tokens_per_forward // per_question_tokens)
    return max(1, min(requested, budget))


def split_context_continuation(tokenizer, context: str,
                               continuation: str) -> tuple[list[int], list[int]]:
    """Tokenize ``(context, continuation)`` the way lm_eval's HFLM does.

    Mirrors ``HFLM._encode_pair``: a trailing space on the context is merged
    into the continuation before encoding, the continuation span is the suffix
    of the whole-string encoding beyond the context prefix length. Both encodes
    use the tokenizer default ``add_special_tokens`` (as HFLM does), preserving
    Gemma's leading ``<bos>`` while leaving BOS-free tokenizers unchanged.
    """
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole_enc = tokenizer.encode(context + continuation)
    context_enc = tokenizer.encode(context)
    context_enc_len = len(context_enc)
    continuation_enc = whole_enc[context_enc_len:]
    return context_enc, continuation_enc


def _pad(values: Sequence[list[int]], value: int, length: int) -> torch.Tensor:
    output = torch.full((len(values), length), value, dtype=torch.long)
    for row, tensor in enumerate(values):
        size = min(len(tensor), length)
        output[row, :size] = torch.tensor(tensor[:size], dtype=torch.long)
    return output


def build_mcq_batch(
    examples: Sequence[tuple[list[int], list[int]]],
    *,
    pad_token_id: int,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    """Build an MCore forward batch for MCQ log-likelihood scoring.

    Each example is ``context_ids + continuation_ids`` with source labels
    ``IGNORE_INDEX`` on context positions and the token id on continuation
    positions.  Labels are shifted to MCore's next-token convention and the
    loss mask covers precisely the continuation positions.
    Over-long sequences are left-truncated so the continuation is preserved.
    """
    rows: list[dict[str, list[int]]] = []
    for context_ids, continuation_ids in examples:
        if not continuation_ids:
            raise ValueError('continuation must contain at least one token')
        full_ids = list(context_ids) + list(continuation_ids)
        source_labels = [IGNORE_INDEX] * len(context_ids) + list(continuation_ids)
        if len(full_ids) > sequence_length:
            overflow = len(full_ids) - sequence_length
            if overflow >= len(context_ids):
                raise ValueError('continuation does not fit after truncating context')
            full_ids = full_ids[overflow:]
            source_labels = source_labels[overflow:]
        rows.append({'input_ids': full_ids, 'labels': source_labels})

    input_ids = _pad([row['input_ids'] for row in rows], pad_token_id, sequence_length)
    source_labels = _pad([row['labels'] for row in rows], IGNORE_INDEX, sequence_length)
    padding_mask = input_ids.ne(pad_token_id)
    labels = torch.full_like(source_labels, IGNORE_INDEX)
    labels[:, :-1] = source_labels[:, 1:]
    labels = labels.masked_fill(~padding_mask, IGNORE_INDEX)
    position_ids = torch.arange(sequence_length, dtype=torch.long).expand(len(rows), -1)
    causal = torch.triu(torch.ones((sequence_length, sequence_length), dtype=torch.bool), diagonal=1)
    attention_mask = causal[None, None, :, :] | (~padding_mask[:, None, None, :])
    return {
        'input_ids': input_ids,
        'labels': labels,
        'loss_mask': labels.ne(IGNORE_INDEX),
        'position_ids': position_ids,
        'attention_mask': attention_mask,
    }


def gather_scoring_targets(
    labels: torch.Tensor, loss_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten continuation positions into ``(row_idx, pos_idx, target_ids)``."""
    mask = loss_mask.to(torch.bool)
    row_idx, pos_idx = torch.nonzero(mask, as_tuple=True)
    target_ids = labels[row_idx, pos_idx]
    return row_idx, pos_idx, target_ids


def reduce_selected_logprobs(
    selected_logits: torch.Tensor,
    row_idx: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    num_rows: int,
    final_logit_softcapping: float | None = None,
) -> torch.Tensor:
    """Sum per-position gold log-probs back into per-row log-probabilities.

    ``final_logit_softcapping`` replays Gemma-family output-side tanh softcapping
    before log-softmax. The selective LM-head path bypasses the model output
    layer, so capped models must apply this transform manually.
    """
    if final_logit_softcapping is not None:
        selected_logits = selected_logits / final_logit_softcapping
        selected_logits = torch.tanh(selected_logits)
        selected_logits = selected_logits * final_logit_softcapping
    logp = torch.log_softmax(selected_logits.float(), dim=-1)
    tok = logp.gather(1, target_ids.view(-1, 1).long()).squeeze(1)
    out = torch.zeros(num_rows, dtype=torch.float32, device=selected_logits.device)
    out.index_add_(0, row_idx.long(), tok.to(out.dtype))
    return out


def _extra_model_kwargs(model, batch: dict) -> dict:
    """Return bridge-specific kwargs required even for pure-text eval batches.

    mcore-bridge's MultimodalGPTModel requires every key declared by
    ``language_model.extra_forward_keys`` to be present. Gemma4 declares
    ``mm_token_type_ids``; it is unused for text-only data, so ``None`` is valid.
    """
    for _ in range(16):
        inner = getattr(model, 'module', None)
        if inner is None:
            break
        model = inner
    language_model = getattr(model, 'language_model', None)
    if language_model is None:
        language_model = model
    keys = getattr(language_model, 'extra_forward_keys', None) or ()
    return {key: batch.get(key) for key in keys}


def _online_eval_log(message: str) -> None:
    """Print one concise online-eval line from global rank 0 only."""
    if int(os.environ.get('LOCAL_RANK', '0')) == 0:
        print(f'[online_eval] {message}', flush=True)


def _import_lm_eval_base():
    from lm_eval.api.model import LM

    return LM


def _make_mcore_eval_lm_class():
    """Build the LM class lazily so lm_eval is only imported on GPU nodes."""
    LM = _import_lm_eval_base()

    class _MCoreEvalLM(LM):
        """lm_eval backend that scores continuations with an MCore forward."""

        def __init__(
            self,
            model,
            tokenizer,
            *,
            sequence_length: int,
            pad_token_id: int,
            dp_group,
            dp_rank: int,
            dp_size: int,
            device,
            questions_per_forward: int = 1,
            max_tokens_per_forward: int = _MAX_TOKENS_PER_FORWARD,
        ) -> None:
            super().__init__()
            self._model = model
            self._tokenizer = tokenizer
            self._sequence_length = sequence_length
            self._pad_token_id = pad_token_id
            self._dp_group = dp_group
            self._dp_rank = dp_rank
            self._dp_size = dp_size
            self._device = device
            self._questions_per_forward = questions_per_forward
            self._max_tokens_per_forward = max_tokens_per_forward
            self.task_name = 'online_eval'

        def _forward_hidden(self, batch: dict[str, torch.Tensor], *, seq_length: int) -> torch.Tensor:
            """Run the decoder and return hidden states as ``[b, s, h]``.

            A forward hook on ``gpt.decoder`` captures exactly the tensor the
            output layer would consume.  ``post_process=False`` returns before
            the output layer materializes fp32 logits over the full vocab.
            Decoder execution (and therefore any EP all-to-all) is identical to
            the training forward.
            """
            gpt = getattr(self._model, 'language_model', self._model)
            captured: dict[str, torch.Tensor] = {}

            def _hook(_module, _inputs, output):
                captured['hidden'] = output[0] if isinstance(output, tuple) else output

            handle = gpt.decoder.register_forward_hook(_hook)
            prev_post_process = gpt.post_process
            gpt.post_process = False
            try:
                # NOTE: do not pass padding_mask here. The instantiated model
                # is mcore_bridge's GPTModel wrapper, whose forward signature
                # does not expose mcore 0.18's padding_mask argument. Padding
                # tokens are already blocked by attention_mask; routing them
                # through the MoE does not contribute to the score.
                self._model(
                    input_ids=batch['input_ids'].to(self._device),
                    position_ids=batch['position_ids'].to(self._device),
                    attention_mask=batch['attention_mask'].to(self._device),
                    labels=None,
                    **_extra_model_kwargs(self._model, batch),
                )
            finally:
                gpt.post_process = prev_post_process
                handle.remove()
            hidden = captured['hidden']  # [s, b, h]
            return hidden.transpose(0, 1)  # [b, s, h]

        def _output_weight(self) -> torch.Tensor:
            """Return the full (unsharded, TP==1) LM-head weight."""
            gpt = getattr(self._model, 'language_model', self._model)
            if getattr(gpt, 'share_embeddings_and_output_weights', False):
                return gpt.shared_embedding_or_output_weight()
            return gpt.output_layer.weight

        def _score_batch(
            self,
            examples: Sequence[tuple[list[int], list[int]]],
            *,
            seq_length: int,
        ) -> torch.Tensor:
            batch = build_mcq_batch(
                examples, pad_token_id=self._pad_token_id, sequence_length=seq_length)
            hidden = self._forward_hidden(batch, seq_length=seq_length)  # [b, s, h]
            labels = batch['labels'].to(self._device)
            loss_mask = batch['loss_mask'].to(self._device)
            row_idx, pos_idx, target_ids = gather_scoring_targets(labels, loss_mask)
            selected_hidden = hidden[row_idx, pos_idx]  # [num_selected, h]
            weight = self._output_weight()
            selected_logits = torch.nn.functional.linear(selected_hidden.to(weight.dtype), weight)
            gpt = getattr(self._model, 'language_model', self._model)
            final_logit_softcapping = getattr(gpt, 'final_logit_softcapping', None)
            return reduce_selected_logprobs(
                selected_logits,
                row_idx,
                target_ids,
                num_rows=labels.shape[0],
                final_logit_softcapping=final_logit_softcapping)

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            args = [request.args for request in requests]
            total = len(args)
            if total == 0:
                return []
            _online_eval_log(f'{self.task_name}: data loading completed ({total} requests)')
            encoded = [
                split_context_continuation(self._tokenizer, context, continuation)
                for context, continuation in args
            ]
            groups = group_choices_by_context([context for context, _ in args])
            num_groups = len(groups)
            max_choices = max(len(group) for group in groups)
            max_len = max(len(ctx) + len(cont) for ctx, cont in encoded)
            seq_length = min(self._sequence_length, round_up_to_multiple(max_len, _SEQ_ALIGN))
            effective_qpf = clamp_questions_per_forward(
                self._questions_per_forward,
                max_choices=max_choices,
                seq_length=seq_length,
                max_tokens_per_forward=self._max_tokens_per_forward,
            )
            groups_per_rank, qpf, forwards = plan_dp_forwards(
                num_groups, self._dp_rank, self._dp_size, effective_qpf)
            local = torch.zeros(
                groups_per_rank, max_choices, dtype=torch.float64, device=self._device)
            for slots in forwards:
                batch_examples: list[tuple[list[int], list[int]]] = []
                for _slot, source_idx in slots:
                    choices = [encoded[m] for m in groups[source_idx]]
                    while len(choices) < max_choices:
                        choices.append(choices[-1])
                    batch_examples.extend(choices)
                logprobs = self._score_batch(batch_examples, seq_length=seq_length)
                logprobs = logprobs.view(qpf, max_choices)
                for row, (slot, _source_idx) in enumerate(slots):
                    if slot < groups_per_rank:
                        local[slot] = logprobs[row].to(local.dtype)
            gathered = [torch.zeros_like(local) for _ in range(self._dp_size)]
            import torch.distributed as dist

            dist.all_gather(gathered, local, group=self._dp_group)
            results: list[tuple[float, bool]] = [None] * total  # type: ignore[list-item]
            for group_idx in range(num_groups):
                rank = group_idx % self._dp_size
                slot = group_idx // self._dp_size
                for choice_pos, request_idx in enumerate(groups[group_idx]):
                    logprob = float(gathered[rank][slot][choice_pos].item())
                    results[request_idx] = (logprob, True)
            return results

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError('online evaluation only supports MCQ loglikelihood tasks')

        def generate_until(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError('online evaluation only supports MCQ loglikelihood tasks')

    return _MCoreEvalLM


def MCoreEvalLM(*args, **kwargs):  # noqa: N802 - factory mimics a class
    return _make_mcore_eval_lm_class()(*args, **kwargs)
