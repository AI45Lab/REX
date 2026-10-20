from __future__ import annotations

import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F
from tqdm import tqdm

try:
    from lm_eval.api.model import TemplateLM
except ImportError:

    class TemplateLM:  # type: ignore[no-redef]
        """Fallback base used until lm-eval validation is actually requested."""


TOKENIZER_INFINITY = 10**20
LoglikelihoodRequest = tuple[tuple[str, str], list[int], list[int]]


@dataclass
class LoglikelihoodBatch:
    request_strings: list[tuple[str, str]]
    context_tokens: list[list[int]]
    continuation_tokens: list[list[int]]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_lengths: list[int]


def _deserialize_gathered_objects(
    gathered_payloads: torch.Tensor,
    payload_sizes: torch.Tensor,
) -> list[Any]:
    sizes = [int(size) for size in payload_sizes.detach().cpu().reshape(-1).tolist()]
    if not sizes:
        return []
    if gathered_payloads.numel() % len(sizes) != 0:
        raise RuntimeError(
            "Distributed lm-eval gathered an invalid padded object buffer."
        )
    padded_size = gathered_payloads.numel() // len(sizes)
    if any(size < 0 or size > padded_size for size in sizes):
        raise RuntimeError(
            "Distributed lm-eval gathered an invalid serialized object size."
        )

    gathered_cpu = gathered_payloads.detach().cpu().reshape(len(sizes), padded_size)
    return [
        pickle.loads(gathered_cpu[rank, :size].numpy().tobytes())
        for rank, size in enumerate(sizes)
    ]


def _first_defined_int_attr(obj: Any, attr_names: tuple[str, ...]) -> int | None:
    for attr_name in attr_names:
        value = getattr(obj, attr_name, None)
        if value is not None:
            return int(value)
    return None


def _first_positive_int_attr(obj: Any, attr_names: tuple[str, ...]) -> int | None:
    for attr_name in attr_names:
        value = getattr(obj, attr_name, None)
        if value is not None and int(value) > 0:
            return int(value)
    return None


def _loglikelihood_collate_key(
    req: LoglikelihoodRequest,
) -> tuple[int, tuple[int, ...]]:
    tokens = req[1] + req[2]
    return -len(tokens), tuple(tokens)


def _loglikelihood_context_cache_key(req: LoglikelihoodRequest) -> list[int]:
    return req[1] + req[2][:-1]


class RexFSDPLMEvalWrapper(TemplateLM):
    backend = "causal"

    def __init__(
        self,
        *,
        accelerator: Any,
        model: torch.nn.Module,
        tokenizer: Any,
        batch_size: int | str = 1,
        max_length: int | None = None,
        distributed: bool = True,
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self._model = model
        self.tokenizer = tokenizer
        self._device = accelerator.device
        self._rank = int(accelerator.process_index) if distributed else 0
        self._world_size = int(accelerator.num_processes) if distributed else 1
        self._batch_size = batch_size
        self._max_length = max_length

    @property
    def eot_token_id(self) -> int:
        token_id = _first_defined_int_attr(
            self.tokenizer, ("eos_token_id", "pad_token_id", "bos_token_id")
        )
        if token_id is not None:
            return token_id
        raise ValueError(
            "Tokenizer must define at least one of eos_token_id, pad_token_id, or bos_token_id."
        )

    @property
    def prefix_token_id(self) -> int:
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        return int(bos_token_id) if bos_token_id is not None else self.eot_token_id

    @property
    def max_length(self) -> int:
        if self._max_length is not None:
            return int(self._max_length)

        model_config = getattr(
            self.accelerator.unwrap_model(self._model), "config", None
        )
        model_max_length = _first_positive_int_attr(
            model_config, ("n_positions", "max_position_embeddings", "n_ctx")
        )
        if model_max_length is not None:
            return model_max_length

        text_config = getattr(model_config, "text_config", None)
        text_max_length = _first_positive_int_attr(
            text_config,
            ("max_position_embeddings", "seq_length", "n_positions", "n_ctx"),
        )
        if text_max_length is not None:
            return text_max_length

        tokenizer_max = getattr(self.tokenizer, "model_max_length", None)
        if tokenizer_max is not None and int(tokenizer_max) < TOKENIZER_INFINITY:
            return int(tokenizer_max)
        return 2048

    @property
    def batch_size(self) -> int | str:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        name = str(
            getattr(self.tokenizer, "name_or_path", type(self.tokenizer).__name__)
        )
        return name.replace("/", "__")

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size <= 1:
            return tensor
        return self.accelerator.gather(tensor)

    def gather_object(self, obj: Any, dst: int = 0) -> list[Any] | None:
        if self.world_size <= 1:
            return [obj]
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "Distributed lm-eval requires torch.distributed to be initialized."
            )

        # torch.distributed.gather_object ultimately uses ProcessGroupNCCL.gather.
        # That path has failed reproducibly on the H200 training workers after a
        # long FSDP run, even though tensor all-gather remains healthy. Serialize
        # explicitly and use the same tensor all-gather path already exercised by
        # distributed request balancing. Every rank participates in both
        # collectives; only ``dst`` deserializes and returns the objects.
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        payload_tensor = torch.tensor(
            bytearray(payload),
            device=self.device,
            dtype=torch.uint8,
        )
        payload_size = torch.tensor(
            [payload_tensor.numel()],
            device=self.device,
            dtype=torch.long,
        )
        gathered_sizes = self.all_gather(payload_size).reshape(-1)
        if gathered_sizes.numel() != self.world_size:
            raise RuntimeError(
                "Distributed lm-eval gathered an unexpected number of object sizes."
            )
        padded_size = int(gathered_sizes.max().item())
        padded_payload = torch.zeros(
            padded_size,
            device=self.device,
            dtype=torch.uint8,
        )
        padded_payload[: payload_tensor.numel()] = payload_tensor
        gathered_payloads = self.all_gather(padded_payload).reshape(-1)
        expected_numel = self.world_size * padded_size
        if gathered_payloads.numel() != expected_numel:
            raise RuntimeError(
                "Distributed lm-eval gathered an unexpected serialized object buffer size."
            )
        if self.rank != dst:
            return None
        return _deserialize_gathered_objects(gathered_payloads, gathered_sizes)

    def barrier(self) -> None:
        if self.world_size > 1:
            self.accelerator.wait_for_everyone()

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        **kwargs: Any,
    ) -> list[int]:
        tokenizer_kwargs: dict[str, Any] = {}
        if add_special_tokens is not None:
            tokenizer_kwargs["add_special_tokens"] = add_special_tokens
        return list(self.tokenizer.encode(string, **tokenizer_kwargs, **kwargs))

    def tok_decode(
        self, tokens: int | Iterable[int], skip_special_tokens: bool = True
    ) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _pad_batch(
        self, sequences: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.eot_token_id
        lengths = [len(sequence) for sequence in sequences]
        max_len = max(lengths)
        input_ids = torch.full(
            (len(sequences), max_len),
            int(pad_token_id),
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_idx, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row_idx, :length] = torch.tensor(
                sequence, device=self.device, dtype=torch.long
            )
            attention_mask[row_idx, :length] = 1
        return input_ids, attention_mask, lengths

    def _forward_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        return outputs.logits

    def _distributed_max_int(self, value: int) -> int:
        if self.world_size <= 1:
            return value
        gathered = self.all_gather(
            torch.tensor(int(value), device=self.device, dtype=torch.long)
        )
        return int(gathered.max().item())

    def _forward_dummy_loglikelihood_chunk(self) -> None:
        input_ids = torch.tensor(
            [[self.prefix_token_id]], device=self.device, dtype=torch.long
        )
        attention_mask = torch.ones_like(input_ids)
        _ = self._forward_logits(input_ids, attention_mask)

    def _prepare_loglikelihood_batch(
        self, chunk: list[LoglikelihoodRequest]
    ) -> LoglikelihoodBatch:
        sequences: list[list[int]] = []
        continuation_tokens: list[list[int]] = []
        context_tokens: list[list[int]] = []
        request_strings: list[tuple[str, str]] = []
        for request_str, context_enc, continuation_enc in chunk:
            if len(continuation_enc) > self.max_length:
                raise ValueError(
                    f"Continuation length {len(continuation_enc)} exceeds lm_eval max_length={self.max_length}."
                )
            if not continuation_enc:
                raise ValueError(
                    "lm-eval loglikelihood requests must have a non-empty continuation."
                )
            tokens = (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
            if not tokens:
                tokens = [self.prefix_token_id]
            sequences.append(tokens)
            continuation_tokens.append(continuation_enc)
            context_tokens.append(context_enc)
            request_strings.append(request_str)

        input_ids, attention_mask, input_lengths = self._pad_batch(sequences)
        return LoglikelihoodBatch(
            request_strings=request_strings,
            context_tokens=context_tokens,
            continuation_tokens=continuation_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
        )

    def _score_loglikelihood_batch(
        self,
        *,
        reordered: Any,
        batch: LoglikelihoodBatch,
        logits: torch.Tensor,
        pbar: Any,
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for request_str, ctx_toks, logit_row, inplen, cont_toks in zip(
            batch.request_strings,
            batch.context_tokens,
            logits,
            batch.input_lengths,
            batch.continuation_tokens,
            strict=True,
        ):
            cont_len = len(cont_toks)
            continuation_log_probs = F.log_softmax(
                logit_row[:inplen][-cont_len:].float(), dim=-1
            )
            greedy_tokens = continuation_log_probs.argmax(dim=-1).unsqueeze(0)
            continuation_log_probs = continuation_log_probs.unsqueeze(0)
            for (
                cached_request_str,
                cached_cont_toks,
                cached_log_probs,
            ) in reordered.get_cache(
                req_str=request_str,
                cxt_toks=ctx_toks,
                cont_toks=cont_toks,
                logits=continuation_log_probs,
            ):
                result = self._score_cached_loglikelihood(
                    cached_cont_toks=cached_cont_toks,
                    cached_log_probs=cached_log_probs,
                    greedy_tokens=greedy_tokens,
                )
                results.append(result)
                if cached_request_str is not None:
                    self.cache_hook.add_partial(
                        "loglikelihood", cached_request_str, result
                    )
                pbar.update(1)
        return results

    @staticmethod
    def _score_cached_loglikelihood(
        *,
        cached_cont_toks: list[int],
        cached_log_probs: torch.Tensor,
        greedy_tokens: torch.Tensor,
    ) -> tuple[float, bool]:
        target = torch.tensor(
            cached_cont_toks,
            device=cached_log_probs.device,
            dtype=torch.long,
        ).unsqueeze(0)
        greedy = bool((greedy_tokens[:, -target.shape[1] :] == target).all())
        token_log_probs = torch.gather(
            cached_log_probs, 2, target.unsqueeze(-1)
        ).squeeze(-1)
        return float(token_log_probs.sum().item()), greedy

    def _loglikelihood_tokens(
        self,
        requests: list[LoglikelihoodRequest],
        disable_tqdm: bool = False,
        **kwargs: Any,
    ) -> list[tuple[float, bool]]:
        if not requests and self.world_size <= 1:
            return []

        from lm_eval.models.utils import Collator

        batch_size = 1 if self.batch_size == "auto" else int(self.batch_size)
        reordered = Collator(
            requests,
            sort_fn=_loglikelihood_collate_key,
            group_by="contexts",
            group_fn=_loglikelihood_context_cache_key,
        )
        chunks = list(reordered.get_batched(n=max(batch_size, 1)))
        max_chunks = self._distributed_max_int(len(chunks))
        results: list[tuple[float, bool]] = []
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm or self.rank != 0,
            desc="Running loglikelihood requests",
        )

        for chunk in chunks:
            batch = self._prepare_loglikelihood_batch(chunk)
            logits = self._forward_logits(batch.input_ids, batch.attention_mask)
            results.extend(
                self._score_loglikelihood_batch(
                    reordered=reordered, batch=batch, logits=logits, pbar=pbar
                )
            )

        # FSDP ranks must enter the same number of model forwards. lm-eval pads
        # request counts, but context caching can collapse those into fewer chunks.
        for _ in range(max_chunks - len(chunks)):
            self._forward_dummy_loglikelihood_chunk()

        pbar.close()
        if len(results) != len(requests):
            raise RuntimeError("lm-eval loglikelihood produced incomplete results.")
        return reordered.get_original(results)

    def generate_until(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[str]:
        raise NotImplementedError(
            "REX lm-eval validation is intentionally limited to loglikelihood tasks."
        )

    def loglikelihood_rolling(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[float]:
        raise NotImplementedError(
            "REX lm-eval validation is intentionally limited to loglikelihood tasks."
        )

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
