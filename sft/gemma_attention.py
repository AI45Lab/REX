"""Packed Gemma attention using PyTorch's CUDA memory-efficient kernel.

The public SDPA interface does not expose packed boundaries or a local window.
Keep the ATen forward/backward pair here; GPU checks must cover both whenever
the external Swift environment's PyTorch version changes. No dense fallback.
"""
from __future__ import annotations

import torch


class _PackedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, cu_q, cu_k, max_q, max_k, scale, window):
        output, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
            query, key, value, None, cu_q, cu_k, max_q, max_k,
            0.0, 1, True, scale=scale, window_size=window,
        )
        ctx.save_for_backward(query, key, value, output, lse, seed, offset, cu_q, cu_k)
        ctx.settings = max_q, max_k, scale, window
        return output

    @staticmethod
    def backward(ctx, grad):
        query, key, value, output, lse, seed, offset, cu_q, cu_k = ctx.saved_tensors
        max_q, max_k, scale, window = ctx.settings
        dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
            grad.contiguous(), query, key, value, None, output, cu_q, cu_k,
            max_q, max_k, lse, 0.0, seed, offset, 1, False,
            scale=scale, window_size=window,
        )
        return dq, dk, dv, None, None, None, None, None, None


def packed_gemma_attention(query, key, value, packed, *, scale, window=None):
    """THD self attention; window counts the current token (HF convention)."""
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("Expected THD query/key/value for packed Gemma attention")
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError("Packed Gemma attention requires self attention with equal head dimensions")
    if query.shape[1] % key.shape[1]:
        raise ValueError("Query heads must be divisible by KV heads")
    if window is not None and window <= 0:
        raise ValueError("Gemma sliding window must be positive")
    for suffix in ("q", "kv"):
        if getattr(packed, f"cu_seqlens_{suffix}_padded", None) is not None:
            raise ValueError("Packed Gemma attention does not support padded THD storage")
    # CUTLASS expects equal head counts. Repeat KV with autograd so the backward
    # sums grouped-query gradients back to the original KV heads.
    groups = query.shape[1] // key.shape[1]
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    output = _PackedAttention.apply(
        query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0),
        packed.cu_seqlens_q, packed.cu_seqlens_kv,
        packed.max_seqlen_q, packed.max_seqlen_kv, scale, window,
    )
    return output.reshape(query.shape[0], 1, -1)


def gemma_packed_core_forward(original, self, query, key, value, attention_mask,
                              attention_bias=None, packed_seq_params=None):
    """Use the packed kernel only for the plugin's text-only training path."""
    packed = packed_seq_params
    if self.config.language_model_only and self.training and (
            packed is None or packed.qkv_format != "thd"):
        raise ValueError("Gemma SFT requires packed THD boundaries; padding_free was disabled")
    if not self.config.language_model_only or packed is None or packed.qkv_format != "thd":
        return original(self, query, key, value, attention_mask,
                        attention_bias=attention_bias, packed_seq_params=packed)
    if attention_mask is not None or attention_bias is not None:
        raise ValueError("Packed text Gemma expects implicit causal/sliding masks only")
    if self.config.context_parallel_size != 1 or self.config.attention_dropout != 0:
        raise ValueError("Packed text Gemma requires CP=1 and attention_dropout=0")
    window = self.config.hf_config.text_config.sliding_window if self.is_sliding else None
    # Megatron passes config.softmax_scale to TE's constructor. TE stores the
    # resolved scale on its backend modules, not on TEDotProductAttention.
    # None retains both TE and ATen's default 1/sqrt(actual layer head_dim).
    return packed_gemma_attention(query, key, value, packed,
                                  scale=self.config.softmax_scale, window=window)
