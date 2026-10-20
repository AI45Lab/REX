"""Text-only Gemma 4 packing adapter for the repository's Swift plugin."""
from __future__ import annotations

from copy import copy, deepcopy
from functools import wraps


class TextOnlyGemmaTemplateMixin:
    # Swift defaults multimodal templates to unsupported, including text-only
    # Gemma. Keep its native encoder/BOS/thinking format and reject media here.
    support_padding_free = True

    def _encode(self, inputs):
        if any(getattr(inputs, key, None) for key in ("images", "videos", "audios", "objects")):
            raise ValueError("rex_gemma4_text only supports text-only inputs")
        return super()._encode(inputs)


class GemmaToolResponseMixin:
    def _format_tool_responses(self, assistant_content, tool_messages):
        # Swift merges consecutive assistant/tool-call turns into segments with
        # parallel loss/loss_scale lists. Keep those boundaries; Gemma's native
        # response formatter only needs to trim the final segment's opener.
        if isinstance(assistant_content, list):
            if not assistant_content or not all(isinstance(x, str) for x in assistant_content):
                raise ValueError("Expected nonempty text segments before Gemma tool responses")
            segments = assistant_content.copy()
            segments[-1], response = super()._format_tool_responses(segments[-1], tool_messages)
            return segments, response
        return super()._format_tool_responses(assistant_content, tool_messages)


def text_only_gemma_config(config):
    """Use native causal/sliding THD attention without altering export config.

    The bridge's vision branch requires a dense attention mask, which Swift's
    packed batches do not carry. For pure text, its vision mask is exactly the
    ordinary causal/sliding mask. The native TE backend supports THD and obtains
    sample boundaries from cu_seqlens. Only the language module receives this
    private config; the enclosing model/bridge retains the original HF config.
    """
    if not config.language_model_only:
        return config
    result = copy(config)
    result.hf_config = deepcopy(config.hf_config)
    text = result.hf_config.text_config
    if text.use_bidirectional_attention not in (None, "vision"):
        raise ValueError("Unsupported Gemma bidirectional attention mode for text-only packing")
    text.use_bidirectional_attention = None
    return result


def check_gemma_padding_free(original, args, config):
    # Swift normally disables THD for TE unfused. Our packed kernel is a
    # separate implementation; retain THD only for this exact plugin path.
    if (getattr(args, "template", None) == "rex_gemma4_text"
            and args.padding_free and args.language_model_only
            and config.attention_backend.name == "unfused"):
        if config.context_parallel_size != 1 or config.attention_dropout != 0:
            raise ValueError("Packed text Gemma requires CP=1 and attention_dropout=0")
        return
    return original(args, config)


def install_gemma_text_adapter():
    # Defer external/GPU imports so config isolation can be checked on CPU.
    from mcore_bridge.model.mm_gpts.gemma4 import Gemma4SelfAttention, Gemma4TextGPTModel
    from sft.gemma_attention import gemma_packed_core_forward
    from swift.template import TEMPLATE_MAPPING, register_template
    from swift.template.templates.gemma import Gemma4Template
    from swift.agent_template import agent_template_map
    from swift.megatron.model import utils as model_utils

    padding_check = model_utils._check_padding_free
    if not getattr(padding_check, "_rex_packed", False):
        @wraps(padding_check)
        def check_padding(args, config):
            return check_gemma_padding_free(padding_check, args, config)

        check_padding._rex_packed = True
        model_utils._check_padding_free = check_padding

    name = "rex_gemma4_text"
    if name not in TEMPLATE_MAPPING:
        agent_cls = type(
            "RexGemma4TextAgent", (GemmaToolResponseMixin, agent_template_map["gemma4"]),
            {"__module__": __name__},
        )
        globals()[agent_cls.__name__] = agent_cls
        agent_template_map[name] = agent_cls
        template_cls = type(
            "RexGemma4TextTemplate", (TextOnlyGemmaTemplateMixin, Gemma4Template),
            {"__module__": __name__},
        )
        globals()[template_cls.__name__] = template_cls
        meta = deepcopy(TEMPLATE_MAPPING["gemma4"])
        meta.template_type, meta.template_cls = name, template_cls
        meta.agent_template = name
        register_template(meta)

    core_forward = Gemma4SelfAttention._forward_core_attention
    if not getattr(core_forward, "_rex_packed", False):
        @wraps(core_forward)
        def packed_forward(self, *args, **kwargs):
            return gemma_packed_core_forward(core_forward, self, *args, **kwargs)

        packed_forward._rex_packed = True
        Gemma4SelfAttention._forward_core_attention = packed_forward

    original = Gemma4TextGPTModel.__init__
    if getattr(original, "_rex_text_only", False):
        return

    @wraps(original)
    def initialize(self, config, *args, **kwargs):
        original(self, text_only_gemma_config(config), *args, **kwargs)

    initialize._rex_text_only = True
    Gemma4TextGPTModel.__init__ = initialize
