"""Text-only Qwen3.5/Intern tokenizer adapter for unmodified ms-swift."""
from __future__ import annotations

from copy import deepcopy


class TextOnlyQwenTemplateMixin:
    def _encode(self, inputs):
        if any(getattr(inputs, key, None) for key in ("images", "videos", "audios", "objects")):
            raise ValueError("rex_qwen3_5_text only supports text-only inputs")
        return super()._encode(inputs)

    def _post_encode(self, model, inputs):
        # Intern's tokenizer has no image processor. The plain-text batch does
        # not need Qwen's vision post-encoding. Native Qwen processors retain
        # their upstream behavior.
        if not hasattr(self.processor, "image_processor"):
            if inputs.get("images") or inputs.get("videos"):
                raise ValueError("A tokenizer-only input cannot encode media")
            return inputs
        return super()._post_encode(model, inputs)


def install_qwen_text_adapter():
    from swift.template import TEMPLATE_MAPPING, register_template
    from swift.template.templates.qwen import Qwen3_5Template

    name = "rex_qwen3_5_text"
    if name in TEMPLATE_MAPPING:
        return
    cls = type("RexQwenTextTemplate", (TextOnlyQwenTemplateMixin, Qwen3_5Template),
               {"__module__": __name__})
    globals()[cls.__name__] = cls
    meta = deepcopy(TEMPLATE_MAPPING["qwen3_5"])
    meta.template_type, meta.template_cls = name, cls
    register_template(meta)
