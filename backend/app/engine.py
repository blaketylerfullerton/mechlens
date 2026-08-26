"""nnsight-backed model handle: loads a model once and exposes it for the
lens functions in app/lenses/ to trace, probe, and intervene on.

Assumes a Llama/Gemma-style HF architecture (`model.model.layers`,
`model.lm_head`) — GPT-2-style models expose these under different attribute
names (`model.transformer.h`, `model.lm_head`), so swap those paths in the
lens modules below if you point this at a different family.
"""

import torch
from nnsight import LanguageModel


class Engine:
    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False,
    ):
        # transformers no longer accepts a bare `load_in_4bit` kwarg — it has
        # to go through BitsAndBytesConfig, and only when actually enabled
        # (passing quantization_config=None is fine, passing load_in_4bit
        # directly raises a TypeError on current transformers).
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        self.model = LanguageModel(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            attn_implementation="eager",
            dispatch=True,
        )
        self.tokenizer = self.model.tokenizer
        self.n_layers = len(self.model.model.layers)
