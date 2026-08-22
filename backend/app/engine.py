"""nnsight-backed engine: load a model once, run traced forward passes, and
hand back JSON-serializable activations/attention/logit-lens data for the UI.

Assumes a Llama/Gemma-style HF architecture (`model.model.layers`,
`model.lm_head`) — GPT-2-style models expose these under different attribute
names (`model.transformer.h`, `model.lm_head`), so swap ARCH paths below if
you point this at a different family.

Attention weights only show up on `self_attn.output` when the model is loaded
with attn_implementation="eager" — sdpa/flash attention never materializes
the full attention matrix, so eager is required (and slower) whenever you
need attention patterns, not just hidden states.
"""

from dataclasses import dataclass

import torch
from nnsight import LanguageModel


@dataclass
class TraceResult:
    prompt: str
    tokens: list[str]
    predicted_next_token: str
    hidden_states: list[list[list[float]]]      # [layer][token][dim]
    attention: list[list[list[list[float]]]]    # [layer][head][query][key], [] where unavailable
    logit_lens: list[list[dict]]                 # [layer][top-k] -> {token, prob}, taken at the last position


class Engine:
    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False,
    ):
        self.model = LanguageModel(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            load_in_4bit=load_in_4bit,
            attn_implementation="eager",
            dispatch=True,
        )
        self.tokenizer = self.model.tokenizer
        self.n_layers = len(self.model.model.layers)

    def generate(self, prompt: str, max_new_tokens: int = 30) -> str:
        with self.model.generate(prompt, max_new_tokens=max_new_tokens):
            out = self.model.generator.output.save()
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def trace(self, prompt: str, top_k: int = 5) -> TraceResult:
        tokens = self.tokenizer.tokenize(prompt)

        with self.model.trace(prompt):
            hidden_states, attentions, logit_lens = [], [], []

            for layer in self.model.model.layers:
                hs = layer.output[0].save()
                hidden_states.append(hs)

                attn_weights = None
                if len(layer.self_attn.output) > 1:
                    attn_weights = layer.self_attn.output[1].save()
                attentions.append(attn_weights)

                # "if generation stopped here": push this layer's residual
                # stream through the final norm + unembedding early.
                normed = self.model.model.norm(hs)
                logit_lens.append(self.model.lm_head(normed).save())

            final_logits = self.model.lm_head.output.save()

        return TraceResult(
            prompt=prompt,
            tokens=tokens,
            predicted_next_token=self._decode_argmax(final_logits),
            hidden_states=[hs.float().cpu().tolist() for hs in hidden_states],
            attention=[a.float().cpu().tolist() if a is not None else [] for a in attentions],
            logit_lens=[self._topk(logits, top_k) for logits in logit_lens],
        )

    def ablate_head(self, prompt: str, layer_idx: int, head_idx: int, max_new_tokens: int = 30) -> str:
        """Zero one attention head's contribution before it's mixed back into
        the residual stream, then generate — a quick way to see what a head
        was doing by removing it."""
        n_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // n_heads
        lo, hi = head_idx * head_dim, (head_idx + 1) * head_dim

        with self.model.generate(prompt, max_new_tokens=max_new_tokens):
            o_proj_input = self.model.model.layers[layer_idx].self_attn.o_proj.input[0]
            o_proj_input[:, :, lo:hi] = 0
            out = self.model.generator.output.save()

        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def _topk(self, logits, k: int) -> list[dict]:
        probs = torch.softmax(logits[0, -1].float(), dim=-1)
        top = torch.topk(probs, k)
        return [
            {"token": self.tokenizer.decode([idx]), "prob": prob.item()}
            for idx, prob in zip(top.indices.tolist(), top.values.tolist())
        ]

    def _decode_argmax(self, logits) -> str:
        return self.tokenizer.decode([logits[0, -1].argmax().item()])
