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
    attention: list[list[list[list[float]]]]    # [layer][head][query][key]
    logit_lens: list[list[dict]]                 # [layer][top-k] -> {token, prob}, taken at the last position


@dataclass
class JacobianLensResult:
    prompt: str
    tokens: list[str]
    target_token: str
    target_token_id: int
    # [layer] -> {grad_norm, top_aligned_tokens: [{token, score}]}, taken at the last position
    layers: list[dict]


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

    def generate(self, prompt: str, max_new_tokens: int = 30) -> str:
        with self.model.generate(prompt, max_new_tokens=max_new_tokens):
            out = self.model.generator.output.save()
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def trace(self, prompt: str, top_k: int = 5) -> TraceResult:
        tokens = self.tokenizer.tokenize(prompt)

        # Declared *before* the trace context: nnsight defers/executes the
        # `with` block's body separately from this frame, so assigning these
        # lists inside it would never propagate back out (UnboundLocalError
        # on exit). Mutating them via .append() from inside works fine since
        # it's a closure over the same list object, not a reassignment.
        hidden_states, attentions, logit_lens = [], [], []

        with self.model.trace(prompt):
            for layer in self.model.model.layers:
                # self_attn.output must be requested before layer.output — nnsight
                # schedules saves in the order they're referenced, and self_attn
                # runs before the parent layer finishes, so asking for it after
                # layer.output raises MissedProviderError ("out of order").
                #
                # attn_implementation="eager" (forced in __init__) means every
                # Llama/Gemma-style self_attn returns (attn_output, attn_weights, ...),
                # so index 1 is always populated here.
                attentions.append(layer.self_attn.output[1].save())

                hs = layer.output[0].save()
                hidden_states.append(hs)

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
            # attn_weights keeps its (batch, head, query, key) shape even
            # though hidden_states above already comes back batch-less on
            # this transformers version — drop the batch=1 dim so the actual
            # output matches the [layer][head][query][key] shape promised above.
            attention=[a[0].float().cpu().tolist() for a in attentions],
            logit_lens=[self._topk(logits, top_k) for logits in logit_lens],
        )

    def jacobian_lens(self, prompt: str, target_token: str | None = None, top_k: int = 5) -> JacobianLensResult:
        """Backward-pass complement to the logit lens: instead of decoding what a
        layer's hidden state currently means (static projection through the final
        norm + unembedding), decode what direction in that hidden state would most
        increase a target token's logit (the local Jacobian, taken via backward-mode
        autodiff). Per layer this surfaces which vocab directions are causally "on
        the path" to the prediction, rather than which vocab entry it already looks
        like.

        Only the last sequence position is used, on both ends: the target logit is
        read at the last position (next-token prediction), and only that position's
        gradient is kept per layer (matches the logit_lens convention elsewhere in
        this module).
        """
        tokens = self.tokenizer.tokenize(prompt)

        target_id = None
        if target_token is not None:
            ids = self.tokenizer.encode(target_token, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"target_token {target_token!r} does not map to a single token")
            target_id = ids[0]

        # This runs a raw forward pass on the wrapped transformers model instead
        # of going through nnsight's trace/.grad machinery: nnsight 0.7's `.grad`
        # proxy (its documented pattern for reading gradients off a non-leaf
        # tensor mid-trace — see nnsight's own
        # modeling/vllm/intervention-gaps/test_4_1.py, which exists specifically
        # to probe whether that pattern still holds) reliably raises
        # MissedProviderError against this model/version, on both a `.save()`d
        # and a live hidden-state proxy. Plain torch autograd on
        # `self.model._model` (the actual LlamaForCausalLM nnsight wraps) has no
        # such issue, so the Jacobian is computed there directly.
        real_model = self.model._model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(real_model.device)

        outputs = real_model(**inputs, output_hidden_states=True)
        # hidden_states[0] is the embedding output; [1:] is one entry per
        # transformer layer, matching trace()'s per-layer convention.
        hidden_states = list(outputs.hidden_states[1:])
        for hs in hidden_states:
            hs.retain_grad()

        last_logits = outputs.logits[0, -1, :]
        if target_id is None:
            target_id = int(last_logits.argmax().item())

        real_model.zero_grad(set_to_none=True)
        last_logits[target_id].backward()

        unembed = real_model.lm_head.weight.detach().float().cpu()

        layers = []
        for hs in hidden_states:
            grad = hs.grad[0, -1, :].float().cpu()
            scores = unembed @ grad
            top = torch.topk(scores, top_k)
            layers.append(
                {
                    "grad_norm": grad.norm().item(),
                    "top_aligned_tokens": [
                        {"token": self.tokenizer.decode([idx]), "score": score}
                        for idx, score in zip(top.indices.tolist(), top.values.tolist())
                    ],
                }
            )

        return JacobianLensResult(
            prompt=prompt,
            tokens=tokens,
            target_token=self.tokenizer.decode([target_id]),
            target_token_id=target_id,
            layers=layers,
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
            # o_proj's input is (..., hidden_size) — batch/seq dims vary (e.g.
            # flattened to (seq_len, hidden) during generate on some transformers
            # versions), so index only the last dim rather than assuming a rank.
            o_proj_input[..., lo:hi] = 0
            out = self.model.generator.output.save()

        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def _topk(self, logits, k: int) -> list[dict]:
        # logits can come back as (batch, seq, vocab) or (seq, vocab) depending
        # on the call site (some transformers versions drop the batch dim for
        # a manually-invoked submodule) — `...` absorbs whichever leading dims
        # are present so this always lands on the last token's vocab vector.
        probs = torch.softmax(logits[..., -1, :].float(), dim=-1)
        top = torch.topk(probs, k)
        return [
            {"token": self.tokenizer.decode([idx]), "prob": prob}
            for idx, prob in zip(top.indices.tolist(), top.values.tolist())
        ]

    def _decode_argmax(self, logits) -> str:
        return self.tokenizer.decode([logits[..., -1, :].argmax().item()])
