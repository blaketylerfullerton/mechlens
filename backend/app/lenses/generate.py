"""Plain sampling through the loaded model, no instrumentation."""

import re


def generate(engine, prompt: str, max_new_tokens: int = 30) -> str:
    with engine.model.generate(prompt, max_new_tokens=max_new_tokens):
        out = engine.model.generator.output.save()
    return engine.tokenizer.decode(out[0], skip_special_tokens=True)


def build_prompt(engine, messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Turn-formats `messages` through the tokenizer's own chat template
    when the loaded model has one; falls back to a plain
    "User: .../Assistant:" transcript for models with none.

    `add_generation_prompt` appends the cue for whoever speaks next (e.g.
    "<|im_start|>assistant\\n") — set it when the result is about to be
    handed to generate(); leave it off to represent the conversation
    exactly as far as it's gotten (e.g. for lens analysis right after a
    reply, where there's nothing pending to generate yet).
    """
    if engine.tokenizer.chat_template is not None:
        return engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    prompt = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages
    )
    return f"{prompt}\nAssistant:" if add_generation_prompt else prompt


def chat(engine, messages: list[dict], max_new_tokens: int = 60) -> dict:
    """Generates a reply to `messages`. Base models with no chat template
    (e.g. SmolLM2-135M without the -Instruct suffix) have no real
    end-of-turn token to stop generation on, so the reply is additionally
    cut at the next turn marker or paragraph break to avoid returning a
    page of repetition.

    Returns {prompt, text}: `prompt` is the exact string handed to
    generate(), so a caller can feed the same prompt into another lens
    (e.g. jacobian_lens) to explain the reply that was actually produced.
    """
    has_template = engine.tokenizer.chat_template is not None
    prompt = build_prompt(engine, messages, add_generation_prompt=True)
    prompt_len = engine.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

    with engine.model.generate(prompt, max_new_tokens=max_new_tokens):
        out = engine.model.generator.output.save()

    # Slicing on token count (rather than string-matching the prompt back
    # out of the decoded output) is exact regardless of any whitespace
    # normalization decode() applies.
    reply = engine.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    if not has_template:
        turn_marker = re.search(r"\n\s*user:", reply, re.IGNORECASE)
        candidates = [i for i in (turn_marker.start() if turn_marker else -1, reply.find("\n\n")) if i != -1]
        if candidates:
            reply = reply[: min(candidates)].strip()

    return {"prompt": prompt, "text": reply}
