"""Plain sampling through the loaded model, no instrumentation."""


def generate(engine, prompt: str, max_new_tokens: int = 30) -> str:
    with engine.model.generate(prompt, max_new_tokens=max_new_tokens):
        out = engine.model.generator.output.save()
    return engine.tokenizer.decode(out[0], skip_special_tokens=True)
