"""Run a trace and print it. No server, no CLI framework — just:

    cd backend
    python inspect_trace.py
    python inspect_trace.py "Some other prompt"

To look at a different layer/head or a different model, edit the constants
below and rerun.
"""

import math
import sys

import torch

from app.engine import Engine

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
TOP_K = 5
LAYER = None  # None -> middle layer
HEAD = 0

# Sequential teal ramp, light->dark, used for every heatmap below.
SEQ_RAMP = [
    (234, 246, 245),
    (191, 232, 228),
    (138, 211, 204),
    (79, 182, 172),
    (31, 143, 133),
    (11, 90, 84),
]
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def seq_color(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    n = len(SEQ_RAMP) - 1
    scaled = t * n
    i = min(n - 1, int(scaled))
    local_t = scaled - i
    c0, c1 = SEQ_RAMP[i], SEQ_RAMP[i + 1]
    return tuple(round(c0[k] + (c1[k] - c0[k]) * local_t) for k in range(3))


def cell(text: str, t: float, width: int) -> str:
    r, g, b = seq_color(t)
    fg = "38;2;20;20;20" if t <= 0.55 else "38;2;245;250;249"
    return f"\033[48;2;{r};{g};{b}m\033[{fg}m{text.center(width)}{RESET}"


def clean_tok(tok: str) -> str:
    return tok.replace("Ġ", "·").replace("Ċ", "\\n")


def print_header(engine, result):
    tokens = " ".join(f"[{clean_tok(t)}]" for t in result.tokens)
    print(f"\n{BOLD}TRACE{RESET}  {DIM}{MODEL}{RESET}")
    print(f"{tokens} -> {BOLD}{clean_tok(result.predicted_next_token)}{RESET}")
    print(f"{DIM}layers={engine.n_layers} heads={engine.model.config.num_attention_heads}{RESET}\n")


def print_logit_lens(result, bar_width: int = 24):
    top1_probs = [layer[0]["prob"] for layer in result.logit_lens]
    p_min, p_max = min(top1_probs), max(top1_probs)
    final_top1 = result.logit_lens[-1][0]["token"]
    first_match = next((i for i, l in enumerate(result.logit_lens) if l[0]["token"] == final_top1), None)

    print(f"{BOLD}LOGIT LENS{RESET} — top guess per layer, bar = confidence")
    for i, layer in enumerate(result.logit_lens):
        top = layer[0]
        t = (top["prob"] - p_min) / (p_max - p_min or 1)
        filled = round(t * bar_width)
        r, g, b = seq_color(t)
        bar = f"\033[38;2;{r};{g};{b}m{'█' * filled}{DIM}{'░' * (bar_width - filled)}{RESET}"
        marker = "  <- matches final output" if i == first_match else ""
        print(f"  L{i:<3}{bar}  {clean_tok(top['token']):<12}{top['prob']*100:5.1f}%{marker}")
    print(
        f"{DIM}confidence ranges {p_min*100:.0f}%-{p_max*100:.0f}% across layers; "
        f"final output first appears as top guess at layer {first_match}{RESET}\n"
    )


def print_attention(result, layer: int, head: int):
    matrix = result.attention[layer][head]
    tokens = [clean_tok(t) for t in result.tokens]
    col_w = max(4, max(len(t) for t in tokens) + 1)

    print(f"{BOLD}ATTENTION{RESET} — layer {layer}, head {head}")
    print(" " * (col_w + 1) + " ".join(t.rjust(col_w) for t in tokens))
    for qi, row in enumerate(matrix):
        cells = []
        for v in row:
            text = f"{v:.2f}" if v >= 0.005 else ""
            cells.append(cell(text, v, col_w))
        print(f"{DIM}{tokens[qi].rjust(col_w)}{RESET} " + " ".join(cells))
    print(f"{DIM}blank cells are exactly 0 — causal masking, a token never attends to a later one{RESET}\n")


def print_norms(result):
    spark_chars = " ▁▂▃▄▅▆▇█"
    print(f"{BOLD}RESIDUAL STREAM NORM{RESET} — L2 norm per token, sparkline across layers (log-scaled)")
    n_tokens = len(result.tokens)
    for ti in range(n_tokens):
        vecs = [layer[ti] for layer in result.hidden_states]
        norms = [sum(v * v for v in vec) ** 0.5 for vec in vecs]
        logs = [math.log10(n) for n in norms]
        lo, hi = min(logs), max(logs)
        spark = "".join(
            spark_chars[min(len(spark_chars) - 1, int((v - lo) / (hi - lo or 1) * (len(spark_chars) - 1)))]
            for v in logs
        )
        print(f"  {clean_tok(result.tokens[ti]):<10} min={min(norms):8.1f}  max={max(norms):9.1f}  {spark}")
    print()


if __name__ == "__main__":
    print(f"loading {MODEL}...")
    engine = Engine(MODEL, device_map="cpu", dtype=torch.float32)
    print(f"tracing '{PROMPT}'...")
    result = engine.trace(PROMPT, top_k=TOP_K)
    layer = LAYER if LAYER is not None else engine.n_layers // 2

    print_header(engine, result)
    print_logit_lens(result)
    print_attention(result, layer, HEAD)
    print_norms(result)
