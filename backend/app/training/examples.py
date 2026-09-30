"""Bounded, checkpoint-scoped high-activation examples for trained SAEs."""
from __future__ import annotations

import heapq
import json
import os
import time
from pathlib import Path

import torch

from .runner import TrainingConfig, text_sources


def token_window(model, chunk, activations, peak: int, chunk_start: int) -> dict:
    """Bounded evidence from the exact sequence seen by the model (32 tokens either side)."""
    left, right = max(0, peak - 32), min(chunk.shape[1], peak + 33)
    ids = chunk[0, left:right].tolist()
    return dict(token_ids=ids, tokens=[model.tokenizer.decode([token], clean_up_tokenization_spaces=False) for token in ids],
                activations=activations[left:right].tolist(), peak_index=peak - left,
                start_position=chunk_start + left, chunk_start_position=chunk_start)


@torch.no_grad()
def collect_feature_examples(model, sae, manifest: dict, config: TrainingConfig, feature_ids: list[int],
                             max_examples: int, max_sequences: int, check=lambda: None,
                             progress=lambda done, total: None) -> dict:
    """Return strongest positive activations, retaining only bounded evidence."""
    heaps: dict[int, list[tuple[float, int, dict]]] = {feature: [] for feature in feature_ids}
    _, texts, dataset = text_sources(config, manifest["dataset"])
    hook = manifest["hook"]
    sequence_count = scanned_tokens = serial = 0
    for document_index, text in enumerate(texts):
        check()
        tokens = model.to_tokens(text, prepend_bos=True, truncate=False)
        for start in range(0, tokens.shape[1], config.context_size):
            check()
            chunk = tokens[:, start:start + config.context_size]
            if start == 0 and chunk.shape[1] <= 1:
                continue
            _, cache = model.run_with_cache(chunk, names_filter=[hook], stop_at_layer=manifest["layer"] + 1)
            acts = sae.encode(cache[hook][0].to(device=sae.W_dec.device, dtype=sae.W_dec.dtype)).float().cpu()
            sequence_count += 1
            first = 1 if start == 0 else 0  # Exclude BOS, not the first real token of later chunks.
            scanned_tokens += len(acts) - first
            for feature in feature_ids:
                activation, local_position = acts[first:, feature].max(dim=0)
                value = float(activation)
                if value <= 0:
                    continue
                peak = int(local_position) + first
                heap = heaps[feature]
                if len(heap) >= max_examples and value <= heap[0][0]:
                    continue
                window = token_window(model, chunk, acts[:, feature], peak, start)
                token_position = start + peak
                row = dict(feature_idx=feature, activation=value, document_index=document_index,
                           token_position=token_position, token_ids=chunk[0].tolist(),
                           context=model.tokenizer.decode(window["token_ids"], clean_up_tokenization_spaces=False),
                           token_window=window)
                item = (value, serial, row)
                serial += 1
                heap = heaps[feature]
                if len(heap) < max_examples:
                    heapq.heappush(heap, item)
                elif value > heap[0][0]:
                    heapq.heapreplace(heap, item)
            progress(sequence_count, max_sequences)
            if sequence_count >= max_sequences:
                break
        if sequence_count >= max_sequences:
            break
    if not sequence_count:
        raise ValueError("Example corpus produced no usable token sequences")
    return dict(schema_version=2, collected_at=time.time(), artifact_id=manifest["artifact_id"],
                layer=manifest["layer"], feature_count=manifest["features"], hook=hook,
                dataset=dataset, max_examples=max_examples, max_sequences=max_sequences,
                sequences_scanned=sequence_count, tokens_scanned=scanned_tokens,
                examples={str(feature): [row for _, _, row in sorted(heap, reverse=True)]
                          for feature, heap in heaps.items()},
                definition="Top positive activations, one candidate per feature per document chunk; exact token windows and per-token activations are retained. Tokens after the peak are following context, unseen at the peak.")


def save_feature_examples(root: Path, run_id: str, report: dict) -> Path:
    """Atomically publish an example report only after its scan succeeds."""
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"examples-{report['artifact_id'][:16]}.json"
    temporary = directory / f".{final.name}-{time.time_ns()}"
    temporary.write_text(json.dumps(report, allow_nan=False))
    os.replace(temporary, final)
    return final
