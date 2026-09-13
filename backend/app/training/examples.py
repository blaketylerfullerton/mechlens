"""Bounded, checkpoint-scoped high-activation examples for trained SAEs."""
from __future__ import annotations

import heapq
import json
import os
import time
from pathlib import Path

import torch

from .runner import TrainingConfig, text_sources


def _context(text: str, token_position: int, token_count: int, limit: int = 600) -> str:
    """A bounded source-text window; token IDs preserve the exact position."""
    centre = int(len(text) * token_position / max(1, token_count))
    start = max(0, centre - limit // 2)
    return text[start:start + limit]


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
        for start in range(0, tokens.shape[1] - 1, config.context_size):
            check()
            chunk = tokens[:, start:start + config.context_size]
            if chunk.shape[1] <= 1:
                continue
            _, cache = model.run_with_cache(chunk, names_filter=[hook], stop_at_layer=manifest["layer"] + 1)
            acts = sae.encode(cache[hook][0, 1:].to(device=sae.W_dec.device, dtype=sae.W_dec.dtype)).float().cpu()
            sequence_count += 1
            scanned_tokens += len(acts)
            for feature in feature_ids:
                activation, local_position = acts[:, feature].max(dim=0)
                value = float(activation)
                if value <= 0:
                    continue
                token_position = start + int(local_position) + 1
                row = dict(feature_idx=feature, activation=value, document_index=document_index,
                           token_position=token_position, token_ids=chunk[0].tolist(),
                           context=_context(text, token_position, tokens.shape[1]))
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
    return dict(schema_version=1, collected_at=time.time(), artifact_id=manifest["artifact_id"],
                layer=manifest["layer"], feature_count=manifest["features"], hook=hook,
                dataset=dataset, max_examples=max_examples, max_sequences=max_sequences,
                sequences_scanned=sequence_count, tokens_scanned=scanned_tokens,
                examples={str(feature): [row for _, _, row in sorted(heap, reverse=True)]
                          for feature, heap in heaps.items()},
                definition="Top positive activations, one candidate per feature per document chunk; token IDs and bounded source context are retained.")


def save_feature_examples(root: Path, run_id: str, report: dict) -> Path:
    """Atomically publish an example report only after its scan succeeds."""
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"examples-{report['artifact_id'][:16]}.json"
    temporary = directory / f".{final.name}-{time.time_ns()}"
    temporary.write_text(json.dumps(report, allow_nan=False))
    os.replace(temporary, final)
    return final
