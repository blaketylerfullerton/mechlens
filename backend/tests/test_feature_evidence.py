"""Exact evidence alignment without downloading a tokenizer or model."""
import json
from types import SimpleNamespace

import torch

from app.training.auto_interp import candidate_request
from app.training.examples import collect_feature_examples


def test_chunk_boundaries_and_token_alignment(monkeypatch):
    from app.training import examples
    pieces = ['<bos>', 'a', ' ', ';', 'Paris', '\n', 'é', '!']
    decode = lambda ids, **kw: ''.join(pieces[i] for i in ids)
    model = SimpleNamespace(
        to_tokens=lambda *a, **kw: torch.tensor([list(range(8))]),
        tokenizer=SimpleNamespace(decode=decode),
        run_with_cache=lambda chunk, **kw: (None, {'hook': chunk.float().unsqueeze(-1)}),
    )
    # Peak at first token of chunk two; previously this position was discarded.
    sae = SimpleNamespace(W_dec=torch.zeros(1), encode=lambda x: torch.where(x == 4, 20., x))
    monkeypatch.setattr(examples, 'text_sources', lambda *a: (None, ['ignored'], {}))
    report = collect_feature_examples(model, sae,
        {'dataset': {}, 'hook': 'hook', 'layer': 0, 'artifact_id': 'abc', 'features': 1},
        SimpleNamespace(context_size=4), [0], 2, 2)
    assert report['schema_version'] == 2
    assert report['tokens_scanned'] == 7
    row, earlier = report['examples']['0']
    window = row['token_window']
    assert row['token_position'] == 4
    assert row['activation'] == 20
    assert row['context'] == 'Paris\né!'
    assert window['peak_index'] == 0
    assert window['start_position'] == window['chunk_start_position'] == 4
    assert window['tokens'] == ['Paris', '\n', 'é', '!']
    assert window['activations'] == [20, 5, 6, 7]
    assert earlier['token_window']['peak_index'] == 3
    request, fingerprint = candidate_request('abc', 0, [row])
    evidence = json.loads(request['messages'][0]['content'].split('Examples: ')[1])[0]
    assert evidence['activating_token'] == 'Paris'
    assert evidence['tokens_through_peak'] == ['Paris']
    assert evidence['following_tokens_unseen_at_peak'] == ['\n', 'é', '!']
    assert fingerprint != candidate_request('abc', 0, [earlier])[1]


def test_legacy_examples_remain_labelable():
    request, _ = candidate_request('abc', 2, [{'activation': 2, 'context': 'old text'}])
    evidence = json.loads(request['messages'][0]['content'].split('Examples: ')[1])[0]
    assert evidence['context'] == 'old text'
    assert evidence['alignment'] == 'legacy: activating token unknown'


def test_window_is_bounded_and_peak_stays_aligned():
    from app.training.examples import token_window
    model = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids, **kw: str(ids[0])))
    chunk = torch.arange(100).unsqueeze(0)
    activations = torch.arange(100).float()
    for peak in (0, 50, 99):
        window = token_window(model, chunk, activations, peak, 200)
        i = window['peak_index']
        assert len(window['tokens']) <= 65
        assert len(window['token_ids']) == len(window['tokens']) == len(window['activations'])
        assert window['token_ids'][i] == peak
        assert window['tokens'][i] == str(peak)
        assert window['activations'][i] == peak
        assert window['start_position'] + i == 200 + peak
