# Validate a training run on Spark

The Training page includes a small validation preset and a downloadable JSON
report. Training completion and correctness-check results are separate: a
completed run can contain failed checks, and passing checks does not establish
feature interpretability. Existing runs retain their old measurements; create a
new run to collect the new report fields.

## Local checks

With the backend environment activated, from the repository root:

```bash
python -m pytest backend/tests/test_training.py -q
```

These use a tiny CPU transformer and offline stand-ins for the Hugging Face
reference. No model downloads or real-model training are required. They cover
frozen base weights, optimization, persistence, resume, checkpoint identity,
correctness-check failures, evaluation controls, and report export.

## First run on Spark

1. Update and restart the backend and frontend with these changes.
2. In Training, load a supported model and wait for weights ready.
3. Select **Use small validation preset**. This chooses the middle layer,
   4,096 features, 32,768 training tokens, TinyStories, and 64 held-out sequences.
4. Enable **Check Hugging Face agreement** when ready. This loads another copy
   of the same model revision on the backend CPU, adding RAM use and runtime.
   It runs through the existing compute queue. The browser can remain on a Mac.
5. Start training. Inspect the validation report after completion. Download its
   JSON, open the saved dictionary in Explore on a fresh prompt, and reload the
   page to confirm the history and artifact remain available.

The preset is a workflow check, not a recommended production training budget.
Increase the evaluation sequence count and training budget for quality studies.
Do not compare runs evaluated on different data as if only training changed.

Alternatively, from `backend/` in the Spark Python environment after loading
the model in the app (choose a valid layer for that model):

```bash
python -m app.training.benchmark --layer 12 --tokens 32768 \
  --evaluation-sequences 64 --compare-huggingface \
  --output data/training-benchmark.json
```

The command submits a new run through the backend queue. Omit
`--compare-huggingface` for a run without the additional reference model.
The output includes the validation report, training metrics, configuration,
wall time, and backend resource snapshot. It does not certify quality.

## Reading the report

- **Hugging Face agreement:** four fixed prompts, identical input token IDs,
  matching tokenizer encoding, same model revision and dtype. Compares the
  final-position next-token distribution. Both maximum probability difference
  and total variation must be at most 1e-5 for float32 or 0.005 for reduced
  precision. These are recorded engineering tolerances, not a proof of model
  equivalence; examine differences on the target hardware before changing them.
  Top-1 agreement is diagnostic because near ties can change the argmax.
  Disabled checks and unknown model revisions are explicitly **not run**.
- **Saved checkpoint agreement:** exact encoding and decoding agreement on the
  first held-out sequence, before and after saving/loading the SAE.
- **Activation identity:** the same artifact loader used by Explore validates
  checkpoint content hash, model/tokenizer identity, dimensions, hook, and
  preprocessing. Unknown provenance remains unknown; this is not the HF
  numerical agreement check.
- **Unchanged-activation control:** substituting the original activations at
  the evaluation hook must preserve loss within an absolute tolerance of 1e-6.
- **Held-out quality:** corpus-wide explained variance, reconstruction MSE, L0,
  and token-weighted original / reconstruction / zero-ablation next-token loss.
  Lower loss increase means better behavior preservation. Loss recovered is
  `(zero loss - SAE loss) / (zero loss - original loss)`; it is not clipped and
  is undefined when ablation does not increase loss. Explained variance is
  undefined when the evaluated activations have zero variance.

Evaluation uses the beginning of the recorded validation split, or the user's
separate evaluation text. Counts refer to document chunks, not necessarily
independent documents. It stops at the requested sequence count or dataset end,
and reports actual sequence/token counts. Each chunk's first activation stays
unchanged in all interventions. Reconstruction metrics include the last token,
which has no next-token target in the language-model loss. Custom text is only
checked for exact equality, not partial train/evaluation overlap.

Reports are available at `GET /training/runs/{run_id}/report`. They include
provenance and identities, but omit the user's raw training and evaluation text.
A failed agreement check calls for investigation before interpreting features.
Low reconstruction error alone does not establish semantic or causal validity.
