Five steps turn a prompt into a picture you can read. Each one is a definition
you can check, so each is written here as the formula the code actually
computes — not a gloss of it.

Every number on this page was measured on `gemma-2-2b` with the Gemma Scope 16k
SAEs. Where a step has a way of failing quietly, the measurement that catches it
is stated with it.

---

## 01 · Get the activations

A transformer does not replace its state at each layer. It *adds* to it. For
layer $\ell$ at one token position, the residual stream is

$$
x_\ell \;=\; x_{\ell-1} \;+\; \mathrm{Attn}_\ell(x_{\ell-1}) \;+\; \mathrm{MLP}_\ell(x_{\ell-1})
$$

That identity is the whole reason this project needs no gradients and no
patching to decompose a layer: the parts are already separate terms in a sum.
Attribution is that equation read backwards, and it is checked against the
captured tensor rather than assumed —

$$
\max_{\ell,\,p}\;\frac{\lVert (x_{\ell-1} + a_\ell + m_\ell) - x_\ell \rVert}{\lVert x_\ell \rVert}\;\le\;1.3\times 10^{-2}
$$

a maximum over $26 \times 31$ positions of bfloat16 rounding. A real bug moves
the *mean*, which sits at $\approx 0.004$ and is flat with depth.

We capture $x_\ell$ at `hook_resid_post` for every layer and every token:

$$
X \in \mathbb{R}^{\,n_{\text{tokens}} \times 26 \times 2304}
$$

Each of those $2304$ numbers is a little bit of everything. That is the problem
the next step exists to solve.

---

## 02 · Sparsify with an SAE

A sparse autoencoder is a **wider, emptier** basis for the same vector. Encode
into $d_{\text{sae}} = 16{,}384$ dimensions, then decode back into $2{,}304$:

$$
z \;=\; W_{\text{enc}}\,x + b_{\text{enc}}, \qquad
\hat{x} \;=\; W_{\text{dec}}\,a + b_{\text{dec}}
$$

The widening is useless on its own — an invertible map to a bigger space
explains nothing. The sparsity is the entire idea, and Gemma Scope enforces it
with a **JumpReLU**: a per-feature learned threshold $\theta_i$ below which the
feature is not merely small but exactly zero.

$$
a_i \;=\; z_i \cdot H(z_i - \theta_i), \qquad
H(u) = \begin{cases} 1 & u > 0 \\ 0 & u \le 0 \end{cases}
$$

A plain ReLU would leave a long tail of tiny activations, and $\lVert a
\rVert_0$ would be meaningless. The hard gate is what makes "this feature fired"
a statement rather than a threshold you picked afterwards.

Training minimises reconstruction error against a sparsity penalty, with the
model frozen throughout:

$$
\mathcal{L} \;=\; \underbrace{\lVert x - \hat{x} \rVert_2^2}_{\text{reconstruct}}
\;+\; \lambda \underbrace{\lVert a \rVert_0}_{\text{stay empty}}
$$

On the `golden-gate` trace at layer 20, token `' Bridge'`, this leaves

$$
\lVert a \rVert_0 \;=\; 83 \quad \text{of} \quad 16{,}384 \qquad (0.5\%)
$$

and those 83 are separable enough to read: `#3124` *San Francisco, Oakland, Bay
Area* at $104.65$, `#1370` *bridge and crossing* at $66.19$. The model is
holding both apart inside one vector, at one instant.

> **What the trace keeps.** Only the strongest $k = 16$ of those 83 are saved
> per cell. The interface reports both numbers, because the lit features are not
> the complete set.

---

## 03 · Check it didn't lie to you

This is the step the whole thing turns on. A badly-fit SAE still produces
confident, plausible, readable features — it simply produces the *wrong* ones.
So three numbers, and none of them is optional.

**Explained variance.** How much of the original vector survives the round trip:

$$
\mathrm{EV} \;=\; 1 - \frac{\operatorname{Var}(x - \hat{x})}{\operatorname{Var}(x)}
$$

Measured across all 26 layers: mean $0.887$, worst layer $0.849$.

**Sparsity.** How many features actually fire, averaged over tokens:

$$
L_0 \;=\; \frac{1}{N}\sum_{t=1}^{N} \bigl|\{\, i : a_i^{(t)} > 0 \,\}\bigr| \;=\; 68.9
$$

The failure modes are loud on purpose. $L_0 = 16{,}384$ means the JumpReLU
threshold was skipped. $L_0 = 0$ means the wrong activation site.

**Loss recovered.** The other two describe the *reconstruction*. Neither says
the model can still use it. So substitute $\hat{x}$ back in at the hook and
measure what the model does downstream:

$$
\mathrm{recovered} \;=\; \frac{\mathcal{L}_{\text{ablate}} - \mathcal{L}_{\text{recon}}}{\mathcal{L}_{\text{ablate}} - \mathcal{L}_{\text{clean}}}
$$

where $\mathcal{L}_{\text{ablate}}$ zeroes the stream entirely. $1.0$ is a
perfect substitute; $0.0$ is no better than deleting the layer. It is **not**
clipped, and it is undefined when ablation fails to increase loss — a suspicious
result should read as suspicious rather than be rounded into range.

**The control.** Any substitution harness can measure itself by accident.
Substituting $x$ for $x$ through the same hook and the same token mask must
change nothing:

$$
\bigl|\mathcal{L}_{\text{identity}} - \mathcal{L}_{\text{clean}}\bigr| \;\le\; 10^{-6}
$$

**The round trip.** Save the dictionary, reload it, and require *exact*
equality on held-out activations — not approximate:

$$
\mathrm{encode}_{\text{saved}}(x) \;=\; \mathrm{encode}_{\text{loaded}}(x)
$$

---

## 04 · Label the features

Feature `#3124` means nothing until it is named. Two sources, and they are not
equally trustworthy.

For Gemma Scope, labels come from Neuronpedia's public export — $425{,}679$ of
them, read from local SQLite, no network. For an SAE you trained yourself, no
label exists anywhere, so the workspace collects each feature's top-activating
examples

$$
E_i \;=\; \operatorname*{arg\,top\text{-}k}_{t \,\in\, \text{corpus}} \; a_i^{(t)}
$$

and hands them to a local language model, which answers into a fixed schema:
`label`, `summary`, `evidence`, **`uncertainty`**.

That last field is load-bearing. A label is one model's guess about another
model's feature — the weakest claim anywhere in this project — and it is the one
step with no formula above it. Treat it accordingly.

> **A caveat worth carrying.** Neuronpedia's export does not use one explainer.
> Layers 16, 18, 20, 22 and 24 carry `gemini-2.5-flash-lite` explanations; the
> other 21 carry `gpt-4o-mini`. They write in visibly different styles, so every
> label records which wrote it.

---

## 05 · Place them in space

A trace says which features fired. It does not say where to *draw* them. So
every feature gets one position, computed once, offline, by UMAP — from $256$
dimensions of label-embedding down to $3$.

Flattening that far throws away a great deal, and the value of the whole picture
rests on how much survived. So that is measured, not assumed. For a sample of
features $S$, with $N_k$ meaning "the $k$ nearest neighbours of":

$$
P_k \;=\; \frac{1}{|S|}\sum_{f \in S} \frac{\bigl| N_k^{\text{source}}(f) \;\cap\; N_k^{\text{layout}}(f) \bigr|}{k}
$$

At $k = 20$ over $4{,}000$ sampled features: $P_{20} = 0.292$, against $0.0002$
for a random layout of the same $98{,}210$ features.

A region earns a name only when its members' label agreement beats a **random
group of the same size**:

$$
\mathrm{coh}(C) - \mathrm{coh}_{\text{random}}(|C|) \;\ge\; 0.150
$$

measured at $0.465$ against a baseline of $0.233$ — a margin of $0.232$. The
relative test replaced an absolute threshold, and that was not cosmetic: an
early pilot cleared an absolute bar by $0.049$ and named 14 clusters that read
plausibly. Explanation embeddings from a single explainer share a similarity
floor well above zero, so an absolute threshold measures the floor, not any
agreement.

Under the relative gate, the decoder-source atlas names **none** of its 12
clusters. That is the correct answer: in $2{,}304$ dimensions HDBSCAN calls
$99.1\%$ of features noise. It is a continuum, not an archipelago.

> **Near means similar. Far means nothing.** UMAP preserves local neighbourhoods
> and distorts everything larger. So the view ships with no axes, no coordinate
> readout and no distance scale — there is no honest way to answer a question
> about distance here, so the question is not offered.

---

## The step that is still missing

The five steps above describe the trace viewer, which reads a *pretrained*
dictionary. The training workspace runs steps 2, 3 and 4 on an SAE you fit
yourself — but it cannot yet hand that dictionary back to step 5. A trained SAE
gets no atlas position and no attribution, because the trace viewer loads Gemma
Scope and nothing else.

Closing that is the next real piece of work.
