# CLIP Reproduction and Investigation

A research-oriented reproduction of **CLIP: Learning Transferable Visual Models From Natural Language Supervision** by Radford et al.

The goal of this project is not to reproduce CLIP's original 400M-pair training run or match its headline numbers. Instead, the project focuses on understanding and reproducing the **scientific mechanism** behind CLIP, validating the implementation through controlled experiments, and eventually extending the reproduction into a small research investigation.

## Research Goals

This project follows the progression:

**understand → implement → validate → reproduce → investigate**

The main questions are:

1. How does CLIP's symmetric image-text contrastive objective work?
2. Can a minimal implementation reproduce the expected alignment behavior?
3. Can a small CLIP-style model learn useful image-text representations?
4. Can pretrained CLIP reproduce zero-shot classification and prompt-engineering effects?
5. How does CLIP's image-text alignment behave under image distribution shift?

The emphasis is on distinguishing:

* implementation correctness,
* reproduction of a reported phenomenon,
* empirical observations,
* and scientific conclusions.

---

## Core CLIP Mechanism

Given a batch of aligned image-text pairs:

```math
(x_1,t_1),\ldots,(x_B,t_B)
```

CLIP encodes images and text into a shared embedding space and L2-normalizes the resulting vectors.

The pairwise similarity matrix is:

```math
S = \hat{I}\hat{T}^{\top}
```

where:

```math
S_{ij}
```

represents the cosine similarity between image `i` and text `j`.

For correctly ordered image-text pairs, the positive examples lie along the diagonal:

```text
             Text
          T0  T1  T2  T3
       ┌─────────────────
I0     │  ✓
I1     │      ✓
I2     │          ✓
I3     │              ✓
       └─────────────────
 Image
```

CLIP treats alignment as two classification problems:

```text
image → text
text  → image
```

For image-to-text matching, every image must identify its corresponding text among all texts in the batch.

For text-to-image matching, every text must identify its corresponding image.

The final objective is the symmetric average:

```math
L_{\mathrm{CLIP}}
=
\frac{
L_{I\rightarrow T}
+
L_{T\rightarrow I}
}{2}
```

A learned temperature/logit-scale parameter controls the sharpness of the resulting probability distribution.

If `s_ij` is the cosine similarity between image `i` and text `j`, CLIP uses scaled logits of the form:

```math
\mathrm{logit}_{ij}
=
\exp(t)\,s_{ij}
```

where `t` is a learned log-scale parameter.

---

## Repository Structure

```text
clip-reproduction/
│
├── README.md
├── pyproject.toml
├── uv.lock
│
├── configs/
│
├── src/
│   └── clip_repro/
│       ├── models/
│       ├── losses/
│       │   └── clip_loss.py
│       ├── data/
│       ├── evaluation/
│       └── utils/
│
├── scripts/
│
├── tests/
│   └── test_clip_loss.py
│
├── notebooks/
│   └── 01_clip_objective.ipynb
│
├── experiments/
│
├── data/
│   └── README.md
│
└── outputs/
```

### Directory Roles

**`src/`**
Contains reusable implementations and is the source of truth for model and loss code.

**`tests/`**
Checks scientific and software invariants of the implementation.

**`scripts/`**
Contains executable training and evaluation experiments.

**`configs/`**
Stores explicit experimental variables and interventions.

**`notebooks/`**
Used for interactive reasoning, visualization, and inspection. Notebooks are not the source of truth for implementations.

**`experiments/`**
Records research questions, hypotheses, predictions, controls, results, and interpretations.

**`outputs/`**
Contains generated checkpoints, logs, metrics, figures, and other run artifacts. These are not committed to Git.

---

## Environment Setup

This project uses `uv` for dependency and environment management.

Install dependencies:

```bash
uv sync
```

Run Python inside the project environment:

```bash
uv run python
```

Run tests:

```bash
uv run pytest -v
```

Launch Jupyter:

```bash
uv run jupyter lab
```

---

## Current Experiment

### Experiment 001 — CLIP Loss Sanity Check

#### Question

Does our implementation of the symmetric CLIP objective behave as expected?

#### Hypothesis

Correctly aligned image-text pairs should produce substantially lower contrastive loss than incorrectly paired embeddings.

#### Method

We constructed small synthetic image and text embedding matrices with known geometric relationships.

For aligned pairs, the cosine-similarity matrix placed high values along the diagonal.

Observed aligned loss:

```text
0.03048
```

We then shuffled the text embeddings while keeping the target correspondence unchanged.

Observed shuffled loss:

```text
10.73849
```

For comparison, a uniform prediction over four candidates would have cross-entropy:

```math
-\log\left(\frac{1}{4}\right)
=
\log(4)
\approx
1.386
```

The shuffled loss being much larger than this indicates that the model is not merely uncertain. The embedding geometry is often **confidently aligned with the wrong target**, which the temperature-scaled cross-entropy strongly penalizes.

### What This Establishes

This result is consistent with the intended CLIP objective: the loss is sensitive to correct image-text correspondence.

### What This Does Not Establish

It does not yet prove that the implementation is fully correct.

A single toy example could still pass despite a subtle bug. Additional tests should examine invariants such as:

* image-text / text-image symmetry,
* permutation behavior,
* gradient flow,
* normalization behavior,
* temperature effects,
* and behavior under perfect or random alignment.

---

## Planned Progression

### Phase 1 — Objective

* [x] Derive the CLIP similarity matrix
* [x] Understand L2 normalization
* [x] Understand temperature scaling
* [x] Implement symmetric contrastive loss
* [x] Compare aligned and shuffled embeddings
* [ ] Add stronger unit tests

### Phase 2 — Minimal CLIP

* [ ] Implement a small image encoder
* [ ] Implement a small text encoder
* [ ] Add learned projection heads
* [ ] Train on a small controlled dataset
* [ ] Observe whether diagonal image-text alignment emerges

### Phase 3 — Zero-Shot CLIP

* [ ] Load a pretrained CLIP model
* [ ] Reproduce zero-shot classification
* [ ] Compare class names against natural-language prompts
* [ ] Investigate prompt ensembles

### Phase 4 — Small-Scale Reproduction

* [ ] Train on real image-caption pairs
* [ ] Evaluate image-to-text retrieval
* [ ] Evaluate text-to-image retrieval
* [ ] Evaluate transfer behavior

### Phase 5 — Research Extension

Investigate how image-text alignment changes under controlled distribution shifts such as:

* blur,
* noise,
* occlusion,
* contrast changes,
* other visual corruptions.

The goal will be to distinguish changes in:

```text
visual representation
        ↓
image-text alignment
        ↓
downstream prediction
```

rather than treating downstream accuracy alone as evidence about the underlying mechanism.

---

## Experimental Philosophy

Every experiment should specify:

**Question** — What are we trying to understand?

**Hypothesis** — What do we believe and why?

**Prediction** — What should happen if the hypothesis is true?

**Alternatives** — What other explanations could produce the result?

**Controls** — What variables must remain fixed?

**Intervention** — What variable are we changing?

**Measurement** — What evidence answers the question?

**Falsification** — What result would weaken the hypothesis?

**Interpretation** — What conclusion is actually justified?

The goal is not merely to make CLIP run.

The goal is to understand **why it behaves the way it does and what evidence is sufficient to support claims about that behavior**.

---

## Reference

Radford, A. et al.
**Learning Transferable Visual Models From Natural Language Supervision.**
2021. arXiv:2103.00020.

