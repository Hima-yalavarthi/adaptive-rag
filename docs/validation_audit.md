# Validation Audit

CAI 6826 · Adaptive RAG for Reliable Question Answering  
Hima Yalavarthi · 28 Oct 2026

Numbers below use the same 100 HotpotQA validation questions (first 100 of the
curated validation split), `llama3.2:3b`, temperature 0, distractor retrieval.
Intervals are **paired bootstrap 95% CIs over questions** (10,000 resamples,
seed 0), not over model seeds. There is **one run** per system because
temperature 0 is deterministic. Code: `evaluation/compare.py`.

**Decision rule:** I only claim a system beats a baseline if the 95% CI of the
difference **excludes 0**.

---

## Q1 · Claim

**Claim:** Adaptive / Verified RAG should give fewer **confidently wrong**
answers than Standard RAG (retrieve + generate), by checking evidence support
and abstaining when confidence is low.

**Data:** HotpotQA distractor, curated validation, n = 100 questions.

**Metric and number (reliability):**

Confidently wrong = **0.26 [0.18, 0.35]** · Δ vs. Standard RAG = **−0.22 [−0.32, −0.13]** · 1 run, n = 100

The CI of the difference **excludes 0**, so I can claim Verified RAG reduces
confident errors vs. Standard RAG on this development set.

I do **not** claim higher answer accuracy. Reranked RAG exact match is
0.44 [0.34, 0.54] vs. Standard 0.40 [0.30, 0.49]; Δ = **+0.04 [−0.04, +0.12]**,
which **includes 0**.

---

## Q2 · Split

| Split | Size | Use |
| --- | --- | --- |
| Train JSON | 2,000 | Index / development only. The LLM is **not** fine-tuned. |
| Validation Q1–100 | 100 | **Development results** reported so far (this audit). |
| Validation Q101–300 | 200 | **Tuning** (confidence threshold). Not used yet. |
| Validation Q301–500 | 200 | **Held-out** final comparison. Untouched. |

Train and validation example IDs do not overlap (see leakage test). Evaluation
searches only that example’s context (HotpotQA distractor), not the train index.

This split **does not yet fully support a final generalization claim**, because
the 100-question numbers are from the development slice, and the confidence
threshold (0.35) was chosen on those same questions. The held-out slice is
reserved for the Nov 4 checkpoint.

---

## Q3 · Leakage

Possible paths and what I checked:

1. **Train/test ID overlap.** Could the same HotpotQA example sit in both
   splits? **Checked:** `tests/test_splits.py` asserts train and validation
   share **no** example IDs (2,000 vs 500, overlap = 0).
2. **Eval files mixing train IDs.** **Checked:** the same test asserts
   `baseline_n100_k5.jsonl`, `reranked_n100_k5.jsonl`, and
   `verified_n100_k5.jsonl` IDs are a subset of validation and disjoint from
   train.
3. **Test-set reuse (peeking).** The 0.35 confidence threshold was chosen
   after looking at these 100 questions. That is **leakage into the reported
   number**. Mitigation: freeze the threshold on Q101–300, report on Q301–500.
4. **Gold supporting facts.** Used only to **score** retrieval/citations, not
   as input to the LLM.
5. **Distractor setting.** Each question retrieves from its own ~10-article
   context, which is easier than full-corpus search. I label results as
   distractor, not open-domain.

---

## Q4 · Metric and uncertainty

**Metrics.** Exact match and F1 (HotpotQA official normalization) for
correctness; **confidently wrong** (wrong and not abstaining) for reliability;
retrieval recall@5 for evidence.

**Baselines.** Standard RAG (no rerank, no verify). Also Reranked RAG.

**Unit of resampling.** One row per **question** (not per sentence). Paired
because every system answers the same 100 IDs.

**95% intervals and paired comparison** (`python -m evaluation.compare`):

| Comparison | Metric | System | Baseline | Δ [95% CI] | Excludes 0? |
| --- | --- | ---: | ---: | ---: | --- |
| Reranked vs Standard | Exact match | 0.44 | 0.40 | +0.04 [−0.04, +0.12] | No |
| Reranked vs Standard | Retrieval recall@5 | 0.72 | 0.60 | **+0.12 [+0.07, +0.18]** | **Yes** |
| Verified vs Standard | Confidently wrong | 0.26 | 0.48 | **−0.22 [−0.32, −0.13]** | **Yes** |
| Verified vs Standard | Exact match | 0.25 | 0.40 | −0.15 [−0.25, −0.05] | Yes (worse) |

McNemar on reranked vs standard EM: 11 fixed, 7 broken, exact p ≈ 0.48.

**Smallest gain that would matter:** about **10 fewer confident errors per 100
questions** (Δ ≤ −0.10 on confidently wrong). The verified-vs-standard drop
(−0.22) is larger than that, but it **costs EM** (−0.15), which is why retry
must come next instead of abstaining only.

I will recompute the same table on **held-out Q301–500 by 4 Nov 2026**, after
freezing the threshold on Q101–300.

---

## Q5 · Trust

Confidence is **not calibrated**. Mean confidence on Verified RAG is **0.78**
while exact match is **0.25**, so a high score does not mean the answer is
usually right.

The grader is an **NLI cross-encoder** (`cross-encoder/nli-deberta-v3-xsmall`),
not an LLM-as-judge and not a person. I have **not** validated it against human
“supported / not supported” labels. Temperature 0 makes generation repeatable;
it does not make confidence trustworthy.

---

## Q6 · After launch

**Monitor:** exact match / F1, abstain rate, confidently-wrong rate, false
abstentions, latency, retry count, NLI contradiction rate.

**What could fail:** over-abstain (user gets no answer) or under-abstain
(wrong answer looks final). Retrieval drift if the corpus changes. Ollama /
model version changes can move EM even at temperature 0.

**Who is harmed:** the **user** who trusts a wrong answer, or who cannot get
an answer when evidence actually exists.

---

## Q7 · Verdict

**Biggest gap:** development-set reuse (threshold + reported metrics on the
same 100 questions) and **uncalibrated** confidence.

**First fix, with date:** by **4 Nov 2026** (Progress Checkpoint), freeze the
confidence threshold on validation **Q101–300**, then report EM, confidently
wrong, and paired bootstrap CIs on held-out **Q301–500**. Do not retune on the
held-out slice. Adaptive retry (more candidates / query rewrite before
abstaining) is the product fix that should recover some of the EM lost to
verification.

---

## How to reproduce the intervals

```bash
python -m evaluation.compare \
  evaluation/results/baseline_n100_k5.jsonl \
  evaluation/results/reranked_n100_k5.jsonl

python -m evaluation.compare \
  evaluation/results/baseline_n100_k5.jsonl \
  evaluation/results/verified_n100_k5.jsonl
```

Leakage test: `tests/test_splits.py`.
