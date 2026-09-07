# Restaurant Review Intelligence

**Turning 691 unstructured customer reviews into an operations brief a two-person restaurant can act on.**

🔗 **[Live Demo](https://restaurant-review-intelligence.streamlit.app/)** · Built for Spice Up Thai Eatery, Los Angeles

---

## The Problem

Spice Up Thai Eatery holds a 4.39-star average across 564 Google and Yelp reviews. By every surface metric, the business is healthy.

But averages hide things. Buried in that rating are negative reviews describing recurring, fixable operational problems — and no one had ever read them systematically. The owner runs the entire restaurant with one other person: cooking, front counter, phones, online orders, cleaning. Reading 564 reviews and synthesizing patterns is not a realistic use of their time.

This project reads them instead, groups them by what they're actually about, and returns three specific recommendations sized for a two-person team.

---

## How I Got Here

I actually did not start with NLP. I felt that the path to making this project mattered as much as the result, so it's documented honestly.

**Attempt 1 — Sales forecasting.** My first plan was a time-series model on the restaurant's POS data to predict daily revenue. I pulled six months of sales, built the pipeline, and then talked to the owner. But the owner already knew his patterns cold: Sunday is busiest, Monday is slowest, 6 PM is peak. A forecast would have told him what he'd known for years. I felt that the project would be valid, but commercially useless.

**Attempt 2 — A promotion experiment.** I pivoted to identifying the slowest weekday and running a flyer campaign to lift it, measuring the before-and-after. I felt that this idea was better because it had a measurable outcome — but the analysis underneath was nothing more than a group-by and a mean. So I felt that this project didn't have enough technical substance to stand as a data science project.

**Attempt 3 — Anomaly detection on revenue.** I considered rolling z-scores over daily sales to flag statistically unusual days. I dropped it after checking the sample size: I felt that 148 days across six months is far too thin for meaningful time-series decomposition, and I had no external variables to explain the anomalies I'd surface.

**Where I landed.** I figured that the owner's blind spot wasn't his numbers — it was his text. He had 691 reviews he'd never analyzed, containing exactly the kind of qualitative signal that doesn't show up in a POS export. That's a real NLP problem with a real business consumer, and it's what this repository builds.

---

## Pipeline

```
SerpApi → Cleaning → Temporal Filter → Embeddings → UMAP → K-Means → LLM Brief
  691         564          25            384-dim      5-dim     k=3      3 memos
```

**1 · Ingestion** — `src/ingest.py`
Pulled 291 Google Maps reviews and 400 Yelp reviews through the SerpApi REST API, handling two different pagination schemes (Google uses opaque page tokens; Yelp uses integer offsets). Raw JSON is cached locally so the API is never called twice for the same data.

**2 · Preprocessing** — `src/preprocess.py`
Unified two different response schemas into one table. Applied HTML unescaping, whitespace normalization, Google-translation-artifact stripping, deduplication, a five-word minimum, and language filtering via `langdetect`. 691 raw records → **564 analysis-ready reviews**.

This is what I deliberately *didn't* apply: lowercasing, emoji removal, stop-word removal, punctuation stripping. I found that Sentence Transformer embeddings now encode case, emphasis, and negation as signal — for example,  "the food was BAD" and "the food was bad" carry different weight, and removing "not" inverts meaning outright.

**3 · Temporal filtering**
The restaurant relocated in August 2021. Reviews of the previous location describe a different kitchen and a different room, so including them would poison the analysis. Filtering to post-September-2021 reduced 91 negative reviews to **25 that describe the current business**. The Silhouette score improved from 0.3975 to 0.4492 as a direct result.

**4 · Embeddings** — `src/embedder.py`
Encoded each review with `all-MiniLM-L6-v2` into a 384-dimensional vector capturing semantic meaning rather than keyword overlap.

**5 · Clustering** — `src/cluster_model.py`
UMAP reduces 384 → 5 dimensions to avoid the curse of dimensionality, then K-Means partitions the reviews. **k=3** was selected by sweeping k=2 through k=6 and maximizing silhouette score.

**6 · Synthesis** — `src/summarizer.py`
Each cluster's raw reviews are passed to Groq LLaMA-3 with a prompt constrained by one hard rule: *this is a two-person operation*. Any recommendation requiring extra staff, meaningful capital, or downtime is rejected.

---

## Why Only Negative Reviews Are Clustered

The first version clustered all 564 reviews. It produced five clusters of praise, and the complaints vanished entirely.

K-Means assumes roughly balanced cluster sizes. With 473 positive and 91 negative reviews, the algorithm split the positive mass into sub-flavors of compliment and scattered the negatives as edge noise. This made the clusters mathematically defensible, but useless as a deliverable.

Filtering to negative reviews *before* clustering was the fix. It reframes the question from "what do customers say" to "what is going wrong" — which is the only version the owner can act on.

---

## Findings

| Cluster | Reviews | Theme |
|---|---|---|
| 0 | 7 | **Front Counter & Staff Interactions** — rude or distracted service, missed calls |
| 1 | 10 | **Kitchen Execution & Consistency** — undercooked vegetables, wrong noodle type, dry proteins |
| 2 | 8 | **Pricing & Operational Logistics** — surprise upcharges, hours posted incorrectly |

The interpretation and context matters as much as the grouping. In a two-person restaurant, "the cashier was on her phone" is rarely apathy — it's one person managing a delivery tablet, a register, and a wok simultaneously. Every recommendation in the brief is framed around reducing that load rather than correcting attitude.

Sample recommendations: QR-code ordering to pull the owner off the register during peak, a pre-portioned prep station to stabilize kitchen consistency under ticket pressure, and one-tap Google Business hours updates to stop customers arriving at a closed door.

Full brief: [`outputs/business_strategic_brief.md`](outputs/business_strategic_brief.md)

---

## Validation

Silhouette score measures geometry, not meaning. A tight cluster of semantically unrelated reviews still scores well. So I validated the reviews against human judgment.

I manually labeled all 25 reviews as Service / Food / Operations without looking at cluster assignments, then compared.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Food Quality | 1.00 | 0.91 | 0.95 | 11 |
| Staff & Service | 1.00 | 0.64 | 0.78 | 11 |
| Operations | 0.38 | 1.00 | 0.55 | 3 |
| **Accuracy** | | | **0.80** | **25** |

Weighted F1: **0.83**. Silhouette: **0.4492**.

**Known limitation — Service/Operations overlap.** Four Service reviews were classified as Operations. Centroid analysis explains it: three of the eight Operations reviews contain staff-interaction language, pulling that centroid toward service-centered embeddings.

The cause is behavioral rather than technical. Customers describe operational failures — surprise charges, wrong hours — in the vocabulary of bad service. Someone who drives twenty minutes to a restaurant that's closed despite its posted hours writes "terrible service," not "inaccurate business listing."

This surfaced independently in three places: manual labeling, the confusion matrix, and live centroid similarity. Convergence across three methods suggests genuine semantic overlap, not a modeling artifact.

It also produced a recommendation the clustering alone wouldn't have: fixing the operational gaps should reduce *perceived* service complaints without changing anything about how the owners behave.

---

## A Note on the Live Demo

The interactive classifier does **not** run new text through UMAP.

UMAP was fit on 25 points. Calling `.transform()` on unseen text forces it to place a new point into a manifold learned from very few examples — numerically unstable, and in testing it produced near-random assignments.

Instead, prediction computes cosine similarity against cluster centroids in the original 384-dimensional embedding space, where semantic relationships are stable regardless of training set size. UMAP is retained purely for the 2D visualization.

---

## Stack

`SerpApi` · `pandas` · `sentence-transformers` · `UMAP` · `scikit-learn` · `Groq LLaMA-3` · `Streamlit` · `Plotly`

---

## Structure

```
├── src/
│   ├── ingest.py           # SerpApi collection, dual pagination
│   ├── preprocess.py       # Schema unification, cleaning, filtering
│   ├── embedder.py         # Sentence-transformer encoding
│   ├── cluster_model.py    # UMAP + K-Means + centroid export
│   ├── evaluate_model.py   # Human-in-the-loop confusion matrix
│   ├── summarizer.py       # LLM brief generation
│   └── app.py              # Streamlit dashboard
├── data/
│   ├── raw/                # Cached API responses (gitignored)
│   └── processed/          # Cleaned and clustered datasets
├── artifacts/              # Embeddings, centroids
└── outputs/                # Generated business brief
```

## Running It

```bash
pip install -r requirements.txt

# Add SERPAPI_KEY and GROQ_API_KEY to .env

python src/ingest.py         # Collect reviews
python src/preprocess.py     # Clean and unify
python src/embedder.py       # Generate embeddings
python src/cluster_model.py  # Cluster and export centroids
python src/summarizer.py     # Generate brief
streamlit run src/app.py     # Launch dashboard
```

---

## What I'd Do Next

**More data.** Twenty-five negative reviews is the binding constraint on everything here. Adding TripAdvisor and delivery-platform reviews would sharpen the Service/Operations boundary considerably.

**Multi-label classification.** Reviews genuinely span categories. Forcing one label per review discards information — the overlap I documented is partly an artifact of that constraint.

**Temporal tracking.** Re-running quarterly would show whether implemented recommendations actually reduce complaint volume in their cluster, closing the loop from analysis to measured outcome.
