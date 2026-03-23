# 📊 Marketing Mix Modeling — Interactive Demo

> *How much of your sales actually came from that TV campaign? And was it worth it compared to digital?*
> This app answers those questions — with math you can verify.

A full end-to-end **Marketing Mix Modeling (MMM)** platform built in Python: synthetic data generation with known ground truth, two fitted models (Ridge + Bayesian), an interactive 6-tab Streamlit dashboard, BigQuery cloud sync, and a Gemini-powered natural language interface to query your results.

---

## What is Marketing Mix Modeling?

Imagine you spend money across TV, Digital, Social, and Out-of-Home advertising every week, and your sales go up and down. How do you know *which channel caused which sales*? Simple correlation doesn't work — there are at least two effects that break it:

**Adstock (carry-over effect)**
An ad doesn't just drive sales on the day it runs. A TV spot on Monday still influences buying decisions on Friday, or even next week. This "memory" effect decays over time — some channels (TV, OOH) have long carry-over, others (Social) fade quickly.

**Saturation (diminishing returns)**
Doubling your ad budget does *not* double your sales. The first £10k of TV spend might generate £8k in incremental sales. The next £10k generates £5k. The next £10k generates £3k. Spend past the saturation point and you're wasting money.

**MMM accounts for both.** It fits a model to your historical data that captures these non-linear effects — then decomposes your total sales into: *"this much came from trend/seasonality, this much from TV, this much from Digital..."* and finally tells you the **ROI per channel**. And because it works entirely on aggregated weekly totals — total spend vs. total sales — there is no user-level tracking involved. MMM is inherently privacy-safe and unaffected by cookie deprecation.

---

## Advantage of Bayesian approach

Most real-world MMM projects suffer from a fundamental problem: *you never know if the model is right*, because the true answer is unknowable from real data.

This demo sidesteps that entirely. The dataset is **synthetically generated with known parameters baked in** — the true adstock decay, saturation points, and ROI for each channel are fixed upfront. The model's job is to *recover* these values from noisy observed data.

This means model accuracy is **objectively measurable**, not just plausible-looking. You can see exactly how close the fitted ROIs are to the true ROIs — which also makes it a clean demonstration of what Bayesian MMM offers over simple regression. Rather than one number that implies false precision, the Bayesian model produces a full posterior distribution: you see where it's confident and where it isn't. It can also encode prior knowledge (past incrementality tests, industry benchmarks) directly into the fitting process, which is why it can reach reliable results with as little as 12–18 months of data rather than the 3+ years typically needed by traditional approaches.

| Channel | True ROI | Adstock decay | Half-saturation |
|---------|----------|---------------|-----------------|
| TV | £0.55/£ | 0.70 (slow decay) | £30k/week |
| Digital | £1.10/£ | 0.40 (fast decay) | £18k/week |
| Social | £0.90/£ | 0.30 (very fast) | £9k/week |
| OOH | £0.40/£ | 0.60 (medium decay) | £6k/week |

---

## The App — 6 tabs, zero setup required

The dashboard loads instantly with Ridge regression results pre-fitted. No waiting, no configuration needed.

### 🎯 Results
The core MMM output. A stacked area chart decomposes every week's sales into baseline (trend + seasonality) and incremental contributions from each channel. KPI cards show model R², best-performing channel by ROI, and total media-driven revenue. Switch between ground-truth decomposition and model estimates side by side.

### ⚙️ Budget Optimizer
The most business-relevant tab. Set a **total weekly budget** with a single slider and the optimizer automatically finds the best allocation across channels to maximise projected media-driven sales. It uses `scipy` SLSQP constrained optimisation with the same adstock and saturation response curves the model learned — so diminishing returns are fully respected. Results are shown as two side-by-side charts: **Effective ROI per channel** (£ sales generated per £ spent, current vs. optimised) and **Spend allocation** (current vs. optimised weekly spend per channel). This is how a marketing team would use MMM in practice: *"given my budget, where should I put every pound?"*

### 📊 The Data
Time series view of sales and all four channel spend series. Toggle channel visibility, normalize spend to 0–1 scale for fair visual comparison, and inspect the underlying raw data table.

### 🔬 Explore
EDA panel: correlation heatmap, scatter plots between any two variables, and spend distribution histograms. Useful for understanding seasonality patterns and channel co-movement before trusting the model.

### 🔗 Cloud Sync
Push the dataset and model results to **Google BigQuery** with one click. Auto-loads existing results from BigQuery on startup if credentials are configured — so model outputs persist across sessions in the cloud.

### 💬 Ask Gemini
Natural language interface powered by **Gemini Data Analytics**. Ask questions like *"which channel had the highest ROI?"* or *"show me total spend by channel as a bar chart"* — Gemini queries the BigQuery tables and renders the answer with auto-generated Plotly charts.

---

## Two models, one dashboard

| | Ridge Regression | Bayesian (PyMC-Marketing) |
|---|---|---|
| Speed | Instant | ~10 min (Colab) |
| Uncertainty | None | Full posterior HDI intervals |
| Best for | Live demo / exploration | Rigorous analysis |
| How to load | Auto-runs on startup | Drop 3 CSVs into `results/` |

The app detects which results are available and switches automatically. Both use identical adstock (geometric) and saturation (Hill) transforms.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) — the dashboard starts immediately with Ridge results.

---

## For full Bayesian results (optional)

1. Open `fit_model.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run all cells (~10 min on Colab CPU)
3. Download the 3 output CSVs
4. Place them in a `results/` folder next to `app.py`
5. Restart the Streamlit app — Bayesian results load automatically

---

## BigQuery + Gemini setup (optional)

Copy `.env.example` to `.env` and fill in your GCP details:

```
GOOGLE_CLOUD_PROJECT=your-project-id
BQ_DATASET=your-dataset-name
BQ_LOCATION=EU
GOOGLE_APPLICATION_CREDENTIALS=service_account.json
BQ_AGENT_ID=your-gemini-agent-id
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Dashboard | Streamlit |
| Visualisation | Plotly |
| Bayesian MMM | PyMC-Marketing (GeometricAdstock + LogisticSaturation) |
| Quick-fit model | scikit-learn Ridge regression |
| Data warehouse | Google BigQuery |
| LLM Q&A | Gemini Data Analytics API |
| Data | Synthetic — generated with NumPy, known ground truth |

---

## Project structure

```
MMM/
├── app.py                  # Main Streamlit dashboard (6 tabs, ~700 lines)
├── bq_agent.py             # BigQuery upload/query + Gemini chat wrapper
├── fit_model.ipynb         # Bayesian fitting notebook (run in Colab)
├── requirements.txt
├── .env.example
├── results/                # Drop Bayesian CSVs here (auto-detected on startup)
│   ├── mmm_model_results.csv
│   ├── mmm_fitted_values.csv
│   └── mmm_contrib_hdi.csv
└── prompts.md              # Gemini prompting strategy
```
