# Marketing Mix Modeling — Demo

A full-stack Bayesian MMM built to demonstrate the complete analytics workflow:
synthetic data generation → Bayesian model fitting → interactive dashboard → cloud data warehouse → LLM-powered Q&A.

Because the dataset is **synthetic with known parameters**, model accuracy is objectively verifiable — something real-world MMM projects cannot offer.

---

## What it does

**Marketing Mix Modeling (MMM)** answers: *how much did each marketing channel contribute to sales?*
It accounts for two real-world effects that simple regression ignores:

- **Adstock (carry-over):** A TV ad from last week still drives sales this week
- **Saturation (diminishing returns):** Doubling spend doesn't double sales

This demo fits a Bayesian model (PyMC-Marketing) on 2 years of weekly data across 4 channels (TV, Digital, Social, OOH), then visualises the results in an interactive Streamlit dashboard.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard auto-loads Ridge regression results on startup (no model fitting required).

---

## For full Bayesian results

1. Open `fit_model.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run all cells (takes ~10 min on Colab CPU)
3. Download the 3 CSV files from Cell 6a–6c
4. Place them in a `results/` folder next to `app.py`
5. Restart the Streamlit app — it auto-loads the Bayesian results

---

## Dashboard tabs

| Tab | What you'll find |
|-----|-----------------|
| 🎯 Results | Sales decomposition chart, fitted vs actual, channel ROI and contributions |
| ⚙️ Budget Optimizer | Interactive sliders to reallocate spend and see projected media uplift |
| 📊 The Data | Time series of sales and channel spend |
| 🔬 Explore | Correlations, scatter plots, distributions |
| 🔗 Cloud Sync | Push data to BigQuery (requires GCP credentials) |
| 💬 Ask Gemini | Natural-language Q&A on data stored in BigQuery |

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Dashboard | Streamlit |
| Bayesian MMM | PyMC-Marketing (GeometricAdstock + LogisticSaturation) |
| Quick-fit fallback | scikit-learn Ridge regression |
| Data warehouse | Google BigQuery |
| LLM Q&A | Gemini Data Analytics API |
| Visualisation | Plotly |

---

## Project structure

```
MMM/
├── app.py                  # Main Streamlit dashboard (6 tabs)
├── bq_agent.py             # BigQuery upload/query + Gemini chat wrapper
├── fit_model.ipynb         # Bayesian fitting notebook (run in Colab)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── results/                # Pre-baked Bayesian CSVs (place here after Colab run)
│   ├── mmm_model_results.csv
│   ├── mmm_fitted_values.csv
│   └── mmm_contrib_hdi.csv
└── prompts.md              # LLM prompting strategy notes
```

---

## Ground-truth parameters (synthetic data)

| Channel | True ROI | Adstock decay | Half-saturation |
|---------|----------|---------------|-----------------|
| TV | £0.55/£ | 0.70 | £30k/week |
| Digital | £1.10/£ | 0.40 | £18k/week |
| Social | £0.90/£ | 0.30 | £9k/week |
| OOH | £0.40/£ | 0.60 | £6k/week |

The model's task is to recover these values from noisy observed data.

---

## BigQuery / Gemini setup

Copy `.env.example` to `.env` and fill in:

```
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
BQ_DATASET=your-dataset-name
BQ_LOCATION=EU
GOOGLE_APPLICATION_CREDENTIALS=service_account.json
BQ_AGENT_ID=your-gemini-agent-id
```
