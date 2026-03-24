# MMM Agent Prompting Strategy

## Architecture: Two Layers

```
User question
     │
     ▼
[bq_agent.py]  ← code-side: wraps question with format instructions
     │
     ▼
[BQ Agent / Gemini Data Analytics]  ← GCP console: system-level instructions
     │                                  (data schema, MMM domain knowledge)
     ▼
Answer + SQL + Vega chart
```

**Division of responsibility:**
- **BQ agent instructions** (GCP console) → what the data means, how to query it, MMM domain context
- **Code-side prefix** (bq_agent.py) → output format enforcement on every call

Both layers are needed. BQ instructions alone can drift on format. Code-side prefix alone can't fix bad SQL or misunderstood columns.

---

## BQ Agent Instructions (set in GCP console)

You analyze Marketing Mix Model (MMM) data (mmm_training_data, mmm_model_results). Strictly follow these rules:

1. Text: Max 3 sentences. Lead with the core insight. No methodology explanations. Round numbers (K/M, 1 decimal max).
2. Charts: Max 1 chart. Use a standard horizontal bar chart for comparisons (Parameters/Channels on Y-axis, Metric on X-axis).
3. Visuals: Draw solid, filled bars. Label every bar with its value. Color-code distinct channels.
4. Data Accuracy: The chart must visually include all data points mentioned in your text. Ensure axis limits accommodate the largest values. Never mix spend and output on the same axis.

---

## Code-Side Question Prefix (bq_agent.py)

Add a brief format reminder to every question before sending to the API.
This acts as a per-call reinforcement on top of the BQ agent's system instructions.

**Prefix to prepend to each question (text only — no chart instructions here):**

```
[FORMAT: Answer in max 3 sentences. Lead with the key number or insight, not background.]

{user_question}
```

**Important:** Do NOT include chart format instructions in this prefix. The Gemini agent generates structured Vega-Lite specs internally — extra chart instructions in the question text can corrupt the spec structure (e.g., change field names or add layers that break rendering). Chart rules belong in the BQ agent system instructions only.

---

## Semantic Layer — BigQuery Glossary Terms

Where to add: **BigQuery UI → Agent Editor → Advanced Features → Glossary → Add term**
Fields: **Term**, **Definition**, **Synonyms** (comma-separated). No Dataplex needed.

| Term | Definition | Synonyms |
|---|---|---|
| TV | Television ad spend per week. Column `tv_spend`. Slowest decay (0.70), ROI £0.55/£. | television, broadcast, linear TV |
| Digital | Online advertising spend per week. Column `digital_spend`. Highest ROI channel at £1.10/£. | online, paid search, PPC, display, performance marketing |
| Social | Social media ad spend per week. Column `social_spend`. Fast decay, ROI £0.90/£. | social media, Facebook, Instagram, paid social |
| OOH | Out-of-home ad spend per week. Column `ooh_spend`. Lowest ROI at £0.40/£. | out-of-home, outdoor, billboard, poster |
| ROI | £ of incremental sales generated per £1 of ad spend. Column `roi` in `mmm_model_results`. | return on investment, return on ad spend, ROAS, effectiveness, payback |
| Media contribution | Incremental sales attributed to a channel's advertising. Column `contribution` in `mmm_model_results`. Excludes baseline. | channel contribution, incremental revenue, lift, attributed sales, ad impact |
| Baseline | Sales that would occur with zero advertising — organic demand and seasonality. Column `baseline` in `mmm_model_results`. | organic sales, base sales, non-media sales, underlying demand |
| Adstock | Carry-over effect of advertising: each week's impact decays into future weeks. TV decays slowest (0.70), Social fastest (0.30). Not a stored column. | carry-over, advertising memory, decay, lagged effect |
| Saturation | Diminishing returns on spend — each extra £1 generates less than the last. Half-saturation points: TV £30k, Digital £18k, Social £9k, OOH £6k/week. | diminishing returns, response curve, Hill curve, spend efficiency |
