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
| TV | Television advertising spend per week. Maps to column `tv_spend` in `mmm_training_data`. | television, broadcast, linear TV |
| Digital | Online advertising spend per week. Maps to column `digital_spend` in `mmm_training_data`. | online, paid search, PPC, display, performance marketing |
| Social | Social media advertising spend per week. Maps to column `social_spend` in `mmm_training_data`. | social media, Facebook, Instagram, paid social |
| OOH | Out-of-home advertising spend per week. Maps to column `ooh_spend` in `mmm_training_data`. | out-of-home, outdoor, billboard, poster |
| ROI | £ of incremental sales generated per £1 of ad spend for a given channel. Query column `roi` in `mmm_model_results` for the actual value. | return on investment, return on ad spend, ROAS, effectiveness, payback |
| Media contribution | Incremental sales attributed to a channel's advertising spend. Column `contribution` in `mmm_model_results`. Does not include baseline organic sales. | channel contribution, incremental revenue, lift, attributed sales, ad impact |
| Baseline | Sales that would occur with zero advertising — organic demand and seasonality. Column `baseline` in `mmm_model_results`. | organic sales, base sales, non-media sales, underlying demand |
| Adstock | The carry-over effect of advertising: impact from one week persists and decays into subsequent weeks. A modelling concept — not a stored column. | carry-over, advertising memory, decay, lagged effect |
| Saturation | Diminishing returns on advertising spend — each additional £1 generates less incremental sales than the last. A modelling concept — not a stored column. | diminishing returns, response curve, Hill curve, spend efficiency |
