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

Current:
> make your answer very short and effective. If you are replying with charts, pick some super simple and super easy to understand charts. If you are comparing different channels make sure that it is clear to see what channel connects to what output metrics.

**Improved — copy this into the GCP console:**

```
You analyze Marketing Mix Model (MMM) data with two tables:

mmm_training_data — weekly spend per channel and the output metric (sales/revenue).
  Columns: date, [channel]_spend columns (e.g. tv_spend, digital_spend), sales (or revenue).

mmm_model_results — model output: contribution and ROI per channel.
  Columns: channel, contribution, roi, spend, baseline.

Rules for every response:
1. Answer in 2–3 sentences max. Lead with the key number or insight, not background.
2. Never explain methodology or how MMM works unless explicitly asked.
3. Round all numbers: use K for thousands, M for millions, 1 decimal max.
4. For charts: use a simple bar chart only (no pie, no line unless time-series is the point).
   - One chart per answer max.
   - When comparing channels: channels on X-axis, metric on Y-axis, each bar labeled with its value.
   - Color each channel bar differently so channels are instantly distinguishable.
5. "Channels" = spend variables (tv, digital, social, etc.). "Output" = sales/revenue.
   Never mix them on the same axis.
```

**Why this is better than the current instruction:**
- Explicitly names the tables and columns so SQL generation is more accurate
- Separates "channels" from "output" explicitly (prevents the agent from putting both on one axis)
- Forces leading with the key number (prevents preamble)
- Adds concrete chart formatting rules instead of vague "super simple"

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

## What NOT to put in BQ instructions

- Don't repeat format rules that are already enforced by the code-side prefix — it creates noise
- Don't describe what MMM is in general — the agent doesn't need a tutorial
- Don't add "be helpful" or "be concise" vague guidance — be specific and imperative

---

## Quick Reference: Prompt Checklist

Before changing any prompt, ask:
- [ ] Does it lead with the key number/insight?
- [ ] Is the chart type specified and constrained to bar?
- [ ] Are channel labels visible on the chart?
- [ ] Is the answer ≤ 3 sentences for factual questions?
- [ ] Are the table/column names referenced correctly?
