"""
Marketing Mix Modeling — Interview Demo
Interactive dashboard: sales decomposition, budget optimizer, EDA, BigQuery + Gemini chat.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# Bridge Streamlit Cloud secrets → os.environ (no-op when running locally)
_BRIDGE_KEYS = ["GOOGLE_CLOUD_PROJECT", "BQ_DATASET", "BQ_LOCATION", "BQ_AGENT_ID",
                "GOOGLE_APPLICATION_CREDENTIALS"]
try:
    for _k in _BRIDGE_KEYS:
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bq_agent import ask_bq_agent, format_response, load_from_bq, upload_to_bq

RESULTS_DIR = Path("results")

# ── Ground-truth simulation parameters (known because data is synthetic) ──────
TRUE_DECAY    = {"tv_spend": 0.70, "digital_spend": 0.40, "social_spend": 0.30, "ooh_spend": 0.60}
TRUE_HALF_SAT = {"tv_spend": 30_000, "digital_spend": 18_000, "social_spend": 9_000, "ooh_spend": 6_000}
TRUE_ROI      = {"tv_spend": 0.55, "digital_spend": 1.10, "social_spend": 0.90, "ooh_spend": 0.40}

CHANNEL_LABELS = {
    "tv_spend": "TV",
    "digital_spend": "Digital",
    "social_spend": "Social",
    "ooh_spend": "OOH",
}

PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA",
           "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]

# Decomposition colour scheme: baseline (grey) then one colour per channel
DECOMP_COLORS = ["rgba(173,181,189,0.85)", "rgba(239,85,59,0.80)",
                 "rgba(0,204,150,0.80)", "rgba(171,99,250,0.80)", "rgba(255,161,90,0.80)"]


# ── Shared transforms ──────────────────────────────────────────────────────────
def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock: carry-over effect decays at rate `decay` each week."""
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] + decay * out[i - 1]
    return out


def hill(x: np.ndarray, half_sat: float, slope: float = 2.0) -> np.ndarray:
    """Hill saturation curve: diminishing returns as spend increases."""
    return x**slope / (half_sat**slope + x**slope)


# ── Vega-Lite → Plotly renderer (for BQ agent charts) ─────────────────────────
def _parse_enc_field(enc_val) -> tuple:
    """Return (field_name, title) from a Vega-Lite encoding value (object or shorthand)."""
    if isinstance(enc_val, dict):
        field = enc_val.get("field")
        return field, enc_val.get("title", field)
    if isinstance(enc_val, str):
        field = enc_val.split(":")[0]
        return field or None, field or None
    return None, None


def _render_vega_as_plotly(spec: dict) -> None:
    """Render a Vega-Lite spec returned by the BQ agent using Plotly."""
    layer0 = (spec.get("layer") or [{}])[0]
    enc = spec.get("encoding") or layer0.get("encoding", {})
    mark_raw = spec.get("mark") or layer0.get("mark", "bar")
    mark = mark_raw.get("type", "bar") if isinstance(mark_raw, dict) else (mark_raw or "bar")

    title = spec.get("title", "")
    if isinstance(title, dict):
        title = title.get("text", "")

    data_values = spec.get("data", {}).get("values", [])
    if not data_values:
        data_values = layer0.get("data", {}).get("values", [])
    if not data_values:
        return

    df_chart = pd.DataFrame(data_values)
    x_field, x_title = _parse_enc_field(enc.get("x"))
    y_field, y_title = _parse_enc_field(enc.get("y"))

    for transform in spec.get("transform", []):
        if "fold" in transform:
            fold_fields = transform["fold"]
            as_fields = transform.get("as", ["key", "value"])
            df_chart = df_chart.melt(
                id_vars=[c for c in df_chart.columns if c not in fold_fields],
                value_vars=fold_fields,
                var_name=as_fields[0],
                value_name=as_fields[1],
            )
            break

    if not x_field or not y_field or x_field not in df_chart or y_field not in df_chart:
        with st.expander("Chart data missing expected fields — raw spec"):
            st.json(spec)
        return

    color_enc = enc.get("color", {})
    default_color = color_enc.get("value", "steelblue")
    condition = color_enc.get("condition", {})
    colors = None
    if condition:
        cond_test = condition.get("test", "")
        cond_value = condition.get("value", default_color)
        m = re.search(r"datum\.(\w+)\s*===?\s*['\"]([^'\"]+)['\"]", cond_test)
        if m:
            cond_field, cond_match = m.group(1), m.group(2)
            colors = [
                cond_value if str(row.get(cond_field, "")) == cond_match else default_color
                for row in data_values
            ]

    if mark == "bar":
        fig = go.Figure(go.Bar(x=df_chart[x_field], y=df_chart[y_field], marker_color=colors or default_color))
    elif mark in ("point", "tick", "circle", "square"):
        fig = go.Figure(go.Scatter(x=df_chart[x_field], y=df_chart[y_field], mode="markers", marker=dict(size=7)))
    elif mark == "line":
        fig = go.Figure(go.Scatter(x=df_chart[x_field], y=df_chart[y_field], mode="lines"))
    else:
        fig = go.Figure(go.Bar(x=df_chart[x_field], y=df_chart[y_field]))

    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=y_title,
        template="plotly_white", height=spec.get("height", 350), margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MMM | Marketing Mix Modeling", page_icon="📊", layout="wide")


# ── Data & model functions ─────────────────────────────────────────────────────
@st.cache_data
def make_dataset(seed: int = 42, n_weeks: int = 104) -> pd.DataFrame:
    """Generate 2 years of synthetic weekly MMM data with known ground-truth parameters."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)

    channels = {
        "tv_spend":      (40_000, 15_000),
        "digital_spend": (25_000,  8_000),
        "social_spend":  (12_000,  4_000),
        "ooh_spend":     ( 8_000,  3_000),
    }

    seasonality = (
        1 + 0.20 * np.sin(2 * np.pi * t / 52)
        + 0.08 * np.sin(4 * np.pi * t / 52 + 0.5)
    )
    trend = 50_000 + 80 * t

    df_out = pd.DataFrame({"date": dates})
    media_contrib = np.zeros(n_weeks)

    for ch, (mean_spend, std_spend) in channels.items():
        raw = rng.normal(mean_spend, std_spend, n_weeks).clip(0)
        raw *= (0.85 + 0.30 * np.abs(np.sin(np.pi * t / 52 + rng.uniform(0, np.pi))))
        df_out[ch] = raw.round(0)

        adstocked = adstock(raw, TRUE_DECAY[ch])
        saturated = hill(adstocked, TRUE_HALF_SAT[ch]) * TRUE_HALF_SAT[ch]
        media_contrib += TRUE_ROI[ch] * saturated

    df_out["sales"] = (
        trend * seasonality + media_contrib + rng.normal(0, 4_000, n_weeks)
    ).round(0).clip(0)

    return df_out


@st.cache_data
def decompose_sales(df: pd.DataFrame) -> tuple:
    """Compute ground-truth weekly contribution per channel and the baseline."""
    contribs = {}
    for ch in TRUE_ROI:
        ads = adstock(df[ch].values.astype(float), TRUE_DECAY[ch])
        sat = hill(ads, TRUE_HALF_SAT[ch]) * TRUE_HALF_SAT[ch]
        contribs[ch] = TRUE_ROI[ch] * sat
    baseline = df["sales"].values.astype(float) - sum(contribs.values())
    return contribs, baseline


def _run_ridge(df: pd.DataFrame, n_fourier: int = 2) -> tuple:
    """Fit Ridge MMM. Returns (summary_df, fitted_df, contrib_hdi_df, r2)."""
    from sklearn.linear_model import Ridge

    channel_list = list(TRUE_ROI.keys())
    n_weeks = len(df)
    t = np.arange(n_weeks)

    ch_feats = {}
    for ch in channel_list:
        raw = df[ch].values.astype(float)
        ads = adstock(raw, TRUE_DECAY[ch])
        ch_feats[ch] = hill(ads, TRUE_HALF_SAT[ch])

    fourier = []
    for k in range(1, n_fourier + 1):
        fourier.append(np.sin(2 * np.pi * k * t / 52))
        fourier.append(np.cos(2 * np.pi * k * t / 52))

    X_mat = np.column_stack([ch_feats[ch] for ch in channel_list] + fourier + [t / n_weeks])
    y_vec = df["sales"].values.astype(float)

    ridge = Ridge(alpha=10.0, fit_intercept=True)
    ridge.fit(X_mat, y_vec)
    y_pred = ridge.predict(X_mat)

    n_ch = len(channel_list)
    ch_contribs = np.maximum(
        [float((ridge.coef_[i] * X_mat[:, i]).sum()) for i in range(n_ch)], 0
    )
    total_spend = df[channel_list].sum()
    baseline_val = float(y_pred.sum()) - float(ch_contribs.sum())

    rows = [{
        "channel":      ch,
        "contribution": round(float(ch_contribs[i]), 0),
        "spend":        round(float(total_spend[ch]), 0),
        "roi":          round(float(ch_contribs[i]) / float(total_spend[ch]), 4) if float(total_spend[ch]) > 0 else 0.0,
        "baseline":     round(baseline_val, 0),
    } for i, ch in enumerate(channel_list)]

    ss_res = float(np.sum((y_vec - y_pred) ** 2))
    ss_tot = float(np.sum((y_vec - y_vec.mean()) ** 2))

    summary_df = pd.DataFrame(rows)
    fitted_df = pd.DataFrame({
        "date":         df["date"],
        "fitted_sales": y_pred.round(0),
        "lower_89":     y_pred.round(0),
        "upper_89":     y_pred.round(0),
    })
    contrib_hdi_df = pd.DataFrame({
        "channel":  channel_list,
        "mean":     ch_contribs.round(0),
        "lower_89": ch_contribs.round(0),
        "upper_89": ch_contribs.round(0),
    })
    return summary_df, fitted_df, contrib_hdi_df, 1 - ss_res / ss_tot


# ── Load data ──────────────────────────────────────────────────────────────────
df = make_dataset()
date_col     = "date"
target_col   = "sales"
channel_cols = list(TRUE_ROI.keys())
bq_project   = os.getenv("GOOGLE_CLOUD_PROJECT", "")
bq_dataset   = os.getenv("BQ_DATASET", "")


# ── Auto-load results on startup ───────────────────────────────────────────────
if "results_loaded" not in st.session_state:
    _m = RESULTS_DIR / "mmm_model_results.csv"
    _f = RESULTS_DIR / "mmm_fitted_values.csv"
    _h = RESULTS_DIR / "mmm_contrib_hdi.csv"

    if _m.exists() and _f.exists() and _h.exists():
        _summary = pd.read_csv(_m)
        _fitted  = pd.read_csv(_f, parse_dates=["date"])
        _hdi     = pd.read_csv(_h)

        st.session_state["summary"]     = _summary
        st.session_state["contrib_hdi"] = _hdi
        st.session_state["source"]      = "bayesian"

        # Detect degenerate fitted_values (Bayesian posterior predictive failed)
        _fitted_ok = _fitted["fitted_sales"].std() > 10
        if _fitted_ok:
            _y_true = df[target_col].values
            _y_pred = _fitted["fitted_sales"].values
            ss_res = float(np.sum((_y_true - _y_pred) ** 2))
            ss_tot = float(np.sum((_y_true - _y_true.mean()) ** 2))
            st.session_state["fitted_values"] = _fitted
            st.session_state["r2"]            = 1 - ss_res / ss_tot
        else:
            # Posterior predictive unavailable — compute Ridge fit for the fitted curve only
            _, _r_fitted, _, _r_r2 = _run_ridge(df)
            st.session_state["fitted_values"]           = _r_fitted
            st.session_state["r2"]                      = _r_r2
            st.session_state["bayesian_fitted_missing"] = True

    else:
        # No pre-baked CSVs — auto-run Ridge so the demo always starts with results
        _r_summary, _r_fitted, _r_hdi, _r_r2 = _run_ridge(df)
        st.session_state["summary"]       = _r_summary
        st.session_state["fitted_values"] = _r_fitted
        st.session_state["contrib_hdi"]   = _r_hdi
        st.session_state["source"]        = "ridge"
        st.session_state["r2"]            = _r_r2

    st.session_state["results_loaded"] = True


# ── Auto-load model results from BigQuery ─────────────────────────────────────
if bq_project and "bq_data_loaded" not in st.session_state:
    _bq_summary = load_from_bq("mmm_model_results", bq_project, bq_dataset)
    if _bq_summary is not None:
        st.session_state["summary"] = _bq_summary
    _bq_fitted = load_from_bq("mmm_fitted_values", bq_project, bq_dataset)
    if _bq_fitted is not None:
        st.session_state["fitted_values"] = _bq_fitted
    st.session_state["bq_data_loaded"] = True


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 Marketing Mix Modeling")
st.markdown(
    "Bayesian MMM on 2 years of synthetic weekly data — 4 channels, known ground truth. "
    "Decompose sales, explore the data, optimise your budget, and query results via Gemini."
)
st.divider()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.info(
        "**Marketing Mix Modeling (MMM)** quantifies how much each marketing channel "
        "contributes to sales — accounting for carry-over effects *(adstock)* and "
        "diminishing returns *(saturation)*.\n\n"
        "This demo uses **synthetic data** so the true ROI per channel is known, "
        "making model recovery fully verifiable."
    )

    st.divider()
    st.markdown("### Model")

    _source = st.session_state.get("source", "none")
    _r2     = st.session_state.get("r2")

    if _source == "bayesian":
        st.success("🟢 Bayesian (PyMC-Marketing)")
        if st.session_state.get("bayesian_fitted_missing"):
            st.caption("Posterior predictive unavailable — Ridge used for fitted curve.")
    elif _source == "ridge":
        st.info("⚪ Ridge regression (quick fit, no uncertainty)")
    else:
        st.warning("⚫ Not fitted")

    if _r2 is not None:
        st.metric("R²", f"{_r2:.3f}")

    st.divider()
    st.markdown("### View Controls")
    selected_channels = st.multiselect("Channels", channel_cols, default=channel_cols,
                                       format_func=lambda c: CHANNEL_LABELS.get(c, c))
    normalize = st.checkbox("Normalize spend (0–1 scale)", value=False)
    show_raw  = st.checkbox("Show raw data table", value=False)

    with st.expander("Advanced"):
        n_fourier = st.slider("Fourier seasonality terms", 1, 4, 2)
        if st.button("Re-fit Ridge", help="Re-run Ridge regression with current Fourier setting"):
            _r_summary, _r_fitted, _r_hdi, _r_r2 = _run_ridge(df, n_fourier)
            st.session_state["summary"]               = _r_summary
            st.session_state["fitted_values"]         = _r_fitted
            st.session_state["contrib_hdi"]           = _r_hdi
            st.session_state["source"]                = "ridge"
            st.session_state["r2"]                    = _r_r2
            st.session_state.pop("bayesian_fitted_missing", None)
            st.rerun()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_results, tab_budget, tab_ts, tab_explore, tab_bq, tab_chat = st.tabs([
    "🎯 Results",
    "⚙️ Budget Optimizer",
    "📊 The Data",
    "🔬 Explore",
    "🔗 Cloud Sync",
    "💬 Ask Gemini",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Results
# ════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.info(
        "**MMM decomposes total sales into components:** "
        "*Baseline* (trend + seasonality that would exist with zero ad spend) "
        "+ *media contributions* (incremental sales each channel drove). "
        "The stacked area chart below shows these week by week."
    )

    summary     = st.session_state.get("summary")
    source      = st.session_state.get("source", "ridge")
    r2          = st.session_state.get("r2", 0.0)
    contrib_hdi = st.session_state.get("contrib_hdi")
    fv          = st.session_state.get("fitted_values")

    # ── KPI row ────────────────────────────────────────────────────────────
    if summary is not None:
        best_roi_idx = summary["roi"].idxmax()
        best_roi_ch  = summary.loc[best_roi_idx, "channel"]
        total_media  = float(summary["contribution"].sum())
        model_label  = "Bayesian" if source == "bayesian" else "Ridge"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", model_label)
        c2.metric("R²", f"{r2:.3f}")
        c3.metric("Best ROI channel", CHANNEL_LABELS.get(best_roi_ch, best_roi_ch))
        c4.metric("Total media contribution", f"£{total_media / 1e6:.1f}M")

    st.divider()

    # ── Sales decomposition stacked area ───────────────────────────────────
    st.subheader("Sales Decomposition")
    st.caption(
        "Ground-truth decomposition using the known simulation parameters. "
        "Baseline captures trend and seasonality; coloured bands show incremental media lift."
    )

    gt_contribs, gt_baseline = decompose_sales(df)

    decomp_labels = ["Baseline"] + [CHANNEL_LABELS[ch] for ch in channel_cols]
    decomp_arrays = [gt_baseline] + [gt_contribs[ch] for ch in channel_cols]

    fig_decomp = go.Figure()
    for label, arr, color in zip(decomp_labels, decomp_arrays, DECOMP_COLORS):
        fig_decomp.add_trace(go.Scatter(
            x=df[date_col], y=arr,
            name=label,
            stackgroup="one",
            mode="none",
            fillcolor=color,
            line=dict(width=0),
        ))
    fig_decomp.add_trace(go.Scatter(
        x=df[date_col], y=df[target_col].values,
        mode="lines", name="Actual Sales",
        line=dict(color="black", width=2, dash="dot"),
    ))
    fig_decomp.update_layout(
        height=420, template="plotly_white", hovermode="x unified",
        yaxis_title="Sales (£)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=30),
    )
    st.plotly_chart(fig_decomp, use_container_width=True)

    st.divider()

    # ── Fitted vs Actual ───────────────────────────────────────────────────
    if fv is not None:
        st.subheader("Fitted vs Actual Sales")
        bayesian_fit = source == "bayesian" and not st.session_state.get("bayesian_fitted_missing")
        if bayesian_fit:
            st.caption("Posterior mean with 89% credible interval band.")
        else:
            st.caption("Ridge regression fit (point estimate, no uncertainty).")

        fig_fit = go.Figure()

        if bayesian_fit and "lower_89" in fv.columns:
            dates_fwd = fv["date"].values
            dates_rev = fv["date"].values[::-1]
            fig_fit.add_trace(go.Scatter(
                x=np.concatenate([dates_fwd, dates_rev]),
                y=np.concatenate([fv["upper_89"].values, fv["lower_89"].values[::-1]]),
                fill="toself", fillcolor="rgba(239,85,59,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="89% CI",
            ))

        fig_fit.add_trace(go.Scatter(
            x=df[date_col], y=df[target_col].values,
            mode="lines", name="Actual", line=dict(color="#636EFA", width=2),
        ))
        fit_label = "Posterior mean" if bayesian_fit else "Ridge fit"
        fig_fit.add_trace(go.Scatter(
            x=fv["date"], y=fv["fitted_sales"],
            mode="lines", name=fit_label,
            line=dict(color="#EF553B", width=2, dash="dash"),
        ))
        fig_fit.update_layout(
            height=360, template="plotly_white", hovermode="x unified",
            yaxis_title="Sales (£)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=20, b=30),
        )
        st.plotly_chart(fig_fit, use_container_width=True)

        if bayesian_fit:
            with st.expander("What is an 89% credible interval?"):
                st.markdown(
                    "An **89% credible interval** means: given the data and priors, there is an "
                    "89% probability the true value lies within this range. Unlike a frequentist "
                    "confidence interval, it is a direct probability statement about the parameter. "
                    "89% follows conventions in Bayesian statistics (McElreath, *Statistical Rethinking*)."
                )

    st.divider()

    # ── Contributions & ROI ────────────────────────────────────────────────
    if summary is not None:
        col_contrib, col_roi = st.columns(2)

        with col_contrib:
            st.subheader("Channel Contributions")
            if contrib_hdi is not None:
                chan_names     = contrib_hdi["channel"].tolist()
                mean_contribs  = contrib_hdi["mean"].values
                lower_ci       = contrib_hdi["lower_89"].values
                upper_ci       = contrib_hdi["upper_89"].values
            else:
                chan_names     = summary["channel"].tolist()
                mean_contribs  = summary["contribution"].values
                lower_ci = upper_ci = mean_contribs

            ci_label = "89% CI" if source == "bayesian" else "point estimate"
            st.caption(f"Total incremental sales attributed to each channel ({ci_label}).")
            fig_bar = go.Figure(go.Bar(
                x=[CHANNEL_LABELS.get(c, c) for c in chan_names],
                y=mean_contribs,
                error_y=dict(type="data", symmetric=False,
                             array=upper_ci - mean_contribs, arrayminus=mean_contribs - lower_ci),
                marker_color=PALETTE[1:len(chan_names) + 1],
                text=[f"£{v / 1e6:.1f}M" for v in mean_contribs],
                textposition="outside",
            ))
            fig_bar.update_layout(height=320, template="plotly_white",
                                  yaxis_title="Total contribution (£)", margin=dict(t=20, b=30))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_roi:
            st.subheader("Channel ROI")
            st.caption("Incremental sales generated per £ of spend.")
            fig_roi = go.Figure(go.Bar(
                x=[CHANNEL_LABELS.get(c, c) for c in summary["channel"]],
                y=summary["roi"],
                marker_color=PALETTE[1:len(summary) + 1],
                text=[f"£{v:.2f}" for v in summary["roi"]],
                textposition="outside",
            ))
            fig_roi.update_layout(height=320, template="plotly_white",
                                  yaxis_title="ROI (£ per £ spent)", margin=dict(t=20, b=30))
            st.plotly_chart(fig_roi, use_container_width=True)

        with st.expander("What does ROI mean here?"):
            st.markdown(
                "**ROI** = total incremental sales attributed to a channel ÷ total spend on that channel. "
                "An ROI of 1.10 means each £1 spent on Digital generated £1.10 in incremental sales — "
                "on top of what would have happened without that spend.\n\n"
                "This is the *average* ROI over the observed spend range. "
                "The *marginal* ROI (value of the next £1 spent) is lower for channels "
                "that are already saturated — head to the **Budget Optimizer** tab to explore this."
            )

        with st.expander("Full channel results table"):
            st.dataframe(
                summary.assign(channel=summary["channel"].map(lambda c: CHANNEL_LABELS.get(c, c)))
                .set_index("channel")
                .style.format({"contribution": "{:,.0f}", "spend": "{:,.0f}",
                               "roi": "{:.3f}", "baseline": "{:,.0f}"}),
                use_container_width=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Budget Optimizer
# ════════════════════════════════════════════════════════════════════════════
with tab_budget:
    st.info(
        "Adjust the weekly spend per channel to see how reallocation affects projected media-driven sales. "
        "The optimizer uses the ground-truth response curves (adstock + saturation) to project outcomes. "
        "**Assumes marginal reallocation** around current spending levels with steady-state adstock."
    )

    # Helper: annual contribution from a constant weekly spend (steady-state adstock)
    def _steady_state_annual(weekly_spend: float, ch: str, n_weeks: int = 104) -> float:
        ads_ss = weekly_spend / (1 - TRUE_DECAY[ch])
        sat = float(hill(np.array([ads_ss]), TRUE_HALF_SAT[ch])[0]) * TRUE_HALF_SAT[ch]
        return TRUE_ROI[ch] * sat * n_weeks

    current_weekly = {ch: float(df[ch].mean()) for ch in channel_cols}
    original_weekly_total = sum(current_weekly.values())
    current_annual = {ch: _steady_state_annual(current_weekly[ch], ch) for ch in channel_cols}

    # ── Sliders ────────────────────────────────────────────────────────────
    st.subheader("Weekly spend allocation")
    st.caption("Drag sliders to reallocate budget. Default = current average weekly spend per channel.")

    slider_cols = st.columns(len(channel_cols))
    new_weekly = {}
    for i, ch in enumerate(channel_cols):
        with slider_cols[i]:
            new_weekly[ch] = st.slider(
                CHANNEL_LABELS[ch],
                min_value=0,
                max_value=int(current_weekly[ch] * 2),
                value=int(current_weekly[ch]),
                step=500,
                format="£%d",
                key=f"budget_slider_{ch}",
            )

    # ── Budget tracker & headline metric ───────────────────────────────────
    new_weekly_total = sum(new_weekly.values())
    budget_delta     = new_weekly_total - original_weekly_total
    budget_delta_pct = budget_delta / original_weekly_total * 100

    projected_annual      = {ch: _steady_state_annual(new_weekly[ch], ch) for ch in channel_cols}
    projected_annual_total = sum(projected_annual.values())
    current_annual_total   = sum(current_annual.values())
    uplift                 = projected_annual_total - current_annual_total
    uplift_pct             = uplift / current_annual_total * 100 if current_annual_total > 0 else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Original weekly budget",  f"£{original_weekly_total:,.0f}")
    m2.metric(
        "New weekly budget", f"£{new_weekly_total:,.0f}",
        delta=f"£{budget_delta:+,.0f} ({budget_delta_pct:+.1f}%)",
        delta_color="inverse",  # overspending = red
    )
    m3.metric(
        "Projected media uplift (104 wks)",
        f"£{projected_annual_total / 1e6:.2f}M",
        delta=f"£{uplift / 1e3:+.0f}k ({uplift_pct:+.1f}%)",
    )

    # ── Before / After bar chart ────────────────────────────────────────────
    ch_labels = [CHANNEL_LABELS[ch] for ch in channel_cols]
    fig_budget = go.Figure()
    fig_budget.add_trace(go.Bar(
        name="Current allocation",
        x=ch_labels,
        y=[current_annual[ch] / 1e3 for ch in channel_cols],
        marker_color="#636EFA",
        text=[f"£{current_annual[ch] / 1e3:.0f}k" for ch in channel_cols],
        textposition="outside",
    ))
    fig_budget.add_trace(go.Bar(
        name="Projected allocation",
        x=ch_labels,
        y=[projected_annual[ch] / 1e3 for ch in channel_cols],
        marker_color="#EF553B",
        text=[f"£{projected_annual[ch] / 1e3:.0f}k" for ch in channel_cols],
        textposition="outside",
    ))
    fig_budget.update_layout(
        barmode="group",
        height=380, template="plotly_white",
        yaxis_title="Media contribution over 104 weeks (£k)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=30),
    )
    st.plotly_chart(fig_budget, use_container_width=True)

    st.caption(
        "**Note:** Projections use ground-truth simulation parameters and assume constant weekly spend "
        "(steady-state adstock). This approximates the marginal impact of reallocation; "
        "it is not an absolute forecast of total sales."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — The Data
# ════════════════════════════════════════════════════════════════════════════
with tab_ts:
    st.info(
        "This dashboard uses **synthetic** weekly marketing data (seed=42, 2 years, 4 channels). "
        "Because the data is simulated with known parameters, we can verify how accurately "
        "the model recovers the true ROI for each channel."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weeks", len(df))
    c2.metric("Weekly avg sales", f"£{df[target_col].mean():,.0f}")
    c3.metric("Total media spend", f"£{df[channel_cols].sum().sum():,.0f}")
    c4.metric("Date range", f"{df[date_col].dt.date.iloc[0]} → {df[date_col].dt.date.iloc[-1]}")

    if show_raw:
        st.dataframe(df, use_container_width=True, height=250)

    st.subheader("Sales & channel spends over time")

    if not selected_channels:
        st.info("Select at least one channel in the sidebar.")
    else:
        fig_ts = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Sales (£)", "Channel spends"),
            vertical_spacing=0.08, row_heights=[0.45, 0.55],
        )
        fig_ts.add_trace(
            go.Scatter(
                x=df[date_col], y=df[target_col],
                mode="lines", name="Sales",
                line=dict(color="#636EFA", width=2),
                fill="tozeroy", fillcolor="rgba(99,110,250,0.10)",
            ), row=1, col=1,
        )
        for i, ch in enumerate(selected_channels):
            s = df[ch].copy()
            if normalize:
                mn, mx = s.min(), s.max()
                if mx > mn:
                    s = (s - mn) / (mx - mn)
            fig_ts.add_trace(
                go.Bar(x=df[date_col], y=s,
                       name=CHANNEL_LABELS.get(ch, ch),
                       marker_color=PALETTE[(i + 1) % len(PALETTE)], opacity=0.75),
                row=2, col=1,
            )
        fig_ts.update_layout(
            height=560, barmode="stack", template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50, b=30),
        )
        fig_ts.update_yaxes(title_text="Sales (£)", row=1, col=1)
        fig_ts.update_yaxes(title_text="Spend (norm.)" if normalize else "Spend (£)", row=2, col=1)
        st.plotly_chart(fig_ts, use_container_width=True)

    with st.expander("How the data was generated"):
        st.markdown("""
The dataset is generated from scratch with realistic marketing dynamics:

- **Adstock (carry-over):** Spend from previous weeks continues to have an effect.
  TV has high carry-over (decay=0.70 — 70% of the effect persists next week); Digital is
  more immediate (decay=0.40).

- **Saturation (diminishing returns):** Extra spend yields less additional return as budgets grow.
  Modelled with a Hill curve where `half_sat` is the spend level at which 50% of the maximum
  effect is reached. OOH saturates early (£6k); TV saturates later (£30k).

- **Seasonality:** Annual (±20%) and quarterly (±8%) cycles.

- **True ROI:** Digital (£1.10) > Social (£0.90) > TV (£0.55) > OOH (£0.40) per £ spent.
  The model's job is to recover these values from the noisy observed data.
        """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Explore
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.info(
        "Raw correlations between spend and sales understate channel impact because they "
        "ignore adstock carry-over and diminishing returns — which is exactly why MMM exists. "
        "Use these charts to build intuition about the raw data before modelling."
    )

    # ── Correlation heatmap ────────────────────────────────────────────────
    st.subheader("Pearson correlations")
    cols_for_corr = channel_cols + [target_col]
    rename_map = {**CHANNEL_LABELS, "sales": "Sales"}
    corr = df[cols_for_corr].rename(columns=rename_map).corr()

    fig_heat = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}", textfont=dict(size=12),
    ))
    fig_heat.update_layout(height=400, template="plotly_white", margin=dict(t=20, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Channel spend vs. sales")
    cols_sc = st.columns(2)
    for idx, ch in enumerate(channel_cols):
        with cols_sc[idx % 2]:
            fig_sc = go.Figure(go.Scatter(
                x=df[ch], y=df[target_col], mode="markers",
                marker=dict(opacity=0.5, color=PALETTE[(idx + 1) % len(PALETTE)]),
                name=CHANNEL_LABELS[ch],
            ))
            fig_sc.update_layout(
                title=f"{CHANNEL_LABELS[ch]} spend vs Sales", height=270,
                xaxis_title=f"{CHANNEL_LABELS[ch]} spend (£)", yaxis_title="Sales (£)",
                template="plotly_white", margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()

    # ── Distributions ──────────────────────────────────────────────────────
    st.subheader("Distributions")
    st.caption("Weekly distribution of sales and channel spend over the 2-year period.")

    all_cols   = [target_col] + channel_cols
    col_labels = ["Sales"] + [CHANNEL_LABELS[ch] for ch in channel_cols]
    dist_cols  = st.columns(2)
    for idx, (col, label) in enumerate(zip(all_cols, col_labels)):
        with dist_cols[idx % 2]:
            fig_h = go.Figure(go.Histogram(
                x=df[col], nbinsx=30,
                marker_color=PALETTE[idx % len(PALETTE)], opacity=0.8,
            ))
            fig_h.update_layout(
                title=label, height=240, template="plotly_white",
                margin=dict(t=40, b=20), showlegend=False,
            )
            st.plotly_chart(fig_h, use_container_width=True)

    with st.expander("Summary statistics"):
        st.dataframe(
            df[all_cols].rename(columns={**CHANNEL_LABELS, "sales": "Sales"})
            .describe().T.style.format("{:.1f}"),
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Cloud Sync
# ════════════════════════════════════════════════════════════════════════════
with tab_bq:
    st.markdown("### BigQuery Integration")
    st.markdown(
        "Push training data and model results to **Google BigQuery** to power "
        "natural-language Q&A in the Ask Gemini tab. Requires a GCP project configured in `.env`."
    )

    if bq_project:
        st.success("Connected to BigQuery")
    else:
        st.warning(
            "No GCP project configured. Set `GOOGLE_CLOUD_PROJECT` and `BQ_DATASET` in `.env` to enable sync."
        )

    has_results = "summary" in st.session_state
    has_fitted  = "fitted_values" in st.session_state

    st.subheader("Upload to BigQuery")
    bq_col1, bq_col2, bq_col3 = st.columns(3)

    with bq_col1:
        st.markdown("**Training data**")
        st.caption("104 weeks × 5 columns (date, 4 channels, sales)")
        if st.button("⬆ Upload training data", disabled=not bq_project):
            with st.spinner("Uploading…"):
                upload_to_bq(df, "mmm_training_data", bq_project, bq_dataset)
            st.success(f"Done — `mmm_training_data`")

    with bq_col2:
        st.markdown("**Model results**")
        st.caption("Channel contributions, ROI, and spend summary")
        if st.button("⬆ Upload model results", disabled=not (bq_project and has_results)):
            with st.spinner("Uploading…"):
                upload_to_bq(st.session_state["summary"], "mmm_model_results", bq_project, bq_dataset)
            st.success(f"Done — `mmm_model_results`")

    with bq_col3:
        st.markdown("**Fitted values**")
        st.caption("Weekly predicted vs actual sales")
        if st.button("⬆ Upload fitted values", disabled=not (bq_project and has_fitted)):
            with st.spinner("Uploading…"):
                upload_to_bq(st.session_state["fitted_values"], "mmm_fitted_values", bq_project, bq_dataset)
            st.success(f"Done — `mmm_fitted_values`")

    if not has_results:
        st.caption("Results are computed automatically on startup.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Ask Gemini
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### Ask Gemini about your MMM data")
    st.markdown(
        "Ask natural-language questions about the training data or model results stored in BigQuery. "
        "Gemini generates SQL, runs it against your tables, and returns insights with charts."
    )

    if not bq_project:
        st.warning("Configure a GCP project in `.env` to enable this feature.")
    else:
        st.caption("**Example questions:**")
        eq1, eq2, eq3 = st.columns(3)
        eq1.info("Which week had the highest total sales?")
        eq2.info("Which channel has the best ROI?")
        eq3.info("How did TV spend vary by quarter?")
        st.divider()

        MAX_QUESTIONS = 3

        if "messages" not in st.session_state:
            st.session_state.messages = []

        questions_asked = sum(1 for m in st.session_state.messages if m["role"] == "user")

        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                is_last = i == len(st.session_state.messages) - 1
                if is_last and msg["role"] == "assistant":
                    for vega_spec in st.session_state.pop("_last_charts", []):
                        try:
                            _render_vega_as_plotly(vega_spec)
                        except Exception as e:
                            st.warning(f"Could not render chart: {e}")
                    if "_last_debug" in st.session_state:
                        with st.expander("Raw error details"):
                            st.json(st.session_state.pop("_last_debug"))

        if questions_asked >= MAX_QUESTIONS:
            st.info(f"Reached the {MAX_QUESTIONS}-question limit for this session. Refresh to start a new conversation.")
        elif prompt := st.chat_input("e.g. Which channel drove the most sales this year?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            prior = st.session_state.messages[:-1]
            with st.spinner("Querying BigQuery agent…"):
                response = ask_bq_agent(prompt, bq_project, bq_dataset, history=prior)
            answer = format_response(response)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            if response.get("charts"):
                st.session_state["_last_charts"] = response["charts"]
            if "_debug" in response:
                st.session_state["_last_debug"] = response["_debug"]
            st.rerun()
