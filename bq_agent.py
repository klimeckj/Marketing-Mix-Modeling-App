"""
BigQuery integration for the MMM Playground.

Provides:
  - upload_to_bq()   : push a DataFrame to a BQ table (replace)
  - ask_bq_agent()   : call Google's Conversational Analytics API
  - format_response(): render the API response as a Markdown string
"""

import json
import os

import pandas as pd
import requests


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_credentials():
    """Return refreshed service account credentials.

    Checks for a ``[gcp_service_account]`` section in ``st.secrets`` first
    (Streamlit Cloud), then falls back to ``GOOGLE_APPLICATION_CREDENTIALS``
    file (local dev).
    """
    from google.oauth2 import service_account
    import google.auth.transport.requests

    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

    # Streamlit Cloud: credentials embedded in st.secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
            creds = service_account.Credentials.from_service_account_info(
                dict(st.secrets["gcp_service_account"]),
                scopes=SCOPES,
            )
            creds.refresh(google.auth.transport.requests.Request())
            return creds
    except ImportError:
        pass  # streamlit not available, fall through to local dev path

    # Local dev: credentials file
    creds = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        scopes=SCOPES,
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds


# ── Public API ─────────────────────────────────────────────────────────────────

def upload_to_bq(
    df: pd.DataFrame,
    table_id: str,
    project_id: str,
    dataset_id: str,
) -> None:
    """Upload *df* to BigQuery as ``project_id.dataset_id.table_id`` (replace)."""
    from google.cloud import bigquery
    import pandas_gbq  # lazy import – only needed for upload

    creds = _get_credentials()

    # Auto-create dataset if it doesn't exist
    client = bigquery.Client(project=project_id, credentials=creds)
    dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
    dataset.location = "EU"
    client.create_dataset(dataset, exists_ok=True)

    destination = f"{dataset_id}.{table_id}"
    pandas_gbq.to_gbq(
        df,
        destination_table=destination,
        project_id=project_id,
        if_exists="replace",
        progress_bar=False,
        credentials=creds,
    )


def load_from_bq(
    table_id: str,
    project_id: str,
    dataset_id: str,
) -> "pd.DataFrame | None":
    """Load a BigQuery table into a DataFrame. Returns None on any error."""
    try:
        import pandas_gbq

        creds = _get_credentials()
        return pandas_gbq.read_gbq(
            f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`",
            project_id=project_id,
            credentials=creds,
            progress_bar_type=None,
        )
    except Exception:
        return None


def ask_bq_agent(
    question: str,
    project_id: str,
    dataset_id: str,
    history: list = None,
    agent_id: str = os.environ.get("BQ_AGENT_ID", ""),
) -> dict:
    """
    Ask a natural-language question via the Gemini Data Analytics API
    (geminidataanalytics.googleapis.com).

    Targets the specific "mmm analyzer" agent resource so that resource-level
    IAM (Gemini Data Analytics Agent User) is respected.

    ``history`` is a list of {"role": "user"|"assistant", "content": "..."} dicts
    from previous turns. Up to the last 3 messages are injected as context.

    Returns a dict with at least:
      "answer" : str   – human-readable answer
      "sql"    : str   – generated SQL (may be empty)
      "raw"    : list  – parsed streaming response chunks for debugging
    """
    creds = _get_credentials()
    token = creds.token

    url = (
        f"https://geminidataanalytics.googleapis.com/v1beta"
        f"/projects/{project_id}/locations/global:chat"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-server-timeout": "300",
    }

    # Text format reminder prepended to every question.
    # Keep this to text-only rules — chart format is controlled by the BQ agent's system instructions.
    FORMAT_PREFIX = (
        "[FORMAT: Answer in max 3 sentences. Lead with the key number or insight, not background.]\n\n"
    )

    # Build the question text, prepending recent conversation context if available
    if history:
        recent = history[-3:]
        context_lines = ["Previous conversation:"]
        for msg in recent:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            context_lines.append(f"{role_label}: {msg['content']}")
        context_lines.append(f"\nCurrent question: {question}")
        full_question = FORMAT_PREFIX + "\n".join(context_lines)
    else:
        full_question = FORMAT_PREFIX + question

    payload = {
        "parent": f"projects/{project_id}/locations/global",
        "messages": [
            {"userMessage": {"text": full_question}}
        ],
        "inline_context": {
            "datasource_references": {
                "bq": {
                    "tableReferences": [
                        {
                            "projectId": project_id,
                            "datasetId": dataset_id,
                            "tableId": "mmm_training_data",
                        },
                        {
                            "projectId": project_id,
                            "datasetId": dataset_id,
                            "tableId": "mmm_model_results",
                        },
                    ]
                }
            }
        },
    }

    response = requests.post(url, headers=headers, json=payload, timeout=300)

    if not response.ok:
        return {
            "answer": f"API error {response.status_code}: {response.text}",
            "sql": "",
            "raw": {},
            "_debug": {
                "status": response.status_code,
                "body": response.text,
                "headers": dict(response.headers),
            },
        }

    # Response is a JSON array of message objects.
    # We collect all parts where textType == "FINAL_RESPONSE".
    raw_parts: list = []
    answer_parts: list[str] = []
    sql_text = ""

    try:
        raw_parts = response.json()
    except json.JSONDecodeError:
        raw_parts = []

    vega_specs: list = []

    for chunk in raw_parts:
        sys_msg = chunk.get("systemMessage", {})

        # Text parts (final response narrative)
        text_obj = sys_msg.get("text", {})
        if text_obj.get("textType") == "FINAL_RESPONSE":
            for part in text_obj.get("parts", []):
                if isinstance(part, str) and part:
                    answer_parts.append(part)

        # SQL generated by the agent
        data_obj = sys_msg.get("data", {})
        if not sql_text and "generatedSql" in data_obj:
            sql_text = data_obj["generatedSql"]

        # Vega-Lite chart spec
        chart_obj = sys_msg.get("chart", {})
        vega_config = chart_obj.get("result", {}).get("vegaConfig")
        if vega_config:
            vega_specs.append(vega_config)

    answer_text = "\n".join(answer_parts) if answer_parts else "_No answer returned._"

    return {
        "answer": answer_text,
        "sql": sql_text,
        "raw": raw_parts,
        "charts": vega_specs,
    }


def create_orchestrator_session(orchestrator_url: str, user_id: str = "streamlit_user") -> str:
    """
    Register a new session with the ADK orchestrator and return the server-assigned session ID.

    Must be called once before the first ``call_orchestrator`` request.
    The returned ID should be stored in ``st.session_state`` and reused for
    the lifetime of the browser session.
    """
    url = f"{orchestrator_url.rstrip('/')}/apps/my_agent/users/{user_id}/sessions"
    resp = requests.post(url, json={}, timeout=30)
    resp.raise_for_status()
    return resp.json()["id"]


def call_orchestrator(question: str, session_id: str, orchestrator_url: str) -> dict:
    """
    Route a question through the ADK orchestrator running on Cloud Run.

    The orchestrator decides whether to answer directly or delegate to the
    BigQuery Conversational Analytics agent.  Returns the same shape as
    ``ask_bq_agent`` so the caller can treat both paths identically.

    ``session_id`` must be a server-side session ID obtained from
    ``create_orchestrator_session`` before the first call.
    """
    url = f"{orchestrator_url.rstrip('/')}/run"
    payload = {
        "app_name": "my_agent",
        "user_id": "streamlit_user",
        "session_id": session_id,
        "new_message": {
            "role": "user",
            "parts": [{"text": question}],
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"answer": f"Orchestrator error: {exc}", "charts": [], "sql": ""}

    events = resp.json() if isinstance(resp.json(), list) else []
    answer_text = ""
    thoughts = []
    bq_called = False

    for event in events:
        content = event.get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("functionCall", {}).get("name") == "call_bq_agent":
                bq_called = True
            if part.get("functionResponse", {}).get("name") == "call_bq_agent":
                bq_called = True
        if content.get("role") == "model":
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text", "").strip()
                if not text:
                    continue
                if part.get("thought") is True:
                    thoughts.append(text)
                else:
                    answer_text = text

    return {
        "answer": answer_text or "_No response received from orchestrator._",
        "charts": [],
        "sql": "",
        "thoughts": thoughts,
        "bq_called": bq_called,
    }


def format_response(api_response: dict) -> str:
    """
    Render an ``ask_bq_agent`` response as a Markdown string for Streamlit.
    Shows the answer text and, if present, the generated SQL in a code block.
    """
    return api_response.get("answer", "_No answer returned._")
