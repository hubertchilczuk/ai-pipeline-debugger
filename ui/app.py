"""Streamlit dashboard for ad-hoc log debugging."""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")

st.set_page_config(page_title="AI Pipeline Debugger", layout="wide")
st.title("AI Pipeline Debugger")


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


with st.sidebar:
    st.header("Pipeline context")
    pipeline = st.text_input("Pipeline", value="etl_users_daily")
    stage = st.text_input("Stage", value="transform_users")
    if st.button("Check API health"):
        try:
            resp = httpx.get(f"{API_URL}/health", timeout=5, headers=_headers()).json()
            st.json(resp)
        except httpx.HTTPError as exc:
            st.error(f"API unreachable: {exc}")

log_text = st.text_area("Paste log excerpt", height=300)

if st.button("Analyze", type="primary", disabled=not log_text):
    with st.spinner("Calling /analyze..."):
        try:
            resp = httpx.post(
                f"{API_URL}/analyze",
                json={"pipeline": pipeline, "stage": stage, "log_excerpt": log_text},
                timeout=120,
                headers=_headers(),
            )
        except httpx.TimeoutException:
            st.error("Request timed out after 120s. Try a shorter log excerpt or switch LLM mode.")
            st.stop()
        except httpx.HTTPError as exc:
            st.error(f"API request failed: {exc}")
            st.stop()

    if resp.status_code != 200:
        st.error(f"{resp.status_code}: {resp.text}")
    else:
        data = resp.json()
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(data["error_type"])
            st.markdown(f"**Severity:** `{data['severity']}` · **Confidence:** `{data['confidence']:.2f}`")
            st.markdown("### Root cause")
            st.write(data["root_cause"])
            st.markdown("### Suggested fix")
            st.write(data["suggested_fix"])
            if data.get("incident_id"):
                with st.expander("Provide feedback"):
                    helpful = st.radio("Was this helpful?", ["yes", "no"], horizontal=True)
                    actual = st.text_area("Actual fix (optional)")
                    notes = st.text_area("Notes (optional)")
                    if st.button("Submit feedback"):
                        try:
                            fb = httpx.post(
                                f"{API_URL}/feedback",
                                json={
                                    "incident_id": data["incident_id"],
                                    "helpful": helpful == "yes",
                                    "actual_fix": actual or None,
                                    "notes": notes or None,
                                },
                                timeout=10,
                                headers=_headers(),
                            )
                            if fb.status_code in (200, 204):
                                st.success("Feedback saved.")
                            else:
                                st.error(f"{fb.status_code}: {fb.text}")
                        except httpx.HTTPError as exc:
                            st.error(f"Feedback failed: {exc}")
        with col2:
            st.markdown("### LLM trace")
            st.json(data["llm"])
            st.markdown("### Similar incidents")
            for hit in data["similar_incidents"]:
                st.markdown(f"- `{hit['similarity']:.2f}` — {hit['error_type']}")
