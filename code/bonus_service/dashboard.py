from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = os.getenv("FOOTBALL_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Football Forecasting Live Replay", layout="wide")
st.title("Forecasting Competitive Football — Live Replay")
st.caption("Bonus dashboard backed by the same Gold snapshot representation used during training.")

@st.cache_data(ttl=30)
def load_matches():
    with httpx.Client(timeout=20) as client:
        return client.get(f"{API_URL}/matches", params={"split": "test"}).raise_for_status().json()

@st.cache_data(ttl=30)
def load_replay(match_id: int):
    with httpx.Client(timeout=60) as client:
        return client.get(f"{API_URL}/replay/{match_id}").raise_for_status().json()

try:
    matches = load_matches()
except Exception as exc:
    st.error(f"API unavailable at {API_URL}: {exc}")
    st.stop()

if not matches:
    st.warning("No held-out matches were returned by the service.")
    st.stop()

labels = {f"{m['home_team']} vs {m['away_team']} — {m['match_id']}": m for m in matches}
selected_label = st.selectbox("Held-out match", list(labels))
selected = labels[selected_label]
replay = load_replay(selected["match_id"])
frame = pd.DataFrame(replay)

# Flatten probability columns for charting.
for name in ["home", "draw", "away"]:
    frame[f"p_{name}"] = frame["outcome_probabilities"].apply(lambda d: d[name])
frame["home_score"] = frame["score"].apply(lambda d: d["home"])
frame["away_score"] = frame["score"].apply(lambda d: d["away"])
frame["home_red"] = frame["red_cards"].apply(lambda d: d["home"])
frame["away_red"] = frame["red_cards"].apply(lambda d: d["away"])

# Infer key events from changes between snapshots. This avoids a second event parser in serving code.
frame["goal_event"] = (frame["home_score"].diff().fillna(0) != 0) | (frame["away_score"].diff().fillna(0) != 0)
frame["card_event"] = (frame["home_red"].diff().fillna(0) != 0) | (frame["away_red"].diff().fillna(0) != 0)

latest_idx = st.slider("Replay position", 0, len(frame) - 1, len(frame) - 1)
shown = frame.iloc[: latest_idx + 1].copy()
latest = shown.iloc[-1]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Score", f"{latest.home_score}–{latest.away_score}")
m2.metric("Expected final margin", f"{latest.expected_final_margin:+.2f}")
m3.metric("Home win", f"{latest.p_home:.1%}")
m4.metric("Snapshot", latest.snapshot_id)

prob_long = shown.melt(
    id_vars=["snapshot_rank"],
    value_vars=["p_home", "p_draw", "p_away"],
    var_name="Outcome",
    value_name="Probability",
)
prob_long["Outcome"] = prob_long["Outcome"].str.replace("p_", "", regex=False).str.title()
fig_prob = px.line(prob_long, x="snapshot_rank", y="Probability", color="Outcome", markers=True,
                   title="Live outcome probabilities")
fig_prob.update_yaxes(range=[0, 1])
for _, row in shown[shown["goal_event"]].iterrows():
    fig_prob.add_vline(x=float(row["snapshot_rank"]), line_dash="dash", annotation_text="Goal")
for _, row in shown[shown["card_event"]].iterrows():
    fig_prob.add_vline(x=float(row["snapshot_rank"]), line_dash="dot", annotation_text="Red card")
st.plotly_chart(fig_prob, use_container_width=True)

fig_margin = px.line(shown, x="snapshot_rank", y="expected_final_margin", markers=True,
                     title="Expected final goal margin (home − away)")
fig_margin.add_hline(y=0, line_dash="dash")
st.plotly_chart(fig_margin, use_container_width=True)

st.subheader("Latest SHAP explanation")
shap_rows = latest["top_shap"]
shap_df = pd.DataFrame(shap_rows)
if "detail" in shap_df.columns and len(shap_df) == 1:
    st.warning(shap_df.iloc[0]["detail"])
else:
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=True)
    fig_shap = px.bar(shap_df, x="shap_value", y="feature", orientation="h",
                      title="Top TreeSHAP attributions for the served outcome model")
    st.plotly_chart(fig_shap, use_container_width=True)

with st.expander("Serving details"):
    st.json({
        "feature_source": latest["feature_source"],
        "outcome_model": latest["outcome_model"],
        "margin_model": latest["margin_model"],
        "model_inference_ms": latest["model_inference_ms"],
        "api": API_URL,
    })
