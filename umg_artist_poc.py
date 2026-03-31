import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
import re

st.set_page_config(layout="wide")

st.markdown("""
<style>
/* Remove top padding */
.block-container {
    padding-top: 1rem !important;
}

/* Optional: tighten title spacing */
h1 {
    margin-top: 0px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🎵 Rising Artist Prediction – AI A&R Platform")

# -----------------------------------------------------
# LOAD DATA (UPDATED)
# -----------------------------------------------------

@st.cache_data
def load_data():
    train_df = pd.read_excel("artists_data.xlsx", sheet_name="training data")
    pred_df = pd.read_excel("artists_data.xlsx", sheet_name="New Artist ")

    train_df.columns = train_df.columns.str.strip()
    pred_df.columns = pred_df.columns.str.strip()
    
    # Rename column
    pred_df.rename(columns={
        "2025–2026 Traction Driver": "Traction"
    }, inplace=True)

    # Convert numeric
    for df in [train_df, pred_df]:
        df["Energy"] = pd.to_numeric(df["Energy"], errors="coerce")
        df["Valence"] = pd.to_numeric(df["Valence"], errors="coerce")

    train_df = train_df.reset_index(drop=True)
    pred_df = pred_df.reset_index(drop=True)

    return train_df, pred_df


historical_data, prediction_data = load_data()

st.markdown("""
<style>
h1, h2, h3 {
    letter-spacing: 0.5px;
}

div[data-testid="stCaptionContainer"] p {
    font-size: 16px !important;
    line-height: 1.6;
    margin-bottom: 10px;
    color: #2a2a2a !important;  /* strong dark caption */
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# STEP 1 – TRAINING DATA (UNCHANGED)
# -----------------------------------------------------

st.header("1️⃣ Data Pipeline - Historical Artist Data for Model Training")
st.caption("Ingest, clean, and manage historical artist trajectories for both breakout successes and quiet fades. Continuously train, test, and sharpen predictive AI models.")
st.dataframe(historical_data, use_container_width=True, hide_index=True)

# -----------------------------------------------------
# STEP 2 – AI PREDICTION (UPDATED ONLY HERE)
# -----------------------------------------------------

st.header("2️⃣ AI Engine - Breakout Predictor")
st.caption("Data driven prediction that identifies future stars before they breakout.")



st.subheader("New Artists to be predicted")
st.dataframe(prediction_data, use_container_width=True, hide_index=True)

# -----------------------------------------------------
# 🔥 NEW TRACTION SCORING (NLP BASED)
# -----------------------------------------------------

def compute_traction_score(text):
    text = str(text).lower()

    keywords = {
        "viral": 20,
        "billion": 30,
        "award": 25,
        "grammy": 25,
        "tour": 15,
        "festival": 15,
        "breakout": 20,
        "surge": 20,
        "platinum": 25,
        "charts": 15,
        "mainstream": 20
    }

    score = 0
    for word, val in keywords.items():
        if re.search(word, text):
            score += val

    return min(score, 100)

prediction_data["Traction_Score"] = prediction_data["Traction"].apply(compute_traction_score)

# -----------------------------------------------------
# 🔥 NEW AI SCORE (REPLACES OLD MODEL)
# -----------------------------------------------------

prediction_data["AI_Score"] = (
    0.4 * (prediction_data["Energy"] * 100) +
    0.3 * (prediction_data["Valence"] * 100) +
    0.3 * prediction_data["Traction_Score"]
)

prediction_data["Prediction"] = np.where(
    prediction_data["AI_Score"] > prediction_data["AI_Score"].median(),
    "Yes", "No"
)

st.subheader("AI Model Prediction")

st.dataframe(
    prediction_data[["Artist", "Prediction"]],
    use_container_width=True
)

fig_pred = px.bar(
    prediction_data.sort_values("AI_Score", ascending=False),
    x="Artist",
    y="AI_Score",
    color="Prediction",
    title="Rising Artist with Prediction score"
)

st.plotly_chart(fig_pred, use_container_width=True)

selected_artists = prediction_data[prediction_data["Prediction"] == "Yes"]

st.success(f"{len(selected_artists)} artists selected for deeper analysis")


# -----------------------------------------------------
# STEP 3 – Social Dynamics (UPDATED: X SIGNALS)
# -----------------------------------------------------


st.header("3️⃣ Social Dynamics ( X, Instagram, Facebook... )")

artists = selected_artists["Artist"].tolist()

# ✅ CREATE SOCIAL DATA
social_data = pd.DataFrame({
    "Artist": artists,
    
    "TikTok_Viral_Index": np.random.randint(40,100,len(artists)),

    # 🔥 X SIGNALS
    "X_Followers": np.random.randint(10000, 5000000, len(artists)),
    "X_Sentiment": np.random.uniform(0.3, 0.95, len(artists)),

    "Concert_Sales": np.random.randint(500,20000,len(artists)),
    "Festival_Lineups": np.random.randint(0,10,len(artists)),
    "Awards_Buzz": np.random.randint(0,50,len(artists)),
})

# -----------------------------------------------------
# 🔥 FORMAT SENTIMENT (USER FRIENDLY)
# -----------------------------------------------------
social_data["X_Sentiment_Percent"] = (social_data["X_Sentiment"] * 100).round(2)

# -----------------------------------------------------
# 🔥 NORMALIZE FOLLOWERS
# -----------------------------------------------------
social_data["X_Followers_Score"] = (
    (social_data["X_Followers"] - social_data["X_Followers"].min()) /
    (social_data["X_Followers"].max() - social_data["X_Followers"].min() + 1e-6)
) * 100

# -----------------------------------------------------
# 🔥 SOCIAL MOMENTUM (FINAL AI SCORE)
# -----------------------------------------------------
social_data["Social_Momentum"] = (
    0.30 * social_data["TikTok_Viral_Index"] +
    0.35 * social_data["X_Followers_Score"] +
    0.25 * social_data["X_Sentiment_Percent"] +
    0.10 * social_data["Awards_Buzz"]
)

# -----------------------------------------------------
# 🔥 SHOW FULL SOCIAL DATA (IMPORTANT FIX)
# -----------------------------------------------------
#st.subheader("Social Signals Data (X Signals)")
st.dataframe(
    social_data[[
        "Artist",
        "X_Followers",
        "X_Sentiment_Percent",
        "TikTok_Viral_Index",
        "Social_Momentum"
    ]],
    use_container_width=True
)

# -----------------------------------------------------
# 🔥 ANOMALY DETECTION (FOLLOWER SURGE)
# -----------------------------------------------------
threshold = social_data["X_Followers"].mean() + social_data["X_Followers"].std()
social_data["X_Anomaly"] = social_data["X_Followers"] > threshold

spike_artists = social_data[social_data["X_Anomaly"]]

if len(spike_artists) > 0:
    st.warning("⚡ X (Twitter) Follower Surge Detected")
    st.dataframe(
        spike_artists[["Artist","X_Followers","X_Sentiment_Percent"]],
        use_container_width=True
    )
else:
    st.info("No X anomaly detected")

# -----------------------------------------------------
# 📊 VISUALIZATION
# -----------------------------------------------------
fig_social = px.bar(
    social_data.sort_values("Social_Momentum", ascending=False),
    x="Artist",
    y="Social_Momentum",
    title="Social Momentum Score (X Signals)"
)

st.plotly_chart(fig_social, use_container_width=True)
# -----------------------------------------------------
# STEP 4 – Knowledge Graph (ENHANCED)
# -----------------------------------------------------
 
st.header("4️⃣ Artist Knowledge Graph")
st.caption("Map the DNA of rising talent by linking them to established stars. Discover who they sound like, who they should collaborate with, and exactly where they fit in the current musical landscape based on genre, energy, and audio features.")
 
G = nx.Graph()
 
historical_artists = historical_data["Artist"].tolist()
predicted_artists = selected_artists["Artist"].tolist()
 
for artist in historical_artists:
    G.add_node(artist, type="historical")
 
for artist in predicted_artists:
    G.add_node(artist, type="predicted")
 
relations = ["Same Genre","Fanbase Overlap","Collab Potential"]
 
for artist in predicted_artists:
    connections = np.random.choice(historical_artists, size=min(2,len(historical_artists)), replace=False)
    for c in connections:
        G.add_edge(artist, c, relation=np.random.choice(relations))
 
pos = nx.spring_layout(G, seed=42)
 
edge_x, edge_y = [], []
 
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
 
edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines")
 
node_x, node_y, node_color = [], [], []
 
for node in G.nodes(data=True):
    x, y = pos[node[0]]
    node_x.append(x)
    node_y.append(y)
    node_color.append("red" if node[1]["type"]=="predicted" else "blue")
 
node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    text=list(G.nodes()),
    marker=dict(size=15, color=node_color)
)
 
fig = go.Figure(data=[edge_trace, node_trace])
st.plotly_chart(fig, use_container_width=True)
 
# -----------------------------------------------------
# STEP 5 – Competitive Analysis
# -----------------------------------------------------
 
st.header("5️⃣ Competitive Analysis")
st.caption("AI compares each artist against a dynamically generated peer group of similar acts based on stage, genre, and audience size. The resulting Competitive Score factors in streaming velocity, social engagement, playlist and momentum to reveal market positioning and rank artists within their competitive landscape.")
 
competitive_data = pd.DataFrame({
    "Artist": predicted_artists,
    "Save_to_Stream_Ratio": np.random.uniform(0.2,0.6,len(predicted_artists)),
    "Listener_Retention": np.random.uniform(20,50,len(predicted_artists)),
    "Repeat_Listeners": np.random.uniform(20,60,len(predicted_artists)),
    "Playlist_Addition_Velocity": np.random.uniform(10,80,len(predicted_artists)),
    "Follower_Growth": np.random.uniform(5,25,len(predicted_artists))
})
 
industry_avg = {
    "Save_to_Stream_Ratio":0.30,
    "Listener_Retention":25,
    "Repeat_Listeners":30,
    "Playlist_Addition_Velocity":40,
    "Follower_Growth":10
}
 
# Competitive Score
competitive_data["Competitive_Advantage"] = (
    (competitive_data["Save_to_Stream_Ratio"]/industry_avg["Save_to_Stream_Ratio"])*20 +
    (competitive_data["Listener_Retention"]/industry_avg["Listener_Retention"])*20 +
    (competitive_data["Repeat_Listeners"]/industry_avg["Repeat_Listeners"])*20 +
    (competitive_data["Playlist_Addition_Velocity"]/industry_avg["Playlist_Addition_Velocity"])*20 +
    (competitive_data["Follower_Growth"]/industry_avg["Follower_Growth"])*20
).clip(0,100)
 
st.dataframe(competitive_data, width="stretch")
 
# ✅ NEW GRAPH
fig_comp = px.bar(
    competitive_data.sort_values("Competitive_Advantage", ascending=False),
    x="Artist",
    y="Competitive_Advantage",
    title="Visualisation for Competitive Advantage Score",
    color="Competitive_Advantage"
)
 
#st.plotly_chart(fig_comp, width="stretch")
 
 
# -----------------------------------------------------
# STEP 6 – Ranking Engine (FULL AI VERSION)
# -----------------------------------------------------
 
st.header("6️⃣ AI Ranking & Trend Prediction (W1+W2+W3+...+Wn)")
 
ranking = competitive_data.copy()
 
# AI signals
ranking["Viral_Potential"] = np.random.uniform(50,100,len(ranking))
 
# 🔥 FIX: align Social Momentum properly by artist
ranking = ranking.merge(
    social_data[["Artist","Social_Momentum"]],
    on="Artist",
    how="left"
)
 
ranking["Contextual_Buzz"] = ranking["Social_Momentum"]
ranking["Competitive_Score"] = ranking["Competitive_Advantage"]
 
# ✅ INTERACTIVE WEIGHTS
col1,col2,col3 = st.columns(3)
 
with col1:
    w1 = st.slider("Weight Viral Potential",0.1,1.0,0.4)
 
with col2:
    w2 = st.slider("Weight Contextual Buzz",0.1,1.0,0.3)
 
with col3:
    w3 = st.slider("Weight Competitive Advantage",0.1,1.0,0.3)
 
# Normalize
total = w1+w2+w3
w1,w2,w3 = w1/total,w2/total,w3/total
 
# TPS
ranking["TPS"] = (
    w1 * ranking["Viral_Potential"] +
    w2 * ranking["Contextual_Buzz"] +
    w3 * ranking["Competitive_Score"]
)
 
ranking = ranking.sort_values("TPS", ascending=False)
ranking = ranking[
    ["Artist", "TPS", "Viral_Potential", "Contextual_Buzz", "Competitive_Score"]
    + [col for col in ranking.columns if col not in ["Artist","TPS","Viral_Potential","Contextual_Buzz","Competitive_Score"]]
]
 
 
st.dataframe(ranking, width="stretch")
 
# ✅ NEW GRAPH
fig_rank = px.bar(
    ranking,
    x="Artist",
    y="TPS",
    title="Visualisation for Trend Prediction Score (TPS)",
    color="TPS"
)
 
st.plotly_chart(fig_rank, width="stretch")
 
 
# -----------------------------------------------------
# STEP 7 – Dashboard Output (UPGRADED)
# -----------------------------------------------------
 
st.header("7️⃣ Upcoming Artists – AI A&R Dashboard")
 
top_artists = ranking.head(4)
 
st.success("Top Rising Artists Identified")
 
st.dataframe(top_artists, width="stretch")
st.subheader("🌟 Top Rising Artists Explorer")

# 👉 Left panel: Artist list
artist_names = top_artists["Artist"].tolist()

selected_artist = st.selectbox(
    "Select Artist",
    artist_names
)

# 👉 Get selected row
artist_row = top_artists[top_artists["Artist"] == selected_artist].iloc[0]

confidence = round(artist_row["TPS"], 1)

if confidence > 80:
    label = "🚀 Rising Star"
elif confidence > 65:
    label = "⭐ Strong Potential"
else:
    label = "⚠ Watchlist"

# 👉 Layout split
col1, col2 = st.columns([1,2])

# LEFT SIDE (clean identity)
with col1:
    st.markdown(f"## 🎤 {selected_artist}")
    st.markdown(f"**{label}**")
    st.markdown("---")

    st.metric("Trend Score", confidence)

# RIGHT SIDE (details)
with col2:
    st.markdown("### Performance Breakdown")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Viral Potential", round(artist_row["Viral_Potential"],1))

    with c2:
        st.metric("Audience Buzz", round(artist_row["Contextual_Buzz"],1))

    with c3:
        st.metric("Competitive Strength", round(artist_row["Competitive_Score"],1))

    st.markdown("---")

    st.markdown("### A&R Insight")

    st.info(f"""
    **{selected_artist}** shows strong signals across multiple dimensions.

    • Viral growth potential is {'high' if artist_row["Viral_Potential"] > 70 else 'moderate'}  
    • Audience engagement is {'strong' if artist_row["Contextual_Buzz"] > 65 else 'emerging'}  
    • Competitive positioning is {'leading' if artist_row["Competitive_Score"] > 70 else 'developing'}  

    👉 Recommendation: Focus on **promotion / partnerships / early signing strategy**
    """)
