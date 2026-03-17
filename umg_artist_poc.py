import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("🎵 Rising Artist Prediction – AI A&R Platform")
#st.markdown("AI-powered pipeline using real + simulated intelligence layers")

# -----------------------------------------------------
# STEP 1 – Load Real Dataset
# -----------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_excel("artists_data.xlsx", sheet_name="Model training data")

    df.columns = df.columns.str.strip()

    df["Energy"] = pd.to_numeric(df["Energy"], errors="coerce")
    df["Valence"] = pd.to_numeric(df["Valence"], errors="coerce")

    df = df.dropna(subset=["Energy", "Valence"])

    # Feature engineering
    df["Follower_Growth"] = df["Energy"] * 100
    df["Engagement"] = df["Valence"] * 100

    def map_status(x):
        x = str(x)
        if "Global" in x: return 90
        elif "Icon" in x: return 85
        elif "Leader" in x: return 80
        elif "Breakout" in x: return 70
        elif "Star" in x: return 75
        else: return 60

    df["Industry_Signal"] = df["Status"].apply(map_status)

    return df

historical_data = load_data()

st.header("1️⃣ Data Overview")
st.dataframe(historical_data, use_container_width=True)

# -----------------------------------------------------
# Eligibility Filters (NEW)
# -----------------------------------------------------

st.subheader("Eligibility Filters")

eligible_artists = historical_data.copy()

st.success(f"{len(eligible_artists)} artists available")

# -----------------------------------------------------
# STEP 2 – AI Prediction Engine
# -----------------------------------------------------

st.header("2️⃣ AI Prediction – Rising Artist Forecast")

col1, col2, col3 = st.columns(3)

with col1:
    time_horizon = st.selectbox(
        "Prediction Horizon",
        ["30 Days", "60 Days", "90 Days", "180 Days", "365 Days"]
    )

with col2:
    genre_filter = st.selectbox(
        "Genre Filter",
        ["All"] + list(historical_data["Genre"].dropna().unique())
    )


filtered_data = eligible_artists.copy()

if genre_filter != "All":
    filtered_data = filtered_data[filtered_data["Genre"] == genre_filter]



st.subheader("Filtered Artist Data")
st.dataframe(filtered_data, use_container_width=True)

# -----------------------------------------------------
# KPI Calculations (ENHANCED)
# -----------------------------------------------------

filtered_data["Growth_Velocity"] = filtered_data["Follower_Growth"]
filtered_data["Engagement_Quality"] = filtered_data["Engagement"]

filtered_data["AI_Score"] = (
    0.4 * filtered_data["Growth_Velocity"] +
    0.35 * filtered_data["Engagement_Quality"] +
    0.25 * filtered_data["Industry_Signal"]
)

filtered_data["Prediction"] = np.where(
    filtered_data["AI_Score"] > filtered_data["AI_Score"].median(),
    "Yes", "No"
)

st.subheader("AI Model Prediction")

st.dataframe(
    filtered_data[["Artist", "Prediction"]],
    use_container_width=True
)

fig_pred = px.bar(
    filtered_data.sort_values("AI_Score", ascending=False),
    x="Artist",
    y="AI_Score",
    color="Prediction",
    title="Visualisation for Rising Artist Prediction Score"
)

st.plotly_chart(fig_pred, use_container_width=True)

selected_artists = filtered_data[filtered_data["Prediction"] == "Yes"]

st.success(f"{len(selected_artists)} artists selected for deeper analysis")

# -----------------------------------------------------
# STEP 3 – Social Dynamics (UPGRADED)
# -----------------------------------------------------

st.header("3️⃣ Social Dynamics – Multi Platform Signals")

artists = selected_artists["Artist"].tolist()

social_data = pd.DataFrame({
    "Artist": artists,
    "TikTok_Viral_Index": np.random.randint(40,100,len(artists)),
    "Twitter_Mentions": np.random.randint(200,5000,len(artists)),
    "Twitter_Sentiment": np.random.uniform(0.4,0.95,len(artists)),
    "Apple_Music_Streams": np.random.randint(50000,500000,len(artists)),
    "Amazon_Music_Streams": np.random.randint(20000,400000,len(artists)),
    "Concert_Sales": np.random.randint(500,20000,len(artists)),
    "Festival_Lineups": np.random.randint(0,10,len(artists)),
    "Awards_Buzz": np.random.randint(0,50,len(artists)),
})

# Spike detection
social_data["Streaming_Spike"] = (
    social_data["Apple_Music_Streams"] +
    social_data["Amazon_Music_Streams"]
) / 2

threshold = social_data["Streaming_Spike"].mean() + social_data["Streaming_Spike"].std()
social_data["Spike_Anomaly"] = social_data["Streaming_Spike"] > threshold

spike_artists = social_data[social_data["Spike_Anomaly"]]

if len(spike_artists) > 0:
    st.warning("⚡ Streaming Spike Detected")
    st.dataframe(spike_artists[["Artist","Streaming_Spike"]])
else:
    st.info("No spike anomaly detected")

# Social Momentum
social_data["Social_Momentum"] = (
    0.30 * social_data["TikTok_Viral_Index"] +
    0.20 * social_data["Streaming_Spike"] +
    0.20 * social_data["Concert_Sales"] +
    0.30 * social_data["Awards_Buzz"]
)

fig_social = px.bar(
    social_data.sort_values("Social_Momentum", ascending=False),
    x="Artist",
    y="Social_Momentum",
    title="Visualisation for Social Momentum Score"
)

st.plotly_chart(fig_social, use_container_width=True)

# -----------------------------------------------------
# STEP 4 – Knowledge Graph (ENHANCED)
# -----------------------------------------------------

st.header("4️⃣ Artist Knowledge Graph")

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

# ✅ CARD STYLE OUTPUT
for _,row in top_artists.iterrows():

    confidence = round(row["TPS"],1)

    if confidence > 80:
        label = "🚀 Rising Star"
    elif confidence > 65:
        label = "⭐ Strong Potential"
    else:
        label = "⚠ Watchlist Artist"

    st.markdown(f"""
### 🎤 {row['Artist']}

Prediction Confidence: **{confidence}/100**

Viral Potential Score: **{round(row['Viral_Potential'],1)}**

Contextual Buzz Score: **{round(row['Contextual_Buzz'],1)}**

Competitive Advantage Score: **{round(row['Competitive_Score'],1)}**

Status: **{label}**
""")


# -----------------------------------------------------
# A&R Decision Support (FINAL INSIGHT)
# -----------------------------------------------------

st.header("🎯 A&R Decision Insights")

top_candidate = ranking.iloc[0]

st.markdown(f"""

### Recommended Artist to Acquire

**Artist:** {top_candidate['Artist']}

**TPS Score:** {round(top_candidate['TPS'],1)}

### Justification

• Strong viral potential from streaming growth  
• High competitive advantage vs peer artists  
• Strong listener retention signals  

### A&R Recommendation

➡ Consider **signing / partnership / promotion campaign** for this artist.

""")
