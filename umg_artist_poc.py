import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

st.title("🎵 Rising Artist Prediction – AI A&R Platform")

st.markdown("Simulated **end-to-end pipeline** for predicting rising music artists")
# -----------------------------------------------------
# STEP 1 – Data Extraction + Historical Dataset
# -----------------------------------------------------

st.header("1️⃣ Data Extraction – Music Platforms")

platforms = pd.DataFrame({
    "Platform":[
        "Spotify","TikTok","Apple Music","Amazon Music","YouTube"
    ],
    "Data Signals":[
        "Streams, playlists, followers",
        "Viral trends, music usage",
        "Streams, listener demographics",
        "Streams, popularity signals",
        "Views, engagement"
    ]
})

st.dataframe(platforms,width="stretch")

st.subheader("Historical Artist Dataset")

historical_data = pd.DataFrame({

"Artist":[
"Neon Lights","Dream Pulse","Sky Echo",
"Digital Love","Future Noise","Midnight Wave"
],

"Genre":[
"Pop","EDM","Pop","Electronic","Hip-Hop","Pop"
],

"Country":[
"USA","UK","USA","Germany","Canada","USA"
],

"Streams":[1200000,950000,1400000,780000,880000,1100000],

"Follower_Growth":[12,9,15,7,8,11],

"Playlist_Additions":[150,120,210,80,95,140],

"Listening_Time":[70,65,75,60,62,72],

"Repeat_Listeners":[40,38,44,30,33,41],

"Save_Play_Ratio":[0.35,0.31,0.40,0.25,0.28,0.33],

"Influencer_Mentions":[20,18,25,10,11,19],

"Festival_Bookings":[4,3,5,1,2,3],

"Award_Buzz":[10,7,12,3,4,9],

"Artist_Age_Days":[200,180,220,140,160,210],

"Releases":[6,5,7,4,4,6],

"Activity_Days":[120,90,150,70,80,100]

})

st.dataframe(historical_data,width="stretch")

# -----------------------------------------------------
# Eligibility Filters
# -----------------------------------------------------

st.subheader("Eligibility Filters")

eligible_artists = historical_data[
(historical_data["Releases"] >= 4) &
(historical_data["Activity_Days"] >= 30) &
(historical_data["Artist_Age_Days"] >= 90)
]

st.success(f"{len(eligible_artists)} artists passed eligibility filters")

st.dataframe(eligible_artists,width="stretch")

# -----------------------------------------------------
# STEP 2 – AI Prediction Engine
# -----------------------------------------------------

st.header("2️⃣ AI Prediction – Rising Artist Forecast")

col1,col2,col3 = st.columns(3)

with col1:
    time_horizon = st.selectbox(
        "Prediction Horizon",
        ["30 Days","60 Days","90 Days","180 Days","365 Days"]
    )

with col2:
    genre_filter = st.selectbox(
        "Genre Filter",
        ["All","Pop","EDM","Hip-Hop","Electronic"]
    )

with col3:
    geo_filter = st.selectbox(
        "Geography Filter",
        ["Global","USA","UK","Canada","Germany"]
    )

filtered_data = eligible_artists.copy()

if genre_filter != "All":
    filtered_data = filtered_data[
        filtered_data["Genre"] == genre_filter
    ]

if geo_filter != "Global":
    filtered_data = filtered_data[
        filtered_data["Country"] == geo_filter
    ]

st.subheader("Filtered Artist Data")

st.dataframe(filtered_data,width="stretch")

# -----------------------------------------------------
# KPI Calculations
# -----------------------------------------------------

filtered_data["Growth_Velocity"] = (
filtered_data["Follower_Growth"] +
filtered_data["Playlist_Additions"]
)

filtered_data["Engagement_Quality"] = (
filtered_data["Listening_Time"] +
filtered_data["Repeat_Listeners"] +
filtered_data["Save_Play_Ratio"]*100
)

filtered_data["Industry_Signal"] = (
filtered_data["Influencer_Mentions"] +
filtered_data["Festival_Bookings"]*10 +
filtered_data["Award_Buzz"]
)

# -----------------------------------------------------
# AI Prediction Score
# -----------------------------------------------------

filtered_data["AI_Score"] = (
0.4 * filtered_data["Growth_Velocity"] +
0.35 * filtered_data["Engagement_Quality"] +
0.25 * filtered_data["Industry_Signal"]
)

filtered_data["Prediction"] = np.where(
filtered_data["AI_Score"] > filtered_data["AI_Score"].median(),
"Yes",
"No"
)

st.subheader("AI Model Prediction")

st.dataframe(
filtered_data[[
"Artist",
"AI_Score",
"Prediction"
]],
width="stretch"
)

fig_pred = px.bar(
filtered_data.sort_values("AI_Score",ascending=False),
x="Artist",
y="AI_Score",
color="Prediction",
title="AI Rising Artist Prediction Score"
)

st.plotly_chart(fig_pred,width="stretch")

# Artists selected for next pipeline steps
selected_artists = filtered_data[
filtered_data["Prediction"]=="Yes"
]

st.success(f"{len(selected_artists)} artists selected for deeper analysis")

st.dataframe(selected_artists,width="stretch")


# -----------------------------------------------------
# STEP 3 – Social Dynamics (Multi-Platform Intelligence)
# -----------------------------------------------------

st.header("3️⃣ Social Dynamics – Multi Platform Signals")

artists = selected_artists["Artist"].tolist()

# Simulated multi-platform signals
social_data = pd.DataFrame({
    "Artist": artists,

    # Viral platforms
    "TikTok_Viral_Index": np.random.randint(40,100,len(artists)),

    # Social platforms
    "Twitter_Mentions": np.random.randint(200,5000,len(artists)),
    "Twitter_Sentiment": np.random.uniform(0.4,0.95,len(artists)),
    "Facebook_Engagement": np.random.randint(1000,20000,len(artists)),

    # Streaming platforms
    "Apple_Music_Streams": np.random.randint(50000,500000,len(artists)),
    "Amazon_Music_Streams": np.random.randint(20000,400000,len(artists)),

    # Media signals
    "Articles_Blogs_Count": np.random.randint(5,60,len(artists)),
    "Sentiment_Score": np.random.uniform(0.5,0.9,len(artists)),

    # Industry signals
    "Concert_Sales": np.random.randint(500,20000,len(artists)),
    "Festival_Lineups": np.random.randint(0,10,len(artists)),
    "Awards_Buzz": np.random.randint(0,50,len(artists)),

    # Charts
    "Billboard_Buzz": np.random.randint(0,100,len(artists))
})

st.dataframe(social_data,width="stretch")

# -----------------------------------------------------
# Anomaly Detection – Spike Detection
# -----------------------------------------------------

st.subheader("⚡ Viral Spike Detection")

social_data["Streaming_Spike"] = (
    social_data["Apple_Music_Streams"] +
    social_data["Amazon_Music_Streams"]
) / 2

threshold = social_data["Streaming_Spike"].mean() + social_data["Streaming_Spike"].std()

social_data["Spike_Anomaly"] = social_data["Streaming_Spike"] > threshold

spike_artists = social_data[social_data["Spike_Anomaly"] == True]

if len(spike_artists) > 0:
    st.warning("Artists showing abnormal streaming spikes")
    st.dataframe(spike_artists[["Artist","Streaming_Spike"]],width="stretch")
else:
    st.info("No strong spike anomaly detected")

# -----------------------------------------------------
# Feature Engineering
# -----------------------------------------------------

social_data["Twitter_Index"] = (
    social_data["Twitter_Mentions"] *
    social_data["Twitter_Sentiment"]
)

social_data["Streaming_Index"] = (
    social_data["Apple_Music_Streams"] +
    social_data["Amazon_Music_Streams"]
)

social_data["Industry_Index"] = (
    social_data["Concert_Sales"] +
    social_data["Festival_Lineups"]*1000 +
    social_data["Awards_Buzz"]*100
)

social_data["Media_Index"] = (
    social_data["Articles_Blogs_Count"] *
    social_data["Sentiment_Score"]
)

social_data["Buzz_Index"] = social_data["Billboard_Buzz"]

# -----------------------------------------------------
# Weighted Social Momentum Score
# -----------------------------------------------------

st.subheader("📈 Social Momentum Score")

social_data["Social_Momentum"] = (
    0.30 * social_data["TikTok_Viral_Index"] +
    0.15 * social_data["Twitter_Index"] +
    0.10 * social_data["Media_Index"] +
    0.20 * social_data["Industry_Index"] +
    0.15 * social_data["Streaming_Index"] +
    0.10 * social_data["Buzz_Index"]
)

st.dataframe(
    social_data[["Artist","Social_Momentum"]],
    width="stretch"
)

# -----------------------------------------------------
# Visualization
# -----------------------------------------------------

fig_social = px.bar(
    social_data.sort_values("Social_Momentum",ascending=False),
    x="Artist",
    y="Social_Momentum",
    color="Social_Momentum",
    title="Artist Social Momentum Score"
)

st.plotly_chart(fig_social,width="stretch")

# -----------------------------------------------------
# Top Trending Artists
# -----------------------------------------------------

st.subheader("🔥 Top Socially Trending Artists")

top_social = social_data.sort_values(
    "Social_Momentum",
    ascending=False
).head(3)

st.dataframe(top_social,width="stretch")
# -----------------------------------------------------
# STEP 4 – Artist Knowledge Graph (Fixed + Edge Labels)
# -----------------------------------------------------

st.header("4️⃣ Similar Artist – Knowledge Graph")

import plotly.graph_objects as go

# Artists from previous steps
historical_artists = historical_data["Artist"].tolist()
predicted_artists = selected_artists["Artist"].tolist()

# Create graph
G = nx.Graph()

# Add nodes
for artist in historical_artists:
    G.add_node(artist, group="historical")

for artist in predicted_artists:
    G.add_node(artist, group="predicted")

# Define relationships
relations = [
    "similar style",
    "same genre",
    "collaboration potential",
    "fanbase overlap",
    "stream similarity"
]

# Connect predicted artists with historical artists
for artist in predicted_artists:
    connections = np.random.choice(historical_artists, size=2, replace=False)

    for c in connections:
        relation = np.random.choice(relations)
        G.add_edge(artist, c, relation=relation)

# Layout for graph (fixed positions)
pos = nx.spring_layout(G, seed=42)

# Create edge traces
edge_x = []
edge_y = []

edge_labels = []

for edge in G.edges(data=True):

    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]

    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

    label = edge[2]["relation"]
    edge_labels.append(((x0+x1)/2, (y0+y1)/2, label))

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=1,color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Node traces
node_x = []
node_y = []
node_text = []
node_color = []

for node in G.nodes(data=True):

    x,y = pos[node[0]]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node[0])

    if node[1]["group"] == "predicted":
        node_color.append("red")
    else:
        node_color.append("blue")

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    text=node_text,
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        color=node_color,
        size=20,
        line_width=2
    )
)

# Edge label annotations
annotations = []

for x,y,label in edge_labels:
    annotations.append(
        dict(
            x=x,
            y=y,
            text=label,
            showarrow=False,
            font=dict(size=10,color="gray")
        )
    )

fig = go.Figure(
    data=[edge_trace,node_trace],
    layout=go.Layout(
        title="Artist Relationship Knowledge Graph",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        width=900
    )
)

# Disable zoom & pan
fig.update_layout(
    dragmode=False
)

st.plotly_chart(fig, config={
    "scrollZoom": False,
    "displayModeBar": False
})
# -----------------------------------------------------
# STEP 5 – Competitive Analysis
# -----------------------------------------------------
# -----------------------------------------------------
# STEP 5 – Competitive Analysis (Benchmarking)
# -----------------------------------------------------

st.header("5️⃣ Competitive Analysis – Benchmarking")

artists = selected_artists["Artist"].tolist()

competitive_data = pd.DataFrame({

"Artist":artists,

# Key KPI signals
"Save_to_Stream_Ratio":np.random.uniform(0.2,0.6,len(artists)),
"Listener_Retention":np.random.uniform(20,50,len(artists)),
"Repeat_Listeners":np.random.uniform(20,60,len(artists)),
"Playlist_Addition_Velocity":np.random.uniform(10,80,len(artists)),
"Follower_Growth":np.random.uniform(5,25,len(artists))

})

# Benchmark values (industry averages)
industry_avg = {
"Save_to_Stream_Ratio":0.30,
"Listener_Retention":25,
"Repeat_Listeners":30,
"Playlist_Addition_Velocity":40,
"Follower_Growth":10
}

# Competitive Advantage Score
competitive_data["Competitive_Advantage"] = (

(competitive_data["Save_to_Stream_Ratio"]/industry_avg["Save_to_Stream_Ratio"])*20 +

(competitive_data["Listener_Retention"]/industry_avg["Listener_Retention"])*20 +

(competitive_data["Repeat_Listeners"]/industry_avg["Repeat_Listeners"])*20 +

(competitive_data["Playlist_Addition_Velocity"]/industry_avg["Playlist_Addition_Velocity"])*20 +

(competitive_data["Follower_Growth"]/industry_avg["Follower_Growth"])*20

)

competitive_data["Competitive_Advantage"] = competitive_data["Competitive_Advantage"].clip(0,100)

st.dataframe(competitive_data,width="stretch")

fig_comp = px.bar(
competitive_data,
x="Artist",
y="Competitive_Advantage",
title="Competitive Advantage Score",
color="Competitive_Advantage"
)

st.plotly_chart(fig_comp,width="stretch")

# -----------------------------------------------------
# STEP 6 – Ranking Engine
# -----------------------------------------------------
# -----------------------------------------------------
# STEP 6 – AI Ranking & Prediction
# -----------------------------------------------------

st.header("6️⃣ AI Ranking & Prediction – Trend Prediction Score")

ranking = competitive_data.copy()

# Viral Potential (from model output simulation)
ranking["Viral_Potential"] = np.random.uniform(50,100,len(ranking))

# Contextual Buzz (from social dynamics)
ranking["Contextual_Buzz"] = np.random.uniform(40,90,len(ranking))

# Competitive score from Step 5
ranking["Competitive_Score"] = ranking["Competitive_Advantage"]

# Adjustable weights
col1,col2,col3 = st.columns(3)

with col1:
    w1 = st.slider("Weight Viral Potential",0.1,1.0,0.4)

with col2:
    w2 = st.slider("Weight Contextual Buzz",0.1,1.0,0.3)

with col3:
    w3 = st.slider("Weight Competitive Advantage",0.1,1.0,0.3)

# Normalize weights
total = w1+w2+w3
w1,w2,w3 = w1/total,w2/total,w3/total

# TPS formula
ranking["TPS"] = (
w1 * ranking["Viral_Potential"] +
w2 * ranking["Contextual_Buzz"] +
w3 * ranking["Competitive_Score"]
)

ranking = ranking.sort_values("TPS",ascending=False)

st.dataframe(ranking,width="stretch")

fig_rank = px.bar(
ranking,
x="Artist",
y="TPS",
title="AI Trend Prediction Score (TPS)",
color="TPS"
)

st.plotly_chart(fig_rank,width="stretch")

# -----------------------------------------------------
# STEP 7 – Dashboard Result
# -----------------------------------------------------
# -----------------------------------------------------
# STEP 7 – Dashboard (Upcoming Artists)
# -----------------------------------------------------

st.header("7️⃣ Upcoming Artists – AI A&R Dashboard")

top_artists = ranking.head(4)

st.success("Top Rising Artists Identified")

st.dataframe(top_artists,width="stretch")

for i,row in top_artists.iterrows():

    confidence = round(row["TPS"],1)

    if confidence > 80:
        label = "🚀 Rising Star"
    elif confidence > 65:
        label = "⭐ Strong Potential"
    else:
        label = "⚠ Early Watchlist"

    st.markdown(f"""
### 🎤 {row['Artist']}

Prediction Confidence: **{confidence}/100**

Viral Potential Score: **{round(row['Viral_Potential'],1)}**

Contextual Buzz Score: **{round(row['Contextual_Buzz'],1)}**

Competitive Advantage: **{round(row['Competitive_Score'],1)}**

Status: **{label}**
""")
    
# -----------------------------------------------------
# A&R Decision Support
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
