import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

st.title("🎵 Rising Artist Prediction Platform")
st.markdown("AI-driven **A&R talent discovery system**")

# ----------------------------------------
# Dummy Artist Dataset
# ----------------------------------------

artists = [
    "Nova Echo",
    "Luna Drift",
    "Atlas Fire",
    "Solar Pulse",
    "Midnight Arcade",
    "Neon Static"
]

genres = ["Synthwave","Indie Pop","Trap","Electronic","Retro Wave","Synth Pop"]

data = {
    "Artist": artists,
    "Genre": genres,
    "Spotify Streams": np.random.randint(500000,5000000,6),
    "TikTok UGC": np.random.randint(200,5000,6),
    "Instagram Followers": np.random.randint(10000,200000,6),
    "YouTube Views": np.random.randint(100000,2000000,6),
    "Stream Growth": np.random.randint(20,250,6),
    "Follower Growth": np.random.randint(10,80,6),
    "Save Rate": np.random.uniform(3,15,6),
    "Repeat Listener": np.random.uniform(10,50,6),
    "Skip Rate": np.random.uniform(5,30,6)
}

df = pd.DataFrame(data)

# ----------------------------------------
# Sidebar Filters
# ----------------------------------------

st.sidebar.header("Prediction Filters")

horizon = st.sidebar.selectbox(
    "Prediction Horizon",
    ["30 Days","60 Days","90 Days","180 Days","365 Days"]
)

genre_filter = st.sidebar.multiselect(
    "Genre",
    df["Genre"].unique(),
    default=df["Genre"].unique()
)

filtered_df = df[df["Genre"].isin(genre_filter)]

# ----------------------------------------
# Compute Scores
# ----------------------------------------

filtered_df["Viral Score"] = (
    0.5*filtered_df["Stream Growth"] +
    0.5*(filtered_df["TikTok UGC"]/50)
)

filtered_df["Buzz Score"] = (
    0.6*(filtered_df["Instagram Followers"]/10000) +
    0.4*(filtered_df["YouTube Views"]/100000)
)

filtered_df["Competitive Score"] = (
    0.4*filtered_df["Save Rate"] +
    0.4*filtered_df["Repeat Listener"] +
    0.2*(100-filtered_df["Skip Rate"])
)

w1,w2,w3 = 0.4,0.3,0.3

filtered_df["TPS"] = (
    w1*filtered_df["Viral Score"] +
    w2*filtered_df["Buzz Score"] +
    w3*filtered_df["Competitive Score"]
)

filtered_df = filtered_df.sort_values("TPS",ascending=False)
filtered_df["Rank"] = range(1,len(filtered_df)+1)

# ----------------------------------------
# Leaderboard
# ----------------------------------------

st.header("🏆 Rising Artist Leaderboard")

st.dataframe(
    filtered_df[[
        "Rank",
        "Artist",
        "Genre",
        "TPS",
        "Viral Score",
        "Buzz Score",
        "Competitive Score"
    ]],
    use_container_width=True
)

fig = px.bar(
    filtered_df,
    x="Artist",
    y="TPS",
    color="Genre",
    text="TPS",
    title="Top Rising Artists"
)

st.plotly_chart(fig,use_container_width=True)

# ----------------------------------------
# Artist Selection
# ----------------------------------------

st.header("🔍 Artist Deep Dive")

artist = st.selectbox(
    "Select Artist",
    filtered_df["Artist"]
)

artist_data = filtered_df[filtered_df["Artist"]==artist]

# ----------------------------------------
# Platform Metrics
# ----------------------------------------

st.subheader("Platform Metrics")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Spotify Streams",int(artist_data["Spotify Streams"]))
col2.metric("TikTok UGC Videos",int(artist_data["TikTok UGC"]))
col3.metric("Instagram Followers",int(artist_data["Instagram Followers"]))
col4.metric("YouTube Views",int(artist_data["YouTube Views"]))

# ----------------------------------------
# KPI Radar Chart
# ----------------------------------------

st.subheader("KPI Performance")

categories = [
    "Stream Growth",
    "Follower Growth",
    "Save Rate",
    "Repeat Listener",
    "Low Skip Rate"
]

values = [
    float(artist_data["Stream Growth"]),
    float(artist_data["Follower Growth"]),
    float(artist_data["Save Rate"]),
    float(artist_data["Repeat Listener"]),
    100-float(artist_data["Skip Rate"])
]

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name=artist
))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=False
)

st.plotly_chart(fig_radar,use_container_width=True)

# ----------------------------------------
# Similar Artist Graph
# ----------------------------------------

st.subheader("🎧 Similar Artist Graph")

G = nx.Graph()

for a in artists:
    G.add_node(a)

edges = [
    ("Nova Echo","Midnight Arcade"),
    ("Nova Echo","Neon Static"),
    ("Luna Drift","Solar Pulse"),
    ("Atlas Fire","Solar Pulse"),
    ("Midnight Arcade","Neon Static")
]

for e in edges:
    G.add_edge(e[0],e[1],weight=np.random.uniform(0.6,0.9))

net = Network(height="400px",width="100%")

for node in G.nodes():
    net.add_node(node,label=node)

for edge in G.edges(data=True):
    net.add_edge(
        edge[0],
        edge[1],
        value=edge[2]["weight"]
    )

net.save_graph("graph.html")

HtmlFile = open("graph.html","r",encoding="utf-8")
components.html(HtmlFile.read(),height=450)

# ----------------------------------------
# Competitive Benchmark
# ----------------------------------------

st.subheader("📊 Competitive Benchmark")

peer_avg = filtered_df.mean(numeric_only=True)

benchmark = pd.DataFrame({
    "Metric":[
        "Stream Growth",
        "Save Rate",
        "Repeat Listener"
    ],
    "Artist":[
        float(artist_data["Stream Growth"]),
        float(artist_data["Save Rate"]),
        float(artist_data["Repeat Listener"])
    ],
    "Genre Avg":[
        peer_avg["Stream Growth"],
        peer_avg["Save Rate"],
        peer_avg["Repeat Listener"]
    ]
})

fig_benchmark = px.bar(
    benchmark,
    x="Metric",
    y=["Artist","Genre Avg"],
    barmode="group"
)

st.plotly_chart(fig_benchmark,use_container_width=True)

# ----------------------------------------
# Prediction Engine
# ----------------------------------------

st.header("🤖 AI Prediction Engine")

viral = float(artist_data["Viral Score"])
buzz = float(artist_data["Buzz Score"])
comp = float(artist_data["Competitive Score"])

tps = float(artist_data["TPS"])

col1,col2,col3 = st.columns(3)

col1.metric("Viral Score",round(viral,2))
col2.metric("Buzz Score",round(buzz,2))
col3.metric("Competitive Score",round(comp,2))

st.write("### TPS Calculation")

st.code(f"""
TPS = w1 * Viral Score
    + w2 * Buzz Score
    + w3 * Competitive Score

w1 = 0.4
w2 = 0.3
w3 = 0.3

TPS = {round(tps,2)}
""")

# ----------------------------------------
# Prediction Gauge
# ----------------------------------------

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=tps,
    title={'text':"Trend Prediction Score"},
    gauge={
        'axis':{'range':[0,100]},
        'steps':[
            {'range':[0,50],'color':"lightgray"},
            {'range':[50,75],'color':"orange"},
            {'range':[75,100],'color':"green"}
        ]
    }
))

st.plotly_chart(fig_gauge,use_container_width=True)

# ----------------------------------------
# A&R Decision Panel
# ----------------------------------------

st.header("🎯 A&R Recommendation")

if tps > 75:
    decision = "🌟 SIGN ARTIST"
elif tps > 60:
    decision = "👀 MONITOR CLOSELY"
else:
    decision = "🎧 EARLY STAGE"

st.success(f"Recommendation: **{decision}**")

st.markdown("""
### Decision Reasoning

• High viral growth signals  
• Strong fan retention  
• Competitive advantage vs peer artists  
• Multi-platform traction
""")

# ----------------------------------------
# Footer
# ----------------------------------------

st.markdown("---")
st.caption("POC: AI Rising Artist Prediction Platform")