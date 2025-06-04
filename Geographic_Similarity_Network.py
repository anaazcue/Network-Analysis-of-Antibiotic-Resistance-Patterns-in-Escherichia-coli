
"""
Geographic Similarity Network for Antibiotic Resistance Vectors

This script builds and visualizes a similarity network of resistance profiles in *Escherichia coli*,
annotated by geographic information. Nodes are connected based on log2-normalized MIC vectors,
and colored by geographical distance.

Author: Ana Azcue Unzueta
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
import pycountry
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from haversine import haversine, Unit

# -----------------------------------------------------------------------------
# 1. Load and preprocess dataset
# -----------------------------------------------------------------------------
data = pd.read_csv("2024_05_28 atlas_antibiotics.csv")

# Select relevant rows and columns
data = data[(data["Species"] == "Escherichia coli") & (data["Study"] != "SPIDAAR")]
data = data[data["Year"].isin([2018, 2019, 2020, 2021, 2022])]

selected_antibiotics = [
    "Cefoperazone sulbactam", "Trimethoprim sulfa", "Ciprofloxacin", "Ampicillin sulbactam",
    "Gentamicin", "Colistin", "Aztreonam avibactam", "Aztreonam", "Ceftazidime avibactam", "Ceftaroline",
    "Imipenem", "Ceftazidime", "Meropenem", "Ampicillin", "Cefepime", "Piperacillin tazobactam",
    "Levofloxacin", "Tigecycline", "Amoxycillin clavulanate", "Amikacin"
]

data = data[["Year", "Country"] + selected_antibiotics].dropna()

# -----------------------------------------------------------------------------
# 2. Clean and normalize (log2) MIC values
# -----------------------------------------------------------------------------
def normalize_mic(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = df[col].astype(str).str.extract(r"([0-9.]+)").astype(float)
    df[cols] = np.log2(df[cols]).round(2)
    return df

data = normalize_mic(data, selected_antibiotics)

# -----------------------------------------------------------------------------
# 3. Build frequency and country-annotated MIC vector list
# -----------------------------------------------------------------------------
def build_vector_table(df, cols, year_col="Year"):
    grouped = df.groupby(cols).agg(
        frequency=(year_col, "count"),
        years=(year_col, lambda x: sorted(x.unique())),
        country=("Country", lambda x: Counter(x).most_common(1)[0][0])
    ).reset_index().sort_values("frequency", ascending=False).reset_index(drop=True)
    return grouped.drop(columns="frequency"), grouped

vectors_unique, vectors_freq = build_vector_table(data, selected_antibiotics)

# -----------------------------------------------------------------------------
# 4. Build similarity network with tolerance-based edges
# -----------------------------------------------------------------------------
def build_similarity_graph(df, cols, tolerance=1.0):
    G = nx.Graph()
    for idx, row in df.iterrows():
        G.add_node(idx, frequency=row.get("frequency"), country=row.get("country"),
                   **{ab: row[ab] for ab in cols})

    for i, col in enumerate(cols):
        grouped = df.groupby(cols[:i] + cols[i+1:], dropna=False)
        for _, subdf in grouped:
            if len(subdf) < 2:
                continue
            subdf = subdf.sort_values(col)
            rows = list(subdf[[col]].itertuples(index=True, name=None))
            start = 0
            for end in range(len(rows)):
                while rows[end][1] - rows[start][1] > tolerance:
                    start += 1
                for mid in range(start, end):
                    G.add_edge(rows[end][0], rows[mid][0])
    return G

G_sim = build_similarity_graph(vectors_freq, selected_antibiotics, tolerance=1)

# -----------------------------------------------------------------------------
# 5. Assign country ISO3 and centroids
# -----------------------------------------------------------------------------
corrections = {"Korea, South": "South Korea", "Russia": "Russian Federation", "Turkey": "Türkiye"}

def to_iso3(name):
    name = corrections.get(name, name)
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

for _, d in G_sim.nodes(data=True):
    d["iso_a3"] = to_iso3(d.get("country"))

centroids = pd.read_csv("world_centroids_iso3.csv").set_index("iso_a3")

for n, d in G_sim.nodes(data=True):
    if d["iso_a3"] in centroids.index:
        d["lat"] = centroids.at[d["iso_a3"], "lat"]
        d["lon"] = centroids.at[d["iso_a3"], "lon"]

# -----------------------------------------------------------------------------
# 6. Compute edge geographic distances
# -----------------------------------------------------------------------------
def edge_distance(u, v):
    a, b = G_sim.nodes[u], G_sim.nodes[v]
    return haversine((a["lat"], a["lon"]), (b["lat"], b["lon"]), unit=Unit.KILOMETERS)

edge_data = [
    {"u": u, "v": v, "dist_km": edge_distance(u, v)}
    for u, v in G_sim.edges()
    if "lat" in G_sim.nodes[u] and "lat" in G_sim.nodes[v]
]

edges_df = pd.DataFrame(edge_data)

# -----------------------------------------------------------------------------
# 7. Plot geographic network with edges by distance
# -----------------------------------------------------------------------------
distance_bins = [
    (0, 1000, "red"), (1000, 2500, "orange"), (2500, 5000, "green"), (5000, float("inf"), "blue")
]
num_levels_to_draw = 4  # Change 1-4

node_geo = pd.DataFrame([
    {"node": n, "country": d["country"], "iso_a3": d["iso_a3"], "lat": d["lat"], "lon": d["lon"]}
    for n, d in G_sim.nodes(data=True) if "lat" in d and "lon" in d
])

per_country = (
    node_geo.groupby(["iso_a3", "country", "lat", "lon"])
             .agg(num_nodes=("node", "count"))
             .reset_index()
)

fig = px.scatter_geo(
    per_country, lat="lat", lon="lon", size="num_nodes", color="num_nodes",
    color_continuous_scale="Plasma", hover_name="country",
    projection="natural earth", size_max=12, opacity=0.8,
    labels={"num_nodes": "Number of nodes"}
)

total_links = 0
for i, (dmin, dmax, color) in enumerate(distance_bins[:num_levels_to_draw]):
    sub = edges_df[(edges_df["dist_km"] > dmin) & (edges_df["dist_km"] <= dmax)]
    total_links += len(sub)
    for _, row in sub.iterrows():
        a, b = G_sim.nodes[row["u"]], G_sim.nodes[row["v"]]
        fig.add_trace(go.Scattergeo(
            lon=[a["lon"], b["lon"]], lat=[a["lat"], b["lat"]],
            mode="lines", showlegend=False,
            line=dict(width=0.1, color=color, dash="dot" if color == "blue" else None),
            hoverinfo="text", text=f"{row['dist_km']:.0f} km"
        ))

fig.update_layout(
    height=650,
    margin=dict(l=0, r=0, t=40, b=0),
    coloraxis_showscale=False,
    coloraxis_colorbar_title="Number of nodes"
)

pio.renderers.default = "browser"
fig.show()
print(f"Total links drawn (up to level {num_levels_to_draw}): {total_links}")
fig.write_image("map_connections.png", scale=3)

# -----------------------------------------------------------------------------
# 8. Plot histogram of edge distances (log-log)
# -----------------------------------------------------------------------------
valid_dists = [d["dist_km"] for d in edge_data if d["dist_km"] > 0]
plt.figure(figsize=(8, 5))
plt.hist(valid_dists, bins=np.logspace(np.log10(min(valid_dists)), np.log10(max(valid_dists)), 30),
         color='steelblue', edgecolor='black', alpha=0.75)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Geographic distance between connected nodes (km) [log scale]")
plt.ylabel("Frequency [log scale]")
plt.title("Distribution of geographic distances (log-log)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

