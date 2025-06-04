"""
Comprehensive Analysis of Antibiotic Resistance Vectors in Escherichia coli (2018–2022)

This script performs a full pipeline analysis of resistance profiles in E. coli based on MIC values 
(log2-transformed), including:

1. Data cleaning and normalization of antibiotic resistance values.
2. Extraction of unique resistance vectors and their frequencies across years.
3. Construction of similarity graphs under different tolerance rules (Rule 1, 2, 3).
4. Building a temporal mutation graph connecting vectors across consecutive years.
5. Graph-theoretic analysis of structure (degree, clustering, assortativity, component sizes, etc.).
6. Threshold sweep to explore how network properties change with increasing MIC tolerance.

All results are exported and visualized, including plots for:
- Degree distribution (log-scale)
- Clustering vs tolerance
- Assortativity vs tolerance
- Edge growth and density vs tolerance
- Component size CDF

Author: Ana Azcue Unzueta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

###############################################################################
# Data loading, filtering, and preprocessing
###############################################################################

# Load dataset
data = pd.read_csv('2024_05_28 atlas_antibiotics.csv')

# Count number of samples per species
species_counts = data["Species"].value_counts().reset_index()
species_counts.columns = ["Species", "Count"]

# List of all possible antibiotics in the dataset
antibiotics = [
    "Amikacin", "Amoxycillin clavulanate", "Ampicillin", "Azithromycin", "Cefepime",
    "Cefoxitin", "Ceftazidime", "Ceftriaxone", "Clarithromycin", "Clindamycin",
    "Erythromycin", "Imipenem", "Levofloxacin", "Linezolid", "Meropenem",
    "Metronidazole", "Minocycline", "Penicillin", "Piperacillin tazobactam", "Tigecycline",
    "Vancomycin", "Ampicillin sulbactam", "Aztreonam", "Aztreonam avibactam", "Cefixime",
    "Ceftaroline", "Ceftaroline avibactam", "Ceftazidime avibactam", "Ciprofloxacin", "Colistin",
    "Daptomycin", "Doripenem", "Ertapenem", "Gatifloxacin", "Gentamicin",
    "Moxifloxacin", "Oxacillin", "Quinupristin dalfopristin", "Sulbactam", "Teicoplanin",
    "Tetracycline", "Trimethoprim sulfa", "Ceftolozane tazobactam", "Cefoperazone sulbactam",
    "Meropenem vaborbactam", "Cefpodoxime", "Ceftibuten", "Ceftibuten avibactam", "Tebipenem"
]

# Selected antibiotics for analysis
selected_antibiotics = [
    "Cefoperazone sulbactam", "Trimethoprim sulfa", "Ciprofloxacin", "Ampicillin sulbactam", 
    "Gentamicin", "Colistin", "Aztreonam avibactam", "Aztreonam", "Ceftazidime avibactam", "Ceftaroline",
    "Imipenem", "Ceftazidime", "Meropenem", "Ampicillin", "Cefepime", "Piperacillin tazobactam",
    "Levofloxacin", "Tigecycline", "Amoxycillin clavulanate", "Amikacin"
]

# Filter dataset for E. coli and selected years, drop missing values
filtered_data = data[data['Species'] == 'Escherichia coli']
filtered_data = filtered_data[filtered_data['Study'] != 'SPIDAAR']
filtered_data = filtered_data[filtered_data['Year'].isin([2018, 2019, 2020, 2021, 2022])]
filtered_data = filtered_data[["Year"] + selected_antibiotics]
filtered_data = filtered_data.dropna()


def clean_and_normalize_dataset(df, cols_to_normalize):
    """
    Extracts numeric values from resistance entries and applies log2 transformation.

    Parameters:
    - df: DataFrame containing the 'Year' column and antibiotic resistance data
    - cols_to_normalize: list of antibiotic columns to clean and normalize

    Returns:
    - df: cleaned and normalized DataFrame
    """
    df = df.copy()

    # Extract numeric part and convert to float
    for col in cols_to_normalize:
        df[col] = df[col].astype(str).str.extract(r"([0-9.]+)").astype(float)

    # Apply log2 transformation and round to 2 decimals
    df[cols_to_normalize] = np.log2(df[cols_to_normalize]).round(2)

    return df


filtered_data = clean_and_normalize_dataset(filtered_data, cols_to_normalize=selected_antibiotics)

###############################################################################
# Unique Resistance Vectors Analysis
###############################################################################

def count_unique_vectors_with_years(df, antibiotic_columns, year_column="Year"):
    """
    Counts the frequency of unique resistance vectors and records the years they appear.

    Parameters:
    - df: DataFrame with antibiotic resistance data
    - antibiotic_columns: list of columns representing antibiotic values
    - year_column: column name containing the year

    Returns:
    - unique_vectors: DataFrame with unique vectors and associated years
    - vector_frequencies: DataFrame with frequency and years for each vector
    """
    # Group by resistance vector
    grouped = df.groupby(antibiotic_columns).agg(
        frequency=(year_column, 'count'),
        years=(year_column, lambda x: sorted(x.unique()))
    ).reset_index()

    # Sort by frequency (descending)
    grouped = grouped.sort_values(by="frequency", ascending=False).reset_index(drop=True)

    vector_frequencies = grouped.copy()
    unique_vectors = grouped.drop(columns="frequency")

    return unique_vectors, vector_frequencies


# Apply function to filtered dataset
unique_vectors, vector_frequencies = count_unique_vectors_with_years(
    df=filtered_data,
    antibiotic_columns=selected_antibiotics,
    year_column="Year"
)


def plot_vector_frequency_histogram(df_frequencies):
    """
    Plots a histogram showing the distribution of unique resistance vector frequencies,
    with a logarithmic scale on the Y-axis.

    Parameters:
    - df_frequencies: DataFrame containing a 'frequency' column
    """
    frequency_counts = df_frequencies['frequency'].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(frequency_counts.index, frequency_counts.values, color="#4682b4")

    plt.xlabel("Frequency of Unique Resistance Vector")
    plt.ylabel("Number of Vectors (log scale)")
    plt.title("Histogram of Unique Resistance Vector Frequencies")
    plt.yscale('log')
    plt.grid(axis='y', linestyle='--', alpha=0.7, which='both')

    # Optional: Annotate each bar
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


# Plot the histogram
plot_vector_frequency_histogram(vector_frequencies)


###############################################################################
# Construct Similarity Graph (Rules 1,2 and 3)
###############################################################################

def build_similarity_graph(df, antibiotic_columns,
                                 filename="multi_antibiotic_graph.gexf",
                                 tolerance=1.0, max_diffs=2):
    """
    Builds an undirected graph where nodes are resistance vectors and edges connect
    those differing in at most `max_diffs` antibiotics, each within a given `tolerance`.

    Parameters:
    - df: DataFrame with resistance vectors and optional 'frequency'
    - antibiotic_columns: list of antibiotic columns to compare
    - filename: path to save the output GEXF graph (optional)
    - tolerance: max difference allowed per antibiotic (Chebyshev distance)
    - max_diffs: max number of antibiotics that can differ between two vectors

    Returns:
    - G: networkx.Graph object
    """
    # Convert antibiotic values to float matrix
    X = df[antibiotic_columns].values.astype(float)

    # Use Chebyshev metric to find nearby vectors
    nn = NearestNeighbors(radius=tolerance, metric='chebyshev', n_jobs=-1)
    nn.fit(X)
    neighbor_indices = nn.radius_neighbors(return_distance=False)

    G = nx.Graph()
    for idx, row in df.iterrows():
        G.add_node(idx,
                   frequency=row.get("frequency", None),
                   **{ab: row[ab] for ab in antibiotic_columns})

    for i, neighbors in tqdm(enumerate(neighbor_indices), total=len(neighbor_indices),
                              desc=f"Creating edges (max_diffs ≤ {max_diffs})"):
        for j in neighbors:
            if j <= i:
                continue
            diff = np.abs(X[i] - X[j])
            num_diff = np.count_nonzero(diff)
            if num_diff <= max_diffs:
                G.add_edge(i, j, diffs=num_diff)

    if filename:
        nx.write_gexf(G, filename)
        print(f"✅ Exported «{filename}» with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")

    return G


# Example usage
similarity_graph = build_similarity_graph(
    df=vector_frequencies,
    antibiotic_columns=selected_antibiotics,
    filename="similarity_graph.gexf",
    tolerance=1.0, # 1 for Rule 1, more for Rule 2
    max_diffs=1  # 1 for Rule 1, more for Rule 3
)

###############################################################################
# Temporal Mutation Graph Construction
###############################################################################

def are_vectors_connected(v1, v2, tolerance=1.0):
    """
    Checks whether two resistance vectors differ in exactly one antibiotic,
    and that the difference is within the given tolerance.

    Parameters:
    - v1, v2: numpy arrays representing resistance vectors
    - tolerance: maximum allowed difference in the differing antibiotic

    Returns:
    - True if vectors are connectable under mutation rule, False otherwise
    """
    diffs = np.abs(v1 - v2)
    return (diffs != 0).sum() == 1 and diffs.max() <= tolerance


def build_temporal_mutation_graph(df, antibiotic_columns,
                                   filename="temporal_mutation_graph.gexf",
                                   tolerance=1.0):
    """
    Builds a directed temporal mutation graph from resistance vector data.
    
    Each node represents a (unique resistance vector, year).
    An edge from year Y to Y+1 is created if two vectors differ in exactly one antibiotic
    and the difference is within the specified tolerance.

    Parameters:
    - df: DataFrame with columns 'years', 'frequency', and antibiotic values
    - antibiotic_columns: list of antibiotic columns to use as vector components
    - filename: name of the output GEXF file (optional)
    - tolerance: maximum allowed change in a single antibiotic to consider a mutation

    Returns:
    - G: networkx.DiGraph object
    """
    # Step 1: Explode 'years' list into separate rows (each vector-year pair is one row)
    df_exp = df.explode('years').reset_index(drop=True)
    df_exp.rename(columns={'years': 'year'}, inplace=True)

    # Step 2: Create directed graph with one node per row in df_exp
    G = nx.DiGraph()
    node_data = {}

    print("🔄 Adding nodes to graph...")
    for i, row in tqdm(df_exp.iterrows(), total=len(df_exp), desc="Adding Nodes"):
        node_id = f"n{i}"
        vector = row[antibiotic_columns].values.astype(float)
        G.add_node(node_id,
                   year=row["year"],
                   frequency=row.get("frequency", None),
                   **{ab: row[ab] for ab in antibiotic_columns})
        node_data[i] = (row["year"], vector)

    # Step 3: Create temporal edges between vectors from year Y to Y+1
    years_sorted = sorted(df_exp["year"].unique())
    print("🔄 Linking temporal mutations (Y → Y+1)...")

    for y in tqdm(years_sorted, desc="Processing Years"):
        next_y = y + 1
        df_y = df_exp[df_exp["year"] == y]
        df_yplus = df_exp[df_exp["year"] == next_y]

        if df_y.empty or df_yplus.empty:
            continue

        # Step 4: For each antibiotic, group by all others to find 1-diff connections
        for ab in antibiotic_columns:
            others = [c for c in antibiotic_columns if c != ab]
            groups_y = df_y.groupby(others, dropna=False)
            groups_yplus = df_yplus.groupby(others, dropna=False)

            for group_key, group_df_y in groups_y:
                if group_key not in groups_yplus.groups:
                    continue

                group_df_yplus = groups_yplus.get_group(group_key)

                # Sort both groups by the antibiotic column being relaxed
                group_df_y = group_df_y.sort_values(by=ab)
                group_df_yplus = group_df_yplus.sort_values(by=ab)

                # Convert to list of (df_index, antibiotic_value)
                rows_y = list(group_df_y[[ab]].itertuples(index=True, name=None))
                rows_yplus = list(group_df_yplus[[ab]].itertuples(index=True, name=None))

                # Step 5: Use two-pointer technique to find close-enough values
                i, j = 0, 0
                while i < len(rows_y) and j < len(rows_yplus):
                    idx_y, val_y = rows_y[i]
                    idx_yplus, val_yplus = rows_yplus[j]
                    diff = val_yplus - val_y

                    if diff > tolerance:
                        i += 1
                    elif diff < -tolerance:
                        j += 1
                    else:
                        node_from = f"n{idx_y}"
                        node_to = f"n{idx_yplus}"
                        G.add_edge(node_from, node_to)
                        i += 1
                        j += 1

    if filename:
        nx.write_gexf(G, filename)
        print(f"✅ Exported temporal mutation graph to «{filename}» with "
              f"{G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")

    return G


# Example usage
temporal_graph = build_temporal_mutation_graph(
    df=vector_frequencies,
    antibiotic_columns=selected_antibiotics,
    filename="temporal_mutation_graph.gexf",
    tolerance=1.0
)


###############################################################################
# Graph Structure Analysis
###############################################################################

def analyze_graph_structure(G, show_plots=True, title_suffix=""):
    """
    Analyzes a graph's structural properties and optionally displays:
    - Degree distribution histogram (log scale)
    - Cumulative degree distribution
    - CDF of connected component sizes

    Parameters:
    - G: networkx.Graph or DiGraph object
    - show_plots: whether to display plots (default: True)
    - title_suffix: string to append to plot titles

    Returns:
    - results: dictionary containing key statistics and distributions
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    results = {
        "num_nodes": num_nodes,
        "num_edges": num_edges
    }

    # Degree statistics
    degrees = [deg for _, deg in G.degree()]
    degree_dist = Counter(degrees)
    results["degree_distribution"] = degree_dist
    results["average_degree"] = sum(degrees) / num_nodes if num_nodes > 0 else 0

    # Connected components
    if isinstance(G, nx.DiGraph):
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    component_sizes = [len(c) for c in components]
    results["num_components"] = len(components)
    results["component_sizes"] = sorted(component_sizes, reverse=True)

    if show_plots:
        ### Degree Distribution and Cumulative Plot ###
        x_deg = sorted(degree_dist.keys())
        y_deg = [degree_dist[k] for k in x_deg]
        total_nodes = sum(y_deg)
        y_cdf = [sum(y_deg[:i + 1]) / total_nodes for i in range(len(y_deg))]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        plt.subplots_adjust(wspace=0.3)

        # Histogram of degree distribution
        bars = axes[0].bar(x_deg, y_deg, color='cornflowerblue', edgecolor='black')
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Degree")
        axes[0].set_ylabel("Node Count (log scale)")
        axes[0].set_title(f"Degree Distribution {title_suffix}")
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)

        # Annotate each bar (optional)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{int(height)}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        # Cumulative degree distribution
        axes[1].plot(x_deg, y_cdf, marker='o', linestyle='-', color='darkorange')
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Degree $k$")
        axes[1].set_ylabel(r"$P(k' \geq k)$ (log scale)")
        axes[1].set_title(f"Cumulative Degree Distribution {title_suffix}")
        axes[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        ### Component Size CDF ###
        sorted_sizes = sorted(component_sizes)
        cdf = np.arange(len(sorted_sizes)) / float(len(sorted_sizes))

        plt.figure(figsize=(6.5, 5))
        plt.plot(sorted_sizes, cdf, marker='o', linestyle='none', color='darkgreen')
        plt.xscale("log")
        plt.xlabel("Component Size (nodes)")
        plt.ylabel("Cumulative Fraction (CDF)")
        plt.title(f"Component Size Distribution {title_suffix}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return results


# Example usage
results_similarity = analyze_graph_structure(
    similarity_graph,
    show_plots=True,
    title_suffix="(Similarity Network)"
)


def analyze_graph(
    G,
    name="Network",
    tolerance=None,
    title_suffix="",
    print_metrics=True,
    show_plots=False
):
    """
    Analyzes a given graph and returns a dictionary of key structural metrics.

    Parameters:
    - G: networkx.Graph or DiGraph
    - name: label for print/logging context
    - tolerance: optional value to include in the result dictionary
    - title_suffix: string suffix for plots (if shown)
    - print_metrics: whether to print metrics to console
    - show_plots: whether to show structural plots

    Returns:
    - metrics: dictionary containing the graph's structural metrics
    """
    if print_metrics:
        print(f"\n📊 Analyzing graph: {name}")

    # Basic structure via degree/component distributions
    metrics = analyze_graph_structure(G, show_plots=show_plots, title_suffix=title_suffix)

    if tolerance is not None:
        metrics["tolerance"] = tolerance

    # Assortativity and clustering
    metrics["degree_assortativity"] = nx.degree_assortativity_coefficient(G)
    metrics["average_clustering"] = nx.average_clustering(G)
    metrics["global_clustering"] = nx.transitivity(G)
    metrics["density"] = nx.density(G)

    if print_metrics:
        print(f"Degree assortativity coefficient: {metrics['degree_assortativity']:.3f}")
        print(f"Average local clustering coefficient (C_L): {metrics['average_clustering']:.4f}")
        print(f"Global clustering coefficient (C): {metrics['global_clustering']:.4f}")
        print(f"Network density: {metrics['density']:.8f}")

    # Bridges
    bridges = list(nx.bridges(G))
    metrics["num_bridges"] = len(bridges)
    if print_metrics:
        print(f"Number of bridges: {metrics['num_bridges']}")

    # Degree (excluding isolates)
    degrees = [deg for _, deg in G.degree()]
    non_isolated = [d for d in degrees if d > 0]
    metrics["avg_degree_non_isolated"] = (
        sum(non_isolated) / len(non_isolated) if non_isolated else 0
    )

    # Giant component metrics
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        components = (
            list(nx.weakly_connected_components(G))
            if isinstance(G, nx.DiGraph)
            else list(nx.connected_components(G))
        )
        giant = max(components, key=len)
        G_giant = G.subgraph(giant).copy()
        n = G_giant.number_of_nodes()
        metrics["giant_component_size"] = n

        if n > 1:
            metrics["radius"] = nx.radius(G_giant)
            metrics["diameter"] = nx.diameter(G_giant)
            metrics["avg_shortest_path_length"] = nx.average_shortest_path_length(G_giant)
        else:
            metrics.update(dict(radius=0, diameter=0, avg_shortest_path_length=0))

        if print_metrics:
            print(f"Size of largest component: {n}")
            print(f"Radius: {metrics['radius']}, Diameter: {metrics['diameter']}")
            print(f"Average shortest path length: {metrics['avg_shortest_path_length']:.3f}")
    else:
        metrics.update(dict(giant_component_size=0, radius=0, diameter=0, avg_shortest_path_length=0))

    return metrics


# Example usage
metrics_sim = analyze_graph(similarity_graph, name="Similarity Net", tolerance=1)
metrics_temp = analyze_graph(temporal_graph, name="Temporal Net", print_metrics=False, show_plots=True)


###############################################################################
# Threshold Sweep: Build and Analyze Similarity Graphs with Varying Tolerance
###############################################################################

# Define thresholds to evaluate
thresholds = list(range(1, 3))  # Adjust range as needed
all_results = []

# Step 1: Build and analyze network for each threshold
for tol in thresholds:
    print(f"\n>>> Analyzing similarity network with tolerance = {tol}")
    G = build_similarity_graph(
        df=vector_frequencies,
        antibiotic_columns=selected_antibiotics,
        filename=f"similarity_graph_tol{tol}.gexf",
        tolerance=tol
    )
    metrics = analyze_graph(
        G,
        name=f"Similarity Network (tolerance = {tol})",
        tolerance=tol,
        print_metrics=False,
        show_plots=False
    )
    metrics["num_edge"] = G.number_of_edges()
    all_results.append(metrics)

# Step 2: Save results
df_metrics = pd.DataFrame(all_results)
df_metrics.to_pickle("results/df_metrics.pkl")

# Step 3: Plot edge count vs. threshold
thresholds = df_metrics["tolerance"].tolist()
edges = df_metrics["num_edge"].tolist()

plt.figure(figsize=(8, 5))
plt.plot(thresholds, edges, marker='o', color='mediumblue', linewidth=2)
for x, y in zip(thresholds, edges):
    plt.text(x, y + 30, str(y), ha='center', va='bottom', fontsize=9)

plt.xlabel("Tolerance")
plt.ylabel("Number of Edges (|E|)")
plt.title("Edge Growth with Increasing Tolerance")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Step 4: Plot average degree
avg_degree_all = df_metrics["average_degree"].tolist()
avg_degree_non_isolated = df_metrics["avg_degree_non_isolated"].tolist()

plt.figure(figsize=(8, 5))
plt.plot(thresholds, avg_degree_all, marker='o', label="Including isolated nodes", color='cornflowerblue')
plt.plot(thresholds, avg_degree_non_isolated, marker='s', label="Excluding isolated nodes", color='darkorange')
for x, y in zip(thresholds, avg_degree_all):
    plt.text(x, y + 0.01, f"{y:.3f}", ha='center', fontsize=8)
for x, y in zip(thresholds, avg_degree_non_isolated):
    plt.text(x, y + 0.03, f"{y:.3f}", ha='center', fontsize=8)

plt.xlabel("Tolerance")
plt.ylabel("Average Degree")
plt.title("Average Degree vs Tolerance")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Step 5: Degree distribution (thresholds 1 and 2)
def extract_degree_distribution(df, tol):
    dist = df[df["tolerance"] == tol]["degree_distribution"].values[0]
    return {int(k): v for k, v in dist.items()}

def plot_degree_distribution(ax, dist, title, color):
    x = sorted(dist.keys())
    y = [dist[k] for k in x]
    bars = ax.bar(x, y, color=color, edgecolor='black')
    ax.set_yscale("log")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Node Count (log scale)")
    ax.set_title(title)
    ax.set_xticks(x)
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height,
                    f"{int(height)}", ha='center', va='bottom', fontsize=9)

deg1 = extract_degree_distribution(df_metrics, 1)
deg2 = extract_degree_distribution(df_metrics, 2) if 2 in thresholds else {}

if deg2:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_degree_distribution(axes[0], deg1, "Degree Distribution (t=1)", "cornflowerblue")
    plot_degree_distribution(axes[1], deg2, "Degree Distribution (t=2)", "darkorange")
else:
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_degree_distribution(ax, deg1, "Degree Distribution (t=1)", "cornflowerblue")

plt.tight_layout()
plt.show()

# Step 6: Assortativity vs Tolerance
assortativity = df_metrics["degree_assortativity"]
plt.figure(figsize=(8, 5))
plt.plot(thresholds, assortativity, marker='o', color='teal')
for x, y in zip(thresholds, assortativity):
    plt.text(x, y + 0.005, f"{y:.3f}", ha='center', fontsize=8)

plt.xlabel("Tolerance")
plt.ylabel("Assortativity Coefficient")
plt.title("Assortativity vs Tolerance")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Step 7: Clustering Coefficients
global_C = df_metrics["global_clustering"]
local_C = df_metrics["average_clustering"]


plt.figure(figsize=(8, 5))
plt.plot(thresholds, global_C, marker='o', label="Global clustering", color='blue')
plt.plot(thresholds, local_C, marker='s', label="Average local clustering", color='orange')

for x, y in zip(thresholds, global_C):
    plt.text(x, y + 0.001, f"{y:.3f}", ha='center', fontsize=8)
for x, y in zip(thresholds, local_C):
    plt.text(x, y + 0.001, f"{y:.3f}", ha='center', fontsize=8)

plt.xlabel("Tolerance")
plt.ylabel("Clustering Coefficient")
plt.title("Clustering vs Tolerance")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Step 8: Density vs Tolerance
densities = df_metrics["density"]
plt.figure(figsize=(8, 5))
plt.plot(thresholds, densities, marker='o', color='green', label="Density")
for x, y in zip(thresholds, densities):
    plt.text(x, y + 1e-7, f"{y:.1e}", ha='center', fontsize=8)

plt.xlabel("Tolerance")
plt.ylabel("Density")
plt.title("Network Density vs Tolerance")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Step 9: Component Size CDF
def plot_component_cdf(ax, sizes, title, color):
    sizes_sorted = sorted(sizes)
    cdf = np.arange(1, len(sizes_sorted) + 1) / len(sizes_sorted)
    ax.plot(sizes_sorted, cdf, marker='o', linestyle='none', color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Component Size (nodes)")
    ax.set_ylabel("Cumulative Fraction (CDF)")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)

sizes_t1 = df_metrics[df_metrics["tolerance"] == 1]["component_sizes"].values[0]
sizes_t2 = df_metrics[df_metrics["tolerance"] == 2]["component_sizes"].values[0] if 2 in thresholds else []

if sizes_t2:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_component_cdf(axes[0], sizes_t1, "Component Size CDF (t=1)", "darkgreen")
    plot_component_cdf(axes[1], sizes_t2, "Component Size CDF (t=2)", "darkorange")
else:
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_component_cdf(ax, sizes_t1, "Component Size CDF (t=1)", "darkgreen")

plt.tight_layout()
plt.show()
