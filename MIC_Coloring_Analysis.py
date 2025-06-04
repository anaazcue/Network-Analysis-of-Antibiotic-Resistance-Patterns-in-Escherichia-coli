"""
Profile-Based Filtering of Antibiotic Resistance Data (Escherichia coli)

This script allows precise filtering of isolates based on predefined MIC (log2-transformed)
resistance profiles for *Escherichia coli*. Each node in the dictionary represents a unique
resistance vector used to extract matching isolates from the final dataset.

Features:
- Loads preprocessed dataset from disk.
- Defines resistance vectors as node profiles.
- Filters isolates matching a given node profile.
- Extracts and displays MIC interpretation consistency.
- Displays overall interpretation statistics.

Author: Ana Azcue Unzueta
"""

import pandas as pd

# Load final dataset (previously saved)
datos_final = pd.read_pickle("datos_final.pkl")

# Selected antibiotics (in order)
antibiotic_columns = [
    "Cefoperazone sulbactam", "Trimethoprim sulfa", "Ciprofloxacin", "Ampicillin sulbactam",
    "Gentamicin", "Colistin", "Aztreonam avibactam", "Aztreonam", "Ceftazidime avibactam",
    "Ceftaroline", "Ceftazidime", "Imipenem", "Meropenem", "Ampicillin", "Cefepime",
    "Piperacillin tazobactam", "Levofloxacin", "Tigecycline", "Amoxycillin clavulanate",
    "Amikacin"
]

# ------------------------------------------------------------------------------
# 1. Filtering function for exact resistance profile
# ------------------------------------------------------------------------------
def filter_by_profile(df, columns, values):
    if len(columns) != len(values):
        raise ValueError("Columns and values must have the same length.")
    return df.loc[(df[columns] == values).all(axis=1)]

# ------------------------------------------------------------------------------
# 2. Dictionary of nodes with antibiotic MIC profiles (order must match columns)
# ------------------------------------------------------------------------------
nodes = {
    0:  [-2.0, 0.0, -4.06, 1.0, -1.0, -3.06, -4.06, -3.06, -3.06, -4.06, -2.0, -2.0, -4.06, 2.0, -3.06, 1.0, -2.0, -3.06, 1.0, 1.0],
    1:  [-3.06, 0.0, -4.06, 0.0, -1.0, -3.06, -5.06, -4.06, -4.06, -5.06, -3.06, -3.06, -4.06, 1.0, -3.06, 0.0, -2.0, -3.06, 1.0, 1.0],
    2:  [-3.06, 0.0, -4.06, 0.0, -1.0, -3.06, -5.06, -4.06, -4.06, -4.06, -3.06, -3.06, -4.06, 2.0, -3.06, 0.0, -2.0, -3.06, 1.0, 1.0],
    3:  [-3.06, 0.0, -4.06, 0.0, -1.0, -3.06, -5.06, -4.06, -4.06, -4.06, -3.06, -3.06, -4.06, 1.0, -3.06, 0.0, -2.0, -3.06, 1.0, 1.0],
    4:  [-2.0, 0.0, -4.06, 1.0, -1.0, -2.0, -4.06, -3.06, -3.06, -4.06, -2.0, -2.0, -4.06, 2.0, -3.06, 1.0, -2.0, -3.06, 1.0, 1.0],
    21: [-3.06, 0.0, -4.06, 0.0, -1.0, -2.0, -5.06, -4.06, -4.06, -4.06, -3.06, -2.0, -4.06, 1.0, -3.06, 0.0, -2.0, -3.06, 1.0, 1.0],
    38: [-3.06, 0.0, -4.06, 0.0, -1.0, -3.06, -5.06, -4.06, -4.06, -4.06, -3.06, -2.0, -4.06, 1.0, -3.06, 0.0, -2.0, -3.06, 1.0, 1.0],
    50: [-3.06, 0.0, -3.06, 0.0, -1.0, -2.0, -5.06, -4.06, -4.06, -5.06, -3.06, -3.06, -4.06, 1.0, -3.06, 0.0, -2.0, -3.06, 1.0, 1.0],
    72: [-3.06, 0.0, -4.06, 0.0, -1.0, -2.0, -5.06, -4.06, -4.06, -5.06, -3.06, -2.0, -4.06, 1.0, -3.06, 0.0, -2.0, -3.06, 1.0, 1.0],
    75: [-3.06, 0.0, -4.06, 0.0, -1.0, -2.0, -5.06, -4.06, -4.06, -4.06, -3.06, -3.06, -4.06, 1.0, -3.06, 0.0, -2.0, -3.06, 1.0, 1.0],
    120: [-1.0, 5.0, -4.06, 3.0, -1.0, -2.0, -5.06, -4.06, -4.06, -3.06, -3.06, -2.0, -4.06, 4.0, -3.06, 0.0, -2.0, -2.0, 2.0, 1.0],
    126: [-3.06, 0.0, -3.06, 0.0, -1.0, -2.0, -5.06, -4.06, -4.06, -4.06, -3.06, -3.06, -4.06, 1.0, -3.06, 0.0, -2.0, -3.06, 1.0, 1.0],
    173: [-3.06, 0.0, -4.06, 0.0, -1.0, -2.0, -5.06, -4.06, -4.06, -4.06, -3.06, -3.06, -4.06, 2.0, -3.06, 1.0, -2.0, -3.06, 1.0, 1.0],
    253: [-2.0, 0.0, -4.06, 1.0, -1.0, -2.0, -4.06, -3.06, -3.06, -3.06, -2.0, -2.0, -4.06, 2.0, -3.06, 1.0, -2.0, -3.06, 2.0, 1.0],
    297: [-3.06, 0.0, -3.06, 0.0, -1.0, -2.0, -6.06, -4.06, -4.06, -4.06, -3.06, -3.06, -4.06, 1.0, -3.06, 0.0, -2.0, -3.06, 1.0, 1.0],
    303: [-3.06, 0.0, -3.06, 0.0, -1.0, -2.0, -5.06, -4.06, -4.06, -4.06, -3.06, -3.06, -4.06, 2.0, -3.06, 0.0, -2.0, -3.06, 1.0, 1.0],
    18424: [-2.0, 0.0, -4.06, 2.0, -1.0, -3.06, -4.06, -3.06, -3.06, -4.06, -2.0, -2.0, -4.06, 2.0, -3.06, 1.0, -2.0, -2.0, 2.0, 3.0],
    20848: [-3.06, 0.0, -4.06, 0.0, -1.0, -2.0, -6.06, -4.06, -4.06, -4.06, -3.06, -3.06, -4.06, 1.0, -3.06, 0.0, -2.0, -3.06, 1.0, 1.0],
    22892: [-3.06, 0.0, -3.06, 1.0, -1.0, -1.0, -6.06, -5.06, -5.06, -5.06, -3.06, -3.06, -4.06, 1.0, -3.06, -1.0, -2.0, -3.06, 1.0, 1.0]
}

# ------------------------------------------------------------------------------
# 3. Select a node and filter the original DataFrame
# ------------------------------------------------------------------------------
selected_node_id = 1  # change this ID to analyze another node
selected_profile = nodes[selected_node_id]
filtered_node = filter_by_profile(datos_final, antibiotic_columns, selected_profile)

# ------------------------------------------------------------------------------
# 4. Print unique values of columns ending with '_I' (MIC interpretations)
# ------------------------------------------------------------------------------
I_columns = [col for col in filtered_node.columns if col.endswith('_I')]

for col in I_columns:
    values = filtered_node[col].dropna().unique()
    if len(values) == 1:
        print(f"{col}: {values[0]}")
    else:
        print(f"{col}: ⚠ Multiple values -> {values}")

# ------------------------------------------------------------------------------
# 5. Display global value counts for each MIC interpretation column
# ------------------------------------------------------------------------------
for col in I_columns:
    print(f"Unique values in column '{col}':")
    print(datos_final[col].value_counts())
    print("\n" + "-"*50 + "\n")
