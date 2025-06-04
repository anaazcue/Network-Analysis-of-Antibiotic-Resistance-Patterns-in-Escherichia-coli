"""
Missing Data Analysis in Antibiotic Resistance Profiles (Escherichia coli)

This script analyzes missing data (NaN values) in antibiotic susceptibility testing results
for *Escherichia coli* collected in the Atlas dataset. It includes:

1. Histogram of missing values per antibiotic.
2. Correlation heatmap of missingness between antibiotics.
3. Temporal trends of missing data (by year).
4. Heatmap of NaN proportions grouped by category (e.g., year).
5. NaN trends by study and country/source, with confidence shading.

The visualizations help assess data quality, detect patterns in missingness,
and support preprocessing decisions for downstream modeling.

Author: Ana Azcue Unzueta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# 1. Load and preprocess data
# ------------------------------------------------------------------------------
data = pd.read_csv("2024_05_28 atlas_antibiotics.csv")
data_ecoli = data[data["Species"] == "Escherichia coli"]

all_antibiotics = [
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

selected_antibiotics = [
    "Cefoperazone sulbactam", "Trimethoprim sulfa", "Ciprofloxacin", "Ampicillin sulbactam",
    "Gentamicin", "Colistin", "Aztreonam avibactam", "Aztreonam", "Ceftazidime avibactam", "Ceftaroline",
    "Imipenem", "Ceftazidime", "Meropenem", "Ampicillin", "Cefepime", "Piperacillin tazobactam",
    "Levofloxacin", "Tigecycline", "Amoxycillin clavulanate", "Amikacin"
]

# ------------------------------------------------------------------------------
# 2. Figure 1: NaN count per antibiotic
# ------------------------------------------------------------------------------
def plot_nan_histogram(df, antibiotics):
    subset = df[antibiotics]
    nan_counts = subset.isna().sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(nan_counts.index, nan_counts.values, color='skyblue')
    plt.xticks(rotation=90)
    plt.xlabel("Antibiotics")
    plt.ylabel("Number of NaNs")
    plt.title("Number of NaNs in Escherichia coli")
    plt.tight_layout()
    plt.show()

    return nan_counts, subset.shape[0]

nan_counts, total_rows = plot_nan_histogram(data_ecoli, all_antibiotics)
updated_antibiotics = nan_counts[nan_counts < 0.95 * total_rows].index.tolist()
data_ecoli = data_ecoli.drop(columns=[col for col in all_antibiotics if col not in updated_antibiotics])

# ------------------------------------------------------------------------------
# 3. Figure 2: Correlation of missing values
# ------------------------------------------------------------------------------
def plot_nan_correlation(df, antibiotics):
    nan_matrix = df[antibiotics].isnull().astype(float)
    corr = nan_matrix.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation of NaNs among antibiotics")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    return corr

correlation_matrix = plot_nan_correlation(data_ecoli, updated_antibiotics)

# ------------------------------------------------------------------------------
# 4. Figure 3: NaN trend by year
# ------------------------------------------------------------------------------
def plot_nan_trend(df, antibiotics, category="Year"):
    trend = df.groupby(category)[antibiotics].apply(
        lambda x: x.isnull().sum().sum() / (x.shape[0] * x.shape[1]) * 100
    ).sort_index()

    plt.figure(figsize=(8, 5))
    plt.plot(trend.index, trend.values, marker='o', color='red')
    for start in [2010.5, 2016.5, 2020.5]:
        plt.axvspan(start, start + 2, color='yellow', alpha=0.3)
    plt.title("NaN % Trend by Year")
    plt.xlabel("Year")
    plt.ylabel("% of NaNs")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return trend

trend_by_year = plot_nan_trend(data_ecoli, updated_antibiotics)

# ------------------------------------------------------------------------------
# 5. Figure 4: NaN heatmap by category
# ------------------------------------------------------------------------------
def plot_nan_heatmap_by_category(df, antibiotics, category):
    if category not in df.columns:
        raise ValueError(f"Column '{category}' not in dataset")
    heat_data = df.groupby(category)[antibiotics].apply(lambda x: x.isnull().mean())

    plt.figure(figsize=(16, 8))
    sns.heatmap(heat_data, cmap="viridis", linewidths=0.5)
    plt.title(f"NaN Proportion by {category}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    return heat_data

nan_by_year = plot_nan_heatmap_by_category(data_ecoli, updated_antibiotics, "Year")

# ------------------------------------------------------------------------------
# 6. Figure 5 & 6: NaN trend by study and grouping
# ------------------------------------------------------------------------------
def plot_nan_trend_by_study(df, antibiotics, studies, group_by='Country'):
    plt.figure(figsize=(10, 6))

    for study in studies:
        df_study = df[df['Study'] == study]
        stats = []

        for year in sorted(df_study['Year'].dropna().unique()):
            df_year = df_study[df_study['Year'] == year]
            valid_cols = df_year[antibiotics].dropna(axis=1, how='all').columns
            if valid_cols.empty:
                continue

            nan_by_group = df_year.groupby(group_by)[valid_cols].apply(lambda x: x.isnull().mean().mean() * 100)
            if nan_by_group.empty:
                continue

            stats.append({
                'Year': year,
                'Mean': nan_by_group.mean(),
                'Std': nan_by_group.std()
            })

        df_stats = pd.DataFrame(stats)
        if df_stats.empty:
            continue

        plt.plot(df_stats['Year'], df_stats['Mean'], label=study, marker='o')
        plt.fill_between(df_stats['Year'], df_stats['Mean'] - df_stats['Std'], df_stats['Mean'] + df_stats['Std'], alpha=0.2)

    plt.title(f"NaN % Trend per Year ({group_by}) by Study")
    plt.xlabel("Year")
    plt.ylabel("NaN Percentage (%)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    years = df['Year'].dropna().astype(int)
    plt.xticks(np.arange(years.min(), years.max() + 1, 1), rotation=45)
    plt.tight_layout()
    plt.show()

plot_nan_trend_by_study(data_ecoli, updated_antibiotics, ["TEST", "Inform", "Atlas"], group_by='Country')
plot_nan_trend_by_study(data_ecoli, updated_antibiotics, ["TEST", "Inform", "Atlas"], group_by='Source')
