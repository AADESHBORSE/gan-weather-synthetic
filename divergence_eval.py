import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


# =======================================================
# This script loads and compares two datasets using
# KL and JS divergence.
# =======================================================

# Load the datasets
try:
    script_dir = os.path.dirname(__file__)
    original_data = pd.read_csv(os.path.join(script_dir, 'weatherHistory.csv'))
    synthetic_data = pd.read_csv(os.path.join(script_dir, 'nowidk.csv'))
except FileNotFoundError as e:
    print(f"Error: {e}. Please check your file paths.")
    exit()

# --- Aligning Datasets and selecting features ---
numerical_features = [
    "Temperature (C)", "Apparent Temperature (C)", "Humidity",
    "Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)",
     "Pressure (millibars)"  #<-- This feature has been removed for evaluation
]

# 1. Align columns
synthetic_data = synthetic_data[numerical_features]
original_aligned = original_data[numerical_features]

# 2. Align rows by sampling the larger dataset
n_rows_original = len(original_aligned)
n_rows_synthetic = len(synthetic_data)

if n_rows_original > n_rows_synthetic:
    # If the original data is larger, sample it
    print(f"Sampling original data from {n_rows_original} to {n_rows_synthetic} rows.")
    real_numerical = original_aligned.sample(n=n_rows_synthetic, random_state=42)
    synthetic_numerical = synthetic_data
else:
    # If the synthetic data is larger, sample it
    print(f"Sampling synthetic data from {n_rows_synthetic} to {n_rows_original} rows.")
    synthetic_numerical = synthetic_data.sample(n=n_rows_original, random_state=42)
    real_numerical = original_aligned

# --- Calculating Divergence ---
print("\n=== GAN Performance Evaluation (Divergence Metrics) ===")
print("-----------------------------------------------------")
print("Lower values indicate a better match between distributions.")
print("-----------------------------------------------------")
print(f"{'Feature':<25}{'KL Divergence':<20}{'JS Divergence':<20}")
print("-----------------------------------------------------")

divergence_scores = {}
for col in numerical_features:
    # Create histograms to represent probability distributions
    bins = 50
    hist_real, _ = np.histogram(real_numerical[col], bins=bins, density=True)
    hist_synthetic, _ = np.histogram(synthetic_numerical[col], bins=bins, density=True)
    
    # Add a small epsilon to avoid division by zero in KL divergence
    hist_real = hist_real + 1e-10
    hist_synthetic = hist_synthetic + 1e-10

    # Calculate divergences
    kl_div = entropy(hist_real, qk=hist_synthetic)
    js_div = jensenshannon(hist_real, hist_synthetic)
    
    divergence_scores[col] = {'KL': kl_div, 'JS': js_div}

    print(f"{col:<25}{kl_div:<20.4f}{js_div:<20.4f}")
