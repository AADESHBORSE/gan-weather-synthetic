import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# =======================================================
# This script loads and compares two datasets.
# It calculates RMSE and normalized RMSE between the
# numerical features of the datasets.
# =======================================================

# Load the datasets
try:
    original_data = pd.read_csv("/Users/AadeshBorse/Desktop/Aadesh/EJ_project/weatherHistory.csv")
    synthetic_data = pd.read_csv("/Users/AadeshBorse/Desktop/Aadesh/EJ_project/synthetic_weather(colab1).csv")

except FileNotFoundError as e:
    print(f"Error: {e}. Please check your file paths.")
    exit()

# --- Aligning Datasets ---

# 1. Align columns: keep only those present in synthetic
synthetic_cols = synthetic_data.columns.tolist()
original_aligned = original_data[synthetic_cols]

# 2. Align rows by sampling the larger dataset
n_rows_original = len(original_aligned)
n_rows_synthetic = len(synthetic_data)

if n_rows_original > n_rows_synthetic:
    print(f"Sampling original data from {n_rows_original} → {n_rows_synthetic} rows.")
    real_numerical = original_aligned.sample(n=n_rows_synthetic, random_state=42)
    synthetic_numerical = synthetic_data
else:
    print(f"Sampling synthetic data from {n_rows_synthetic} → {n_rows_original} rows.")
    synthetic_numerical = synthetic_data.sample(n=n_rows_original, random_state=42)
    real_numerical = original_aligned

# Select only the numerical columns
numerical_features = real_numerical.select_dtypes(include=np.number).columns

# --- Handle NaNs ---
real_numerical = real_numerical.fillna(0).reset_index(drop=True)
synthetic_numerical = synthetic_numerical.fillna(0).reset_index(drop=True)

# --- Calculating RMSE & Normalized RMSE ---
rmse_scores = {}
nrmse_scores = {}

for col in numerical_features:
    mse = mean_squared_error(real_numerical[col], synthetic_numerical[col])
    rmse = np.sqrt(mse)
    rmse_scores[col] = rmse
    
    # Normalized RMSE (scaled by std of real data)
    std_real = real_numerical[col].std()
    if std_real != 0:
        nrmse_scores[col] = rmse / std_real
    else:
        nrmse_scores[col] = np.nan  # avoid divide by zero

# --- Display Results ---
print("\n🔍 **RMSE per feature:**")
for k, v in rmse_scores.items():
    print(f"{k}: {v:.4f}")

print("\n📊 **Normalized RMSE per feature (scaled by std):**")
for k, v in nrmse_scores.items():
    print(f"{k}: {v:.4f}")

# Overall averages
mean_rmse = np.mean(list(rmse_scores.values()))
mean_nrmse = np.nanmean(list(nrmse_scores.values()))

print(f"\n⚡ Mean RMSE across all numerical features: {mean_rmse:.4f}")
print(f"⚡ Mean Normalized RMSE across all numerical features: {mean_nrmse:.4f}")

# --- Save results ---
results_df = pd.DataFrame({
    "Feature": numerical_features,
    "RMSE": [rmse_scores[col] for col in numerical_features],
    "Normalized_RMSE": [nrmse_scores[col] for col in numerical_features]
})

results_df.to_csv("rmse_results(colab1).csv", index=False)
print("\n✅ Results saved to rmse_results(colab1).csv")
