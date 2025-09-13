# pip installs (only needed once if missing)
# !pip install seaborn scipy scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression

# ==============================
# 1. Load real + synthetic datasets
# ==============================
real_path = "/Users/AadeshBorse/Desktop/Aadesh/EJ_project/weatherHistory.csv"
synth_path = "/Users/AadeshBorse/Desktop/Aadesh/EJ_project/synthetic_weather(colab1).csv"

df = pd.read_csv(real_path)
synthetic_df = pd.read_csv(synth_path)

# --- Clean column names ---
df.columns = df.columns.str.strip()
synthetic_df.columns = synthetic_df.columns.str.strip()

# --- Keep only common columns and align order ---
common_cols = df.columns.intersection(synthetic_df.columns)
df = df[common_cols]
synthetic_df = synthetic_df[common_cols]

print("Aligned dataset columns:", list(common_cols))

# ==============================
# 2. Quick EDA
# ==============================
print(df.describe())

# Pairplot (small sample for speed)
sns.pairplot(df.sample(min(1000, len(df))))  
plt.savefig("pairplot_sample.png", dpi=200)
plt.close()

# ==============================
# 3. Histogram comparison
# ==============================
def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\s-]", "_", name).replace(" ", "_")

def plot_feature_hist(real_df, synth_df, feature, bins=50):
    plt.figure(figsize=(6,4))
    sns.histplot(real_df[feature], stat="density", bins=bins, label="Real", alpha=0.6)
    sns.histplot(synth_df[feature], stat="density", bins=bins, label="Synthetic", alpha=0.6)
    plt.title(feature)
    plt.legend()
    plt.tight_layout()
    safe_name = sanitize_filename(feature)
    plt.savefig(f"hist_{safe_name}.png", dpi=200)
    plt.close()

for feat in common_cols:
    plot_feature_hist(df, synthetic_df, feat)

# ==============================
# 4. Statistical metrics per feature
# ==============================
def feature_metrics(real, synth, n_bins=100):
    ks = ks_2samp(real, synth).statistic
    wd = wasserstein_distance(real, synth)
    hr, _ = np.histogram(real, bins=n_bins, density=True)
    hs, _ = np.histogram(synth, bins=n_bins, density=True)
    js = jensenshannon(hr + 1e-12, hs + 1e-12)
    return ks, wd, js

metrics = {}
for feat in common_cols:
    ks, wd, js = feature_metrics(df[feat].values, synthetic_df[feat].values)
    metrics[feat] = {"KS": ks, "Wasserstein": wd, "JS": js}

metrics_df = pd.DataFrame(metrics).T
print(metrics_df)
metrics_df.to_csv("feature_metrics(colab11).csv")

# ==============================
# 5. Classifier two-sample test
# ==============================
real_X = df.values
synth_X = synthetic_df.values
X = np.vstack([real_X, synth_X])
y = np.hstack([np.ones(len(real_X)), np.zeros(len(synth_X))])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
clf.fit(X_train, y_train)
auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
print("Two-sample classifier AUC:", auc)

# ==============================
# 6. Downstream utility example
# ==============================
target_col = "Temperature (C)"
if target_col in common_cols:
    features = [c for c in common_cols if c != target_col]

    X_real = df[features].values
    y_real = df[target_col].values
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_real, y_real, test_size=0.3, random_state=0
    )
    lr = LinearRegression().fit(X_train_r, y_train_r)
    mse_real = np.mean((lr.predict(X_test_r) - y_test_r)**2)

    X_synth = synthetic_df[features].values
    y_synth = synthetic_df[target_col].values
    lr2 = LinearRegression().fit(X_synth, y_synth)
    mse_synth_train = np.mean((lr2.predict(X_test_r) - y_test_r)**2)

    print("MSE (train real -> test real):", mse_real)
    print("MSE (train synthetic -> test real):", mse_synth_train)
else:
    print(f"⚠️ Target column '{target_col}' not found in common features. Skipping downstream test.")
