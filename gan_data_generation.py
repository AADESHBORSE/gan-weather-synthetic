import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# ==============================
# 1. Load dataset (local file)
# ==============================
# Place "weatherHistory.csv" in the same folder as this script, or update the path
DATA_PATH = "weatherHistory.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please place the CSV file in your project folder.")

df = pd.read_csv(DATA_PATH)

# Keep only numerical features
numerical_features = [
    "Temperature (C)", "Apparent Temperature (C)", "Humidity",
    "Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)",
    "Pressure (millibars)"
]

df = df[numerical_features].dropna()
print("Dataset shape:", df.shape)
print(df.head())

# ==============================
# 2. Normalize features
# ==============================
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[numerical_features])

data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
dataset = TensorDataset(data_tensor)

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==============================
# 3. Define GAN models
# ==============================
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ==============================
# 4. Setup
# ==============================
latent_dim = 100
data_dim = len(numerical_features)
num_epochs = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

G = Generator(latent_dim, data_dim).to(device)
D = Discriminator(data_dim).to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

# ==============================
# 5. Training Loop
# ==============================
for epoch in range(num_epochs):
    for i, (real_samples,) in enumerate(dataloader):
        real_samples = real_samples.to(device)
        batch_size_curr = real_samples.size(0)

        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        labels_real = torch.ones(batch_size_curr, 1).to(device)
        output_real = D(real_samples)
        loss_real = criterion(output_real, labels_real)

        noise = torch.randn(batch_size_curr, latent_dim).to(device)
        fake_samples = G(noise)
        labels_fake = torch.zeros(batch_size_curr, 1).to(device)
        output_fake = D(fake_samples.detach())
        loss_fake = criterion(output_fake, labels_fake)

        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # --- Train Generator ---
        optimizer_G.zero_grad()
        output_fake_for_G = D(fake_samples)
        loss_G = criterion(output_fake_for_G, labels_real)
        loss_G.backward()
        optimizer_G.step()

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"[Epoch {epoch}/{num_epochs}] D_loss: {loss_D.item():.4f}, G_loss: {loss_G.item():.4f}")

# ==============================
# 6. Generate synthetic data
# ==============================
G.eval()
with torch.no_grad():
    noise = torch.randn(5000, latent_dim).to(device)
    synthetic_data = G(noise).cpu().numpy()

# Inverse transform
synthetic_data_original_scale = scaler.inverse_transform(synthetic_data)

synthetic_df = pd.DataFrame(synthetic_data_original_scale, columns=numerical_features)
print("\nSynthetic data (first 10 rows):")
print(synthetic_df.head(10))

# ==============================
# 7. Save synthetic data
# ==============================
OUTPUT_PATH = "synthetic_weather.csv"
synthetic_df.to_csv(OUTPUT_PATH, index=False)
print(f"Synthetic dataset saved as {OUTPUT_PATH}")
