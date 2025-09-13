import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# =======================================================
# This script assumes 'weatherHistory.csv' is in the same
# directory. You can place the file there directly.
# =======================================================

# ==============================
# 1. Load dataset
# ==============================
try:
    # Use the script's directory to find the file, making it robust
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "weatherHistory.csv")
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file 'weatherHistory.csv' was not found at '{file_path}'.")
    print("Please make sure the dataset is in the same directory as this script.")
    exit()

# Keep only numerical features
numerical_features = [
    "Temperature (C)", "Apparent Temperature (C)", "Humidity",
    "Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)",
    "Pressure (millibars)"
]

df = df[numerical_features].dropna()  # Drop missing values
print("Dataset shape:", df.shape)
print(df.head())

# ==============================
# 2. Normalize numerical features
# ==============================
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[numerical_features])

# === Reshape data into sequences for LSTM ===
sequence_length = 24  # Example: 24 hours of data
num_sequences = data_scaled.shape[0] // sequence_length
data_tensor = torch.tensor(data_scaled[:num_sequences * sequence_length], dtype=torch.float32)
data_tensor = data_tensor.view(num_sequences, sequence_length, -1)

dataset = TensorDataset(data_tensor)

# You can try increasing this for a speed boost, e.g., to 128 or 256
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==============================
# 3. Define GAN models
# ==============================
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # Initial hidden state and cell state for the LSTM
        h0 = torch.zeros(self.num_layers, z.size(0), self.hidden_dim).to(z.device)
        c0 = torch.zeros(self.num_layers, z.size(0), self.hidden_dim).to(z.device)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(z, (h0, c0))
        
        # Pass through linear layer to get the final output sequence
        output_seq = self.linear(lstm_out)
        return torch.sigmoid(output_seq)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Pass through LSTM
        _, (h_n, c_n) = self.lstm(x)
        
        # Take the hidden state of the last layer for classification
        out = self.linear(h_n[-1])
        return torch.sigmoid(out)

# ==============================
# 4. Setup
# ==============================
latent_dim = 100
data_dim = len(numerical_features)
hidden_dim = 128
num_layers = 1
num_epochs = 1200

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# === Pass LSTM parameters to models ===
G = Generator(latent_dim, hidden_dim, data_dim, num_layers).to(device)
D = Discriminator(data_dim, hidden_dim, num_layers).to(device)

criterion = nn.BCELoss()
# === CHANGE: Reduced D_lr to prevent it from overpowering G ===
optimizer_G = optim.Adam(G.parameters(), lr=0.0001)
optimizer_D = optim.Adam(D.parameters(), lr=0.00002)

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

        # === Generate a sequence of noise for the LSTM ===
        noise = torch.randn(batch_size_curr, sequence_length, latent_dim).to(device)
        fake_samples = G(noise)
        labels_fake = torch.zeros(batch_size_curr, 1).to(device)
        output_fake = D(fake_samples.detach())
        loss_fake = criterion(output_fake, labels_fake)

        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # === Train Generator less frequently to maintain balance ===
        if i % 2 == 0:
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
    noise = torch.randn(6000, sequence_length, latent_dim).to(device)
    synthetic_data = G(noise).cpu().numpy()

# === Reshape synthetic data back to a flat array before inverse transform ===
synthetic_data_flat = synthetic_data.reshape(-1, data_dim)
synthetic_data_original_scale = scaler.inverse_transform(synthetic_data_flat)

synthetic_df = pd.DataFrame(synthetic_data_original_scale, columns=numerical_features)
print("\nSynthetic data (first 10 rows):")
print(synthetic_df.head(10))

# ==============================
# 7. Save synthetic data to CSV
# ==============================
synthetic_df.to_csv("synthetic_weather_lstm4.csv", index=False)
print("\nSynthetic data saved to 'synthetic_weather_lstm4.csv'")
