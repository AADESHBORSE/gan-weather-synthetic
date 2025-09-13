import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time

# =========================
# 0. Config (choose mode)
# =========================
MODE = "debug"   # change to "full" for long training

if MODE == "debug":
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    SAMPLE_SIZE = 5000
    PRINT_EVERY = 10
    NUM_SAMPLES_GENERATE = 1000
else:  # full training mode
    NUM_EPOCHS = 1000
    BATCH_SIZE = 64
    SAMPLE_SIZE = None   # use full dataset
    PRINT_EVERY = 50
    NUM_SAMPLES_GENERATE = 5000

# =========================
# 1. Load and Preprocess Data
# =========================
df = pd.read_csv("/Users/AadeshBorse/Downloads/weatherHistory.csv")

numerical_features = [
    "Temperature (C)",
    "Apparent Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)",
]

df = df[numerical_features].dropna()

# For debug, sample a subset
if SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=42)

data = df.values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

tensor_data = torch.tensor(data_scaled, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# 2. Define Models
# =========================
latent_dim = 16

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.model(x)

# =========================
# 3. Device Configuration
# =========================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (no GPU found)")
print(f"Using device: {device}")

generator = Generator(latent_dim, len(numerical_features)).to(device)
discriminator = Discriminator(len(numerical_features)).to(device)

criterion = nn.BCELoss()
optimizer_d = optim.Adam(discriminator.parameters(), lr=5e-5, betas=(0.5, 0.999))
optimizer_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))

d_losses, g_losses = [], []

# =========================
# 4. Training Loop
# =========================
print(f"\n🚀 Starting training in {MODE.upper()} mode for {NUM_EPOCHS} epochs...\n")
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    for real_samples, in dataloader:
        real_samples = real_samples.to(device)
        batch_size = real_samples.size(0)

        # Train Discriminator
        optimizer_d.zero_grad()
        real_labels = torch.full((batch_size, 1), 0.9, device=device)
        fake_labels = torch.full((batch_size, 1), 0.0, device=device)

        real_output = discriminator(real_samples)
        d_loss_real = criterion(real_output, real_labels)

        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_samples = generator(noise).detach()
        fake_output = discriminator(fake_samples)
        d_loss_fake = criterion(fake_output, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        noise = torch.randn(batch_size, latent_dim, device=device)
        generated_samples = generator(noise)
        output = discriminator(generated_samples)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizer_g.step()

    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())

    if epoch % PRINT_EVERY == 0:
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Time: {elapsed:.1f}s")

# =========================
# 5. Plot Loss Curves
# =========================
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title(f"GAN Training Losses ({MODE.upper()} mode)")
plt.show()

# =========================
# 6. Generate Synthetic Data & Evaluate
# =========================
noise = torch.randn(NUM_SAMPLES_GENERATE, latent_dim, device=device)
synthetic_data = generator(noise).detach().cpu().numpy()
synthetic_data_original_scale = scaler.inverse_transform(synthetic_data)

real_batch = df.sample(NUM_SAMPLES_GENERATE).to_numpy()
rmse = np.sqrt(mean_squared_error(real_batch, synthetic_data_original_scale))
print(f"\n⚡ Overall RMSE: {rmse:.4f}")

for i, feature in enumerate(numerical_features):
    feature_rmse = np.sqrt(mean_squared_error(real_batch[:, i], synthetic_data_original_scale[:, i]))
    print(f"{feature}: {feature_rmse:.4f}")

# =========================
# 7. Save Synthetic Data
# =========================
synthetic_df = pd.DataFrame(synthetic_data_original_scale, columns=numerical_features)
output_path = f"/Users/AadeshBorse/Desktop/Aadesh/EJ_project/synthetic_weather1_{MODE}.csv"
synthetic_df.to_csv(output_path, index=False)
print(f"\n✅ Synthetic data saved to {output_path}")
