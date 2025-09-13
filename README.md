# 🌦️ Weather Data GAN

This project trains a **Generative Adversarial Network (GAN)** using PyTorch to generate **synthetic weather data** based on the `weatherHistory.csv` dataset.

---

## 📂 Project Structure
weather-gan/
│── weather_gan.py
│── weatherHistory.csv
│── synthetic_weather.csv
│── requirements.txt
│── README.md


---

## ⚙️ Requirements

- Python **3.9+** (recommended: 3.10 or 3.11)
- PyTorch
- numpy, pandas, scikit-learn

Optional GPU: install PyTorch with CUDA for faster training.

---

## 🔧 Setup Instructions

1. Place `weatherHistory.csv` in the project folder.
2. (Optional) Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
