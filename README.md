# 📘 CAC40 Market Movement Prediction  
**Midterm CAPSTONE Project – MLZoomcamp 2025**

---

## 📌 Overview

This project applies Machine Learning to financial market data from CAC40 companies to **predict short-term market movement**.  
It follows the complete MLZoomcamp pipeline:

1. Pick a problem & dataset  
2. Describe how ML helps  
3. Prepare data & run EDA  
4. Train several models & select the best  
5. Export the trained model  
6. Package the model as a FastAPI service  
7. Deploy the model with Docker  

This repository includes:

- Dataset  
- Jupyter Notebooks (EDA, feature engineering, training, predictions)  
- Python scripts (`train.py`, `predict.py`, `api.py`)  
- Trained model files (`model.pkl`, `scaler.pkl`, `features.json`)  
- Dockerfile for deployment  

---

# 🎯 1. Problem Definition

Financial markets move quickly and are influenced by many variables such as price changes, volume, and technical indicators.  
The objective is to build a machine learning model that **predicts a binary market movement signal ("Target")** for CAC40 stocks.

The ML model can be used for:

- Short-term signal generation  
- Automated trading strategies  
- Market monitoring tools  

---

# 📚 2. Dataset

Daily data for CAC40 companies including technical indicators.

**Columns include:**

symbol, date, open, high, low, close, volume,
adjclose, Return, MA20, MA50, Volatility, RSI, Target



- All values are numerical except symbol/date (string).  
- Stored in `data/cac40_features.csv`.

---

# 🔎 3. Exploratory Data Analysis (EDA)

Performed in `cac40_analysis.ipynb`:

- Data inspection  
- Missing value analysis  
- Feature distribution  
- Correlation heatmap  
- Target distribution  
- Visual analysis of market features  

This ensures data reliability before training.

---

# 🧠 4. Model Training

Models evaluated:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost (**best performance**)  

**Evaluation metrics:**

- Accuracy  
- Precision / Recall  
- F1-score  
- Confusion matrix  

Training workflow available in:

- `train.ipynb`
- `train.py`

**Files exported:**

model.pkl
scaler.pkl
features.json



---

# 🔮 5. Prediction Pipeline

Available in:

- `predict.ipynb`
- `predict.py`

Prediction workflow:

1. Load `model.pkl`  
2. Load `scaler.pkl`  
3. Read feature order from `features.json`  
4. Validate and process input  
5. Scale features  
6. Predict binary value (0 or 1)  

---

# 🚀 6. FastAPI Web Service

The file **`api.py`** provides a real-time prediction API.

### Endpoints

#### `GET /`
Simple health check.

#### `POST /predict`
Accepts JSON input and returns the model prediction.

### Example input

```json
{
  "open": 8000,
  "high": 8100,
  "low": 7950,
  "close": 8050,
  "volume": 12000000,
  "adjclose": 8050,
  "Return": 0.0041,
  "MA20": 7900,
  "MA50": 7800,
  "Volatility": 0.012,
  "RSI": 54
}
```
Swagger UI:

👉 http://localhost:8000/docs

## 🐳 7. Docker Deployment

This project is fully containerized.

Step 1 — Build the Docker Image

docker build -t ml-api .


Step 2 — Run the Container

docker run -d -p 8000:8000 ml-api


Step 3 — Access API

Swagger interface:

👉 http://localhost:8000/docs

📂 Repository Structure

MLzoomcamp_Midterm2025/
│
├── data/
│   └── cac40_features.csv
│
├── cac40_analysis.ipynb
├── train.ipynb
├── predict.ipynb
│
├── train.py
├── predict.py
├── api.py
│
├── Dockerfile
│
├── model.pkl
├── scaler.pkl
├── features.json
│
└── README.md

## Clone the Project (Windows 11 + WSL Recommended)

All commands should be run inside Ubuntu WSL, not Windows PowerShell.

✅ Open WSL (Ubuntu)

Search Windows → Ubuntu → open it.

✅ Clone the GitHub repository
cd ~
git clone https://github.com/YOUR_USERNAME/MLzoomcamp_Midterm2025.git
cd MLzoomcamp_Midterm2025

## Run Locally (Without Docker)
✔ Create a virtual environment
python3 -m venv venv
source venv/bin/activate

✔ Install dependencies
pip install -r requirements.txt

✔ Start the FastAPI service
python api.py


Your API runs at:

👉 http://127.0.0.1:8000/docs

(Interactive Swagger documentation)

## Run Using Docker (Recommended)
✔ Build the Docker image
docker build -t ml-api .

✔ Run the container
docker run -d -p 8000:8000 ml-api

✔ Access the API

👉 http://localhost:8000/docs




API is available at:

👉 http://127.0.0.1:8000/docs

✔ Docker containerization

