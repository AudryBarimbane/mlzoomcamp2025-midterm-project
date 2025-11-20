# 📘 CAC40 Market Movement Prediction  
**Midterm CAPSTONE Project – MLZoomcamp 2025**

---

## 📌 Overview

This project applies Machine Learning to financial market data from CAC40 companies to predict **short-term market movement**.  
It follows the complete MLZoomcamp pipeline:

1. Pick a problem & dataset  
2. Describe how ML helps  
3. Prepare data & run EDA  
4. Train several models & select the best  
5. Export the trained model  
6. Package the model as a FastAPI service  
7. Deploy the service with Docker  

This repository includes:

- Dataset  
- Jupyter Notebooks (EDA, model training, predictions)  
- Python scripts (`train.py`, `predict.py`, `api.py`)  
- Trained model (`model.pkl`), scaler (`scaler.pkl`), and feature order (`features.json`)  
- Dockerfile  

---

# 🎯 1. Problem Definition

Financial markets move quickly and depend on many technical and price-based indicators.  
The goal is to build a machine learning model that predicts **the probability of an upward market movement (“Target”)** for CAC40 stocks.

Originally the project used binary output (0 or 1), but the final API returns a **probability between 0 and 1**, which reflects:

👉 the model’s confidence instead of a hard class.  
👉 better usability for trading systems (thresholds can be customized).  

---

# 📚 2. Dataset

Daily market data for CAC40 companies, including both OHLCV features and engineered technical indicators.

**Columns:**

symbol, date, open, high, low, close, volume,
adjclose, Return, MA20, MA50, Volatility, RSI, Target


- All numeric except `symbol` and `date`.  
- Stored in: `data/cac40_features.csv`.

---

# 🔎 3. Exploratory Data Analysis (EDA)

Performed in `cac40_analysis.ipynb`:

- Data inspection  
- Missing values  
- Feature distributions  
- Correlation heatmap  
- Technical indicator behavior  
- Target distribution  

This ensures dataset consistency before training.

---

# 🧠 4. Model Training

Three models were evaluated:

- **Logistic Regression**
- **Random Forest**
- **XGBoost (best performance)**

XGBoost delivered the highest F1-score and the best calibration for probability prediction.

**Metrics used:**

- Accuracy  
- Precision/Recall  
- F1-score  
- Confusion Matrix  

The training workflow appears in:

- `train.ipynb`
- `train.py`

Artifacts exported:


model.pkl
scaler.pkl
features.json


---

# 🔮 5. Prediction Pipeline

Available in:

- `predict.ipynb`  
- `predict.py`

The prediction steps are:

1. Load `model.pkl` (XGBoost)  
2. Load `scaler.pkl`  
3. Reorder input features according to `features.json`  
4. Scale numerical data  
5. Predict **a probability between 0 and 1**  
   - Example output: `{ "prediction": 0.63 }`  

This probability means:

- 0.63 → 63% chance of upward movement  
- Users may apply their own threshold (e.g., 0.5 or 0.6)

---

# 🚀 6. FastAPI Web Service

`api.py` provides a fully operational REST API.

### **Endpoints**

#### `GET /`
Returns a simple health check.

#### `POST /predict`
Accepts JSON input and returns the predicted probability.

**Example input:**

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



**Access the automatic API documentation:**
👉 http://localhost:8000/docs
