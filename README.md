# CAC40 Market Movement Prediction

Midterm CAPSTONE project for **MLZoomcamp 2025**

---

## 📌 What is this?

This repository contains all components required to meet the MLZoomcamp 2025 midterm project requirements. It includes:

* Dataset
* Jupyter notebooks for EDA, feature creation, model training, predictions
* Python scripts (`train.py`, `predict.py`)
* Serialized model files (`model.pkl`, `scaler.pkl`)

The project focuses on applying machine learning to **financial market data** from CAC40 companies.

---

## 🎯 What is the problem?

This is a **regression problem** aimed at predicting a market movement indicator (`Target`) for stocks in the French CAC40 index.

The problem involves:

* Loading cleaned daily market data (OHLCV + technical indicators)
* Engineering features such as returns, moving averages, volatility, RSI
* Training several models (Decision Tree, Random Forest, XGBoost)
* Saving the best model and its scaler
* Providing a prediction interface via notebook and Python script

This enables use cases such as:

* Short-term market signal prediction
* Automated trading decision support
* Evaluating model performance on financial time‑series

---

## 📂 Repository Structure

```
MLzoomcamp_Midterm2025/
│
├── data/
│   └── cac40_features.csv
│
├── cac40_analysis.ipynb
├── train.ipynb
├── predict.ipynb
├── train.py
├── predict.py
├── features.json
├── model.pkl
├── scaler.pkl
└── README.md
```

Note: `features.json`, `model.pkl`, and `scaler.pkl` are located **at the repository root**, alongside the notebooks.

---

## 🧠 Scripts Overview

### **1. `train.ipynb`**

* Full Exploratory Data Analysis
* Feature verification
* Train/test split
* Model training (Logistic regression, Random Forest, XGBoost)
* Model comparison
* Saving `model.pkl`, `scaler.pkl`, `features.json`

### **2. `predict.ipynb`**

* Loads model + scaler
* Creates manual samples for testing
* Predicts market `Target` values
* Tests prediction on last row of dataset
* Contains reusable prediction function (future FastAPI integration)

### **3. `train.py`**

Standalone Python script to:

* Load dataset
* Train final model
* Export model + scaler + features

### **4. `predict.py`**

Standalone Python script to:

* Load saved model
* Prepare a sample dictionary
* Scale and reorder features
* Make a prediction

(This can later be wrapped into a FastAPI endpoint.)

---

## 📦 Dataset

Source: **Custom dataset of CAC40 stocks**, containing:

* 14 features per day
* OHLCV data + engineered indicators
* 1 binary target (`Target`)

### Columns:

```
symbol, date, open, high, low, close, volume,
adjclose, Return, MA20, MA50, Volatility, RSI, Target
```

All values are numerical (dates handled as strings).

---

## ⚙️ Model Selection

The following models were evaluated:

* Logistic Regression (baseline)
* Random Forest
* XGBoost (best performance)

The final exported model is an **XGBoost classifier**.

Metrics used:

* Accuracy
* F1-score
* Confusion matrix

---

## ▶️ How to Install and Run

### **Option 1 — Local execution (Python 3.10+)**

#### 1. Clone the repository

```
git clone https://github.com/YourUsername/MLzoomcamp_Midterm2025.git
cd MLzoomcamp_Midterm2025
```

#### 2. Create virtual environment

```
python -m venv venv
source venv/bin/activate     # Linux & MacOS
venv\Scripts\activate        # Windows
```

#### 3. Install dependencies

```
pip install -r requirements.txt
```

#### 4. Train the model (optional — pre-trained files included)

```
python train.py
```

#### 5. Run prediction locally

```
python predict.py
```

You will see output like:

```
Predicted Target value: 1
```

---

## 🧪 What exactly is going on?

When running `predict.py` or `predict.ipynb`, the script:

1. Loads the saved XGBoost model
2. Loads the saved scaler
3. Loads the correct feature order from `features.json`
4. Creates a dictionary of input features
5. Converts it into a DataFrame
6. Scales the features
7. Makes a prediction (`0` or `1`)

You can change input values to test different market scenarios and see how predictions evolve.

---

