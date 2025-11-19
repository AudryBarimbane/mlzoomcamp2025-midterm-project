#!/usr/bin/env python
# coding: utf-8

# In[15]:


# 1.Import required libraries
import pandas as pd
import numpy as np
import joblib

# For warnings
import warnings
warnings.filterwarnings("ignore")



# In[17]:


# 2.Load the trained model
model = joblib.load("model.pkl")

# Load the scaler (used during training)
scaler = joblib.load("scaler.pkl")

print("Model and scaler loaded successfully.")


# In[19]:


# 3.Features that must be provided for prediction
features = [
    "symbol", "date", "open", "high", "low", "close", "volume",
    "adjclose", "Return", "MA20", "MA50", "Volatility", "RSI"
]



# In[21]:


# 4.Example input (you can modify these values)
sample = {
    "symbol": "ACA",
    "date": "2024-01-15",
    "open": 12.45,
    "high": 12.60,
    "low": 12.40,
    "close": 12.55,
    "volume": 1520000,
    "adjclose": 12.55,
    "Return": 0.004,
    "MA20": 12.30,
    "MA50": 12.10,
    "Volatility": 0.012,
    "RSI": 58.3
}

# Convert to DataFrame with correct column order
sample_df = pd.DataFrame([sample])[features]

sample_df


# In[29]:


#5. Remove non-numeric columns before prediction
X = sample_df_scaled.drop(columns=["symbol", "date"])

#6. Make prediction
prediction = model.predict(X)[0]

print("Predicted Target value:", prediction)



# In[ ]:





# In[ ]:




