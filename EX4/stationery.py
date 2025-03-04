#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


# In[2]:


cad_jpy_df = pd.read_csv("cad_jpy.csv", index_col="Date", infer_datetime_format=True, parse_dates=True)


# In[3]:


plt.figure(figsize=(12, 6))
plt.plot(cad_jpy_df["Price"], label="CAD/JPY Price")
plt.title("CAD/JPY Exchange Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.show()


# In[4]:


# Check for stationarity using Rolling Mean & Standard Deviation
rolling_mean = cad_jpy_df["Price"].rolling(window=30).mean()
rolling_std = cad_jpy_df["Price"].rolling(window=30).std()

plt.figure(figsize=(12, 6))
plt.plot(cad_jpy_df["Price"], label="Original Price")
plt.plot(rolling_mean, label="Rolling Mean", color='red')
plt.plot(rolling_std, label="Rolling Std Dev", color='black')
plt.title("Rolling Mean & Standard Deviation")
plt.legend()
plt.show()


# In[5]:


# Augmented Dickey-Fuller Test
def adf_test(series):
    result = adfuller(series.dropna())
    labels = ['ADF Statistic', 'p-value', '# Lags Used', 'Number of Observations']
    for label, value in zip(labels, result[:4]):
        print(f"{label}: {value}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"{key}: {value}")
    if result[1] <= 0.05:
        print("\nData is stationary (reject H0).")
    else:
        print("\nData is non-stationary (fail to reject H0).")

print("ADF Test on Original Data:")
adf_test(cad_jpy_df["Price"])


# In[6]:


# If non-stationary, apply differencing
cad_jpy_df['Price_Diff'] = cad_jpy_df['Price'].diff()

# Re-check stationarity
plt.figure(figsize=(12, 6))
plt.plot(cad_jpy_df['Price_Diff'], label="Differenced Price")
plt.title("Differenced Time Series Data")
plt.xlabel("Date")
plt.ylabel("Differenced Exchange Rate")
plt.legend()
plt.show()

print("\nADF Test on Differenced Data:")
adf_test(cad_jpy_df['Price_Diff'])


# In[ ]:




