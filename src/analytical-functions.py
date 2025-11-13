# importing necessary libraries:

import yfinance as yf
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import statsmodels.api as sm

# fetches per-month FF3 Factor data for specified timeframe using Ken French library
def fetch_ff3_data(start_date, end_date):
    ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=start_date, end=end_date)
    del(ff_data["DESCR"], ff_data[1])
    ff_data = pd.DataFrame(ff_data[0]) / 100
    return(ff_data)

# fetches monthly relative returns for specified timeframe using yfinance. Can be transformed into stock data loader by removing
# bottom code
def rel_returns(ticker_symbol, start_date, end_date):
    resample_logic = {"Open" : "first",
                      "High" : "max",
                      "Low" : "min",
                      "Close": "last",
                      "Volume" : "sum",
                     }
    data = yf.Ticker(ticker_symbol).history(start = start_date, end = end_date)
    monthly_data = data.resample("ME").agg(resample_logic)
    monthly_returns = monthly_data["Close"].pct_change().dropna()
    monthly_returns.columns = "Relative Returns"
    return(monthly_returns)

# requires array of stock we want to test to compute excess returns
def excess_returns(arr):
    arr.index = ff3.index
    Y = arr - ff3.iloc[:,3]
    return Y

# creates dataframe for regressors from FF3 data
def reg_X():
    X = ff3.iloc[:,0:3]
    X = sm.add_constant(X)
    return X

# Creates multiple regression for Stock and regressors; also adding constant for intercept
def OLS_Regression(Stock, X):
    model = sm.OLS(Stock, X)
    results = model.fit()
    return results.summary()

# Function for plotting Data; may be used for stock or FF3

def stock_plot(arr, title, labelx, labely, grid_stat=True):
    if not isinstance(title, str):
        raise TypeError(f"Expected String input, instead received {type(title)}")
    if not isinstance(labelx, str):
        raise TypeError(f"Expected String input, instead received {type(labelx)}")
    if not isinstance(labely, str):
        raise TypeError(f"Expected String input, instead received {type(labely)}")
    if not isinstance(grid_stat, bool):
        raise TypeError(f"Expected True or False, instead received {type(grid_stat)}")

    x_axis = np.linspace(start=arr.index.year[1], stop=arr.index.year[-1], num=arr.index.year.size)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, arr)
    plt.title(title)
    plt.ylabel(labely)
    plt.xlabel(labelx)
    plt.grid(grid_stat)
    plt.show()

