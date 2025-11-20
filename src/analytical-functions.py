# importing necessary libraries:

import yfinance as yf
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime

def get_date(start_date, end_date=None):
    '''
    fetches dates for ff3 and stock data.
    deploy this function to reduce complexity/potential misalignment of data
    Formatting for dates: "YYYY-MM-DD"
    '''
    if end_date == None:
        end_date = datetime.datetime.now()
    else:
        end_date = pd.to_datetime(end_date)

    start_date = pd.to_datetime(start_date)

    return start_date, end_date

def fetch_ff3_data(start_date, end_date="None"):
    '''
    Fetching FF3 Factor data for a given timeframe using Ken French API
    '''
    start_date, end_date = get_date(start_date, end_date)
    ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=start_date, end=end_date)
    del(ff_data["DESCR"], ff_data[1])
    ff_data = pd.DataFrame(ff_data[0]) / 100
    return(ff_data)


def fetch_stock(stock_ticker, start_date, end_date="None"):
    '''
    fetching daily stock returns, which are then converted to monthly stock data in order to match ff3 data
    '''
    start_date, end_date = get_date(start_date, end_date)
    resample_logic = {"Close": "last"}

    daily = yf.Ticker(stock_ticker).history(start=start_date, end=end_date)

    monthly = daily.resample("M").agg(resample_logic).to_period("M")

    monthly_returns = monthly.pct_change().dropna()
    return monthly_returns

def prep_stocks(ticker_list, start_date, end_date):
    '''
    Function that creates dataframe that summarises the returns for a list of provided stock tickers
    '''
    new_df = pd.concat((fetch_stock(i, start_date, end_date) for i in ticker_list), axis = 1)
    new_df.columns = ticker_list
    return new_df


def reg_prep(stock_df, ff3_data):
    '''
    Creating a function that cleans and prepares our stock data for regression
    '''
    joined_df = stock_df.join(ff3_data, how="inner")

    stock_cols = stock_df.columns
    ff3_cols = ff3_data.columns[:-1]

    joined_df[stock_cols] = joined_df[stock_cols].sub(joined_df["RF"], axis=0)

    Y = joined_df[stock_cols]
    X = sm.add_constant(joined_df[ff3_cols])
    return X, Y

def regress(X, Y):
    '''
    Creates dictionary that stores the regression summaries for every stock contained in Returns Y
    Prints Summary statistics for regression.
    '''
    regression_data = {}
    for i in Y.columns:
        model = sm.OLS(Y[i], X)
        results = model.fit()
        regression_data[i] = results
    [print(regression_data[j].summary()) for j in regression_data]
    return regression_data


def stock_plot(Y, title="Stock Alpha", xlabel="Date", ylabel="Returns", grid=True):
    '''
    plots any dataframe of stocks
    x-axis default: date (PeriodIndex: "YYYY-MM"
    y-axis default: Stock Alpha (in percent)
    '''

    if not isinstance(title, str):
        raise TypeError(f"Expected String input, instead received {type(title)}")
    if not isinstance(xlabel, str):
        raise TypeError(f"Expected String input, instead received {type(xlabel)}")
    if not isinstance(ylabel, str):
        raise TypeError(f"Expected String input, instead received {type(ylabel)}")
    if not isinstance(grid, bool):
        raise TypeError(f"Expected True or False, instead received {type(grid)}")

    # plt.figure(figsize=(10, 6)) - not necessary
    Y.plot(figsize=(10,6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    plt.show()


#test execution block:

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

start, end = get_date("2020-12-31")
ff3 = fetch_ff3_data(start, end)
stock_data = prep_stocks(tickers, start, end) # if fetch stocks, theres an error I think if used a list
X, Y = reg_prep(stock_data, ff3)
regress(X=X, Y=Y)
stock_plot(Y)
