# src/data.py

import pandas as pd
import yfinance as yf
import pandas_datareader as web
import datetime

def get_date(start_date, end_date = None):
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

def fetch_ff3_data(start_date, end_date = None):
    '''
    Fetching FF3 Factor data for a given timeframe using Ken French API as well as the get_date function
    Output will be a timeseries dataframe with relative monthly returns for ff3 data
    '''
    # Retrieves appropriate dates using get_date function
    start_date, end_date = get_date(start_date, end_date)

    # fetching ff3 data using web.DataReader

    ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=start_date, end=end_date)

    # Cleaning up dictionary and turning it into a Dataframe
    del(ff_data["DESCR"], ff_data[1])
    ff_data = pd.DataFrame(ff_data[0]) / 100

    return(ff_data)


def get_portfolio(ticker_dict, start_date, end_date=None):
    '''
    UPDATED: Returns INDIVIDUAL ASSETS instead of portfolios.
    1. Downloads ALL tickers at once.
    2. Resamples to Monthly Returns.
    3. Returns the dataframe of individual stock returns.
    '''
    start_date, end_date = get_date(start_date, end_date)

    # Flatten the dictionary of lists into a single list of tickers
    all_tickers = []
    if isinstance(ticker_dict, dict):
        for sector in ticker_dict:
            all_tickers.extend(ticker_dict[sector])
    elif isinstance(ticker_dict, list):
        all_tickers = ticker_dict

    # Deduplicate list
    all_tickers = list(set(all_tickers))

    print(f"Downloading data for {len(all_tickers)} tickers...")

    # auto_adjust=True ensures we get dividends/splits included (Total Return)
    raw_data = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

    # Resample to Monthly End ('ME') and get the last price
    monthly_close = raw_data.resample("ME").agg("last").to_period("M")

    # Calculate returns
    monthly_returns = monthly_close.pct_change()

    # Remove the first row (NaN) and columns that are entirely NaN (failed downloads)
    monthly_returns = monthly_returns.dropna(how='all').dropna(axis=1, how='all')

    return monthly_returns
