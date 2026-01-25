# importing requirements

import setuptools
import yfinance as yf
import numpy as np
import pandas as pd
import pandas_datareader as web
import statsmodels.api as sm
import datetime
import scipy.stats as stats

def get_date(start_date, end_date = None):
    '''
    fetches dates for ff3 and stock data.
    deploy this function to reduce complexity/potential misalignment of data
    Formatting for dates: "YYYY-MM-DD"

    if end date is empty, end date will be today
    provides start and end date
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
    start_date, end_date = get_date(start_date, end_date)
    ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=start_date, end=end_date)
    del(ff_data["DESCR"], ff_data[1])
    ff_data = pd.DataFrame(ff_data[0]) / 100
    return(ff_data)


# Efficiency on the download may be improved at the for loop

def get_portfolio(ticker_list, start_date, end_date=None):
    '''
    Creates a Dataframe that contains monthly returns for all portfolios
    Input data should be a dict of the following form:

    {
    "Portfolio i" : [Tickers],
    "Portfolio i+1" : [Tickers]
    }
    1. takes start and end date via get_date
    2. creates empty portfolio dictionary
    3. For loop
        a. download every portfolio in specified date range
        b. create a series with only the closing prices
        c. resample closing prices to monthly returns, further changing the DateTimeIndex to PeriodIndex
        d. compute monthly returns, dropping all NA
        e. appending portfolio average returns to portfolio_dict
    4. returns Portfolio Dataframe
    '''
    start_date, end_date = get_date(start_date, end_date)

    portfolio_dict = {}

    for x in ticker_list.keys():
        daily = yf.download(ticker_list[x], start=start_date, end=end_date)

        close = daily["Close"]

        monthly_close = close.resample("ME").agg("last").to_period("M")

        monthly_returns = monthly_close.pct_change().dropna()

        portfolio_dict[x] = monthly_returns.mean(axis=1)

    return pd.DataFrame(portfolio_dict)


def timeseries_prep(portfolio_df, ff3_data):
    '''
    Creating a function that cleans and prepares our stock data for regression

    1. Create joined dataframe that connects portfolio data with ff3 data using pd.join(how = "inner") to align time series data
    2. Save column names to separate later on
    3. Risk free returns are subtracted from portfolio returns
    4. Add Constant to FF3-Regressor Matrix
    5. Joined DF is now again separated into regression ready X and Y variables
    '''
    joined_df = portfolio_df.join(ff3_data, how="inner")

    portfolio_cols = portfolio_df.columns
    ff3_cols = ff3_data.columns[:-1]

    joined_df[portfolio_cols] = joined_df[portfolio_cols].sub(joined_df["RF"], axis=0)

    portfolio_excess_returns = joined_df[portfolio_cols]
    risk_factors = sm.add_constant(joined_df[ff3_cols])
    return portfolio_excess_returns, risk_factors


def timeseries(portfolio_excess_returns, risk_factors):
    '''
    time series regression to compute the factor loadings
    stores factor_loadings which is essentially the beta matrix that is needed for the cross sectional regression
    jensen alpha also stored

    1. store model parameters
    2. save column names
    3. transpose parameter matrix, renaming the columns
    4. extract factor_loadings and jensen_alpha

    '''
    model = sm.OLS(portfolio_excess_returns, risk_factors)
    results = model.fit()

    factor_loadings_t = results.params
    factor_loadings_t.columns = portfolio_excess_returns.columns

    factor_loadings_total = factor_loadings_t.T
    new_columns = list(factor_loadings_total.columns)
    new_columns[0] = "jensen_alpha"
    factor_loadings_total.columns = new_columns

    factor_loadings = factor_loadings_total.iloc[:,1:]
    jensen_alpha = factor_loadings_total.iloc[:,0]

    return factor_loadings, jensen_alpha


def crosssection(portfolio_excess_returns, factor_loadings):
    '''
    prepare the betas for regression; Since OLS takes the Form BETA = (X'X)^-1 X' and betas stay constant in this regression,
    we will use this attribute to compute the regression instead.


    '''
    betas = sm.add_constant(factor_loadings)

    cols = betas.columns
    projection_matrix = np.linalg.inv(betas.T @ betas) @ betas.T
    gammas = projection_matrix @ portfolio_excess_returns.T

    gammas.index = cols
    gammas = gammas.T
    return gammas

def intercept_testing(gammas):
    '''
    To test, whether the intercept is significantly 0

    1. Compute Standard Error
    2. Compute Mean of Intercept
    3. T-Test
    4. P-Value
    '''
    intercept = gammas.iloc[:,0]


    intercept_tstat, intercept_pvalue = stats.ttest_1samp(intercept,0)

    return intercept_tstat, intercept_pvalue

def factor_significance_testing(gammas):
    '''
    Computes significance of factors.
    '''

    # Testing for significance of Gammas
    factors = gammas.iloc[:,1:]

    factor_tstat, factor_pvalue = stats.ttest_1samp(factors, 0)

    return factor_tstat, factor_pvalue

def factor_comparison(gammas, risk_factors):
    '''
    Compares estimated factor gammas to average ff3 returns.
    If FF3 is valid, they should NOT be significantly different from each other
    '''

    factors = gammas.iloc[:,1:]
    # note, we assume that ff3 array already omitted RF; might need to run a double check later on

    factor_comparison_ttest, factor_comparison_pvalue = stats.ttest_rel(factors, risk_factors)

    return factor_comparison_ttest, factor_comparison_pvalue