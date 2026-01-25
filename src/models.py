# src/models.py

import statsmodels.api as sm
import pandas as pd
import numpy as np

def timeseries_prep(portfolio_df, ff3_data):
    '''
        Prepares stock data for regression:
        1. Joins portfolio data with FF3 data (aligning dates).
        2. Subtracts Risk Free Rate (RF) from portfolio returns to get Excess Returns.
        3. Adds Constant to FF3 factors for the regression intercept.
    '''

    # join both dataframes together
    joined_df = portfolio_df.join(ff3_data, how="inner")

    portfolio_cols = portfolio_df.columns
    ff3_cols = ff3_data.columns[:-1]

    joined_df[portfolio_cols] = joined_df[portfolio_cols].sub(joined_df["RF"], axis=0)

    portfolio_excess_returns = joined_df[portfolio_cols]

    risk_factors = sm.add_constant(joined_df[ff3_cols])

    return portfolio_excess_returns, risk_factors


def timeseries(portfolio_excess_returns, risk_factors):
    '''
        Time series regression to compute factor loadings (betas).
    '''


    model = sm.OLS(portfolio_excess_returns, risk_factors)
    results = model.fit()

    # Transpose to get factors as columns
    factor_loadings_t = results.params
    factor_loadings_t.columns = portfolio_excess_returns.columns

    factor_loadings_total = factor_loadings_t.T

    # Rename Columns
    new_columns = list(factor_loadings_total.columns)
    new_columns[0] = "jensen_alpha"
    factor_loadings_total.columns = new_columns

    # Separate jensen alpha from betas
    factor_loadings = factor_loadings_total.iloc[:,1:]
    jensen_alpha = factor_loadings_total.iloc[:,0]

    return factor_loadings, jensen_alpha


def crosssection(portfolio_excess_returns, factor_loadings):
    '''
        Cross-sectional regression (Fama-MacBeth 2nd pass) using Manual Matrix Algebra.
    '''

    betas = sm.add_constant(factor_loadings)

    cols = betas.columns

    projection_matrix = np.linalg.inv(betas.T @ betas) @ betas.T
    gammas = projection_matrix @ portfolio_excess_returns.T

    gammas.index = cols
    gammas = gammas.T

    gammas = pd.DataFrame(gammas)
    return gammas