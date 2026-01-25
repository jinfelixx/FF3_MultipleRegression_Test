# src/analysis.py

from scipy import stats
import pandas as pd

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

    risk_factors = risk_factors.iloc[:,1:]

    factor_comparison_ttest, factor_comparison_pvalue = stats.ttest_rel(factors, risk_factors)

    return factor_comparison_ttest, factor_comparison_pvalue