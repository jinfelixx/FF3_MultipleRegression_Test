# main.py

import setuptools
import src.config as config
import src.data as data
import src.models as models
import src.analysis as analysis
import pandas as pd


def main():
    print("--- 1. Starting Data Ingestion ---")

    # 1. Fetch FF3 Factor
    print(f"Fetching Fama-French data from {config.start_date}...")
    ff3_df = data.fetch_ff3_data(config.start_date, config.end_date)
    print(f"  > FF3 Data retrieved. Shape: {ff3_df.shape}")

    # 2. Fetch Portfolio Returns
    print("Fetching Portfolio data (this may take a moment)...")
    port_df = data.get_portfolio(config.ticker_dict, config.start_date, config.end_date)
    port_df = port_df.dropna()
    print(f"  > Portfolio Data retrieved. Shape: {port_df.shape}")

    print("\n--- 2. Data Preparation & Alignment ---")

    # 3. Align and Prepare
    # This function joins them. If dates don't match, the result will be empty.
    portfolio_excess, risk_factors = models.timeseries_prep(port_df, ff3_df)

    print(f"  > Alignment Complete.")
    print(f"  > Excess Returns Shape: {portfolio_excess.shape}")
    print(f"  > Risk Factors Shape: {risk_factors.shape}")

    print("\n--- 3. Running Regressions ---")

    # 4. Time Series Regression (Get Betas)
    print("Running Time-Series Regression...")
    factor_loadings, jensen_alpha = models.timeseries(portfolio_excess, risk_factors)
    print("  > Betas (Factor Loadings) computed.")


    # 5. Cross-Sectional Regression (Get Gammas)
    print("Running Cross-Sectional Regression...")
    gammas = models.crosssection(portfolio_excess, factor_loadings)
    print("  > Gammas (Risk Premiums) computed.")

    print("\n--- 4. Statistical Analysis ---")

    # 6. Test Intercept (Alpha)
    t_stat, p_val = analysis.intercept_testing(gammas)
    print(f"Intercept (Alpha) Significance:")
    print(f"  > T-Statistic: {t_stat:.4f}")
    print(f"  > P-Value:     {p_val:.4f}")

    # 7. Test Factors
    f_tstat, f_pval = analysis.factor_significance_testing(gammas)
    print(f"Factor Significance (Average):")
    print(f"  > P-Values: \n{f_pval}")

    print("\n--- 5. Saving Results ---")

    # 8. Compare Estimated Factors vs Realized FF3 Factors
    print("\nRunning Factor Comparison (Estimated vs Realized)...")
    comp_tstat, comp_pval = analysis.factor_comparison(gammas, risk_factors)

    print(f"Factor Comparison P-Values (H0: Estimated == Realized):")
    print(f"{comp_pval}")

    # Interpreting the result
    if any(p < 0.05 for p in comp_pval):
        print("  > WARNING: Some estimated factors differ significantly from realized FF3 factors.")
    else:
        print("  > SUCCESS: Estimated factors match realized FF3 factors (Model Validity Supported).")

    print("\n--- 5. Saving Results ---")

    import os
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    save_path = os.path.join(config.RESULTS_DIR, 'final_gammas.csv')
    gammas.to_csv(save_path)
    print(f"Results saved to: {save_path}")
    print("--- Pipeline Finished Successfully ---")


if __name__ == "__main__":
    main()
