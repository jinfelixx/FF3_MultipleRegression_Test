# src/config.py

# Portfolio

ticker_dict = {

    "Big_Growth": [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA',
        'NFLX', 'ADBE', 'CRM', 'AMD', 'QCOM', 'INTU', 'NOW', 'BKNG'
    ],

    "Big_Value": [
        'JPM', 'BAC', 'XOM', 'CVX', 'WFC', 'CSCO', 'VZ',
        'PFE', 'MRK', 'KO', 'PEP', 'WMT', 'BMY', 'GS', 'MS'
    ],

    "Small_Growth": [
        'PLUG', 'FCEL', 'RUN', 'ENPH', 'SEDG',
        'RGEN', 'EXAS', 'CHWY', 'DKNG', 'ROKU',
        'COIN', 'TDOC', 'TWLO', 'NET', 'ZS'
    ],

    "Small_Value": [
        'URBN','KSS','CLF', 'M', 'AAL', 'AA', 'KEY', 'ZION',
        'HBAN','IVZ', 'BEN', 'LUMN', 'WU', 'XRX', 'CNK'
    ]
}


# Date Settings
start_date = "2000-01-01"
end_date = None

# Data Source

ff3_data = 'F-F_Research_Data_Factors'

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'processed')