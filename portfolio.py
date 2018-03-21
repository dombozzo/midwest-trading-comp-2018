'''
File:
    portfolio.py

Description:
    Extend this interface with a class that provides for a portfolio
    allocation strategy. See sample_strategy.py for an example. Your
    strategy will be tested with a hidden test dataset that will be
    included as 'ticker_data.csv' and 'stock_data.csv 'in the same
    stock_data/ directory included in the repository. It is highly
    recommended that you study the documentation of the pandas libary.

Notes:
    - Do NOT make any modifications to this file; when testing your
    implementation, we will use the original version of this file if any
    modifications were made.
    - You are required to implement (override) the build_signal function
    after extending this interface. Attempting to override any other method
    in this class will lead to immediate disqualification from the round.

Questions:
    Please contact jigar@uchicago.edu or vtalasani@uchicago.edu
'''

import pandas as pd
import numpy as np

MAX_LOOKBACK = 50

class PortfolioGenerator(object):

    def __init__(self):
        pass

    def read_stock_data(self):
        '''
        Description:
            Reads in simulated stock data from stock_data.csv
        Returns:
            stock_df (DataFrame): standardized ticker/factor data in pandas df
        Raises:
            AssertionError: ticker_data.csv/factor_data.csv has an invalid format
        '''
        ticker_df = pd.read_csv('stock_data/ticker_data.csv')
        factor_df = pd.read_csv('stock_data/factor_data.csv')
        assert 'timestep' in ticker_df.columns, "ticker_data.csv has an invalid format"
        assert 'ticker' in ticker_df.columns, "ticker_data.csv has an invalid format"
        assert 'returns' in ticker_df.columns, "ticker_data.csv has an invalid format"
        assert 'timestep' in factor_df.columns, "factor_data.csv has an invalid format"
        ticker_df.set_index('timestep', inplace=True)
        factor_df.set_index('timestep', inplace=True)
        stock_df = ticker_df.join(factor_df, how='left')
        return stock_df

    def build_signal(self, stock_features):
        '''
        Description:
            Using stock features, generate a buy/sell stock
            signal (suggested from -100 to 100 where -100 is strong
            sell signal) for the stock on a given day.
        Args:
            hist_features (DataFrame) - dataframe of historical features
        Return:
            signal (pandas Series) - key = ticker, value = signal
        Raises:
            NotImplementedError - throws error if the function is not implemented
        '''
        raise NotImplementedError("build_signal must be implemented")

    def simulate_portfolio(self):
        '''
        Description:
            Simulates performance of the portfolio on historical data
        Return:
            sharpe (int) - sharpe ratio for the portfolio
        '''
        daily_returns = []
        stock_df = self.read_stock_data()
        for idx in stock_df.index.unique():
            print("timestep", idx)
            if idx < MAX_LOOKBACK:
                continue
            stock_features = stock_df.loc[idx-MAX_LOOKBACK:idx-1]
            returns = stock_df.loc[idx:idx].set_index('ticker')['returns']
            signal = self.build_signal(stock_features)
            signal_return = returns * signal
            daily_returns.append(np.mean(signal_return))
        sharpe_ratio = np.sqrt(252) * (np.mean(daily_returns) / np.std(daily_returns))
        return sharpe_ratio

