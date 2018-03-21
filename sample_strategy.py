'''
File:
    sample_strategy.py

Description:
    A sample implementation of a portfolio construction strategy
    that would qualify as a submission to the contest.

Questions:
    Please contact jigar@uchicago.edu or vtalasani@uchicago.edu
'''

from portfolio import PortfolioGenerator
import pandas as pd
import numpy as np

class SampleStrategy(PortfolioGenerator):

    def __init__(self):
        pass

    def build_signal(self, stock_features):
        return self.momentum(stock_features)

    def momentum(self, stock_features):
        return stock_features.groupby(['ticker'])['returns'].mean()

# Test out performance by running 'python sample_strategy.py'
if __name__ == "__main__":
    portfolio = SampleStrategy()
    sharpe = portfolio.simulate_portfolio()
    print("*** Strategy Sharpe is {} ***".format(sharpe))

