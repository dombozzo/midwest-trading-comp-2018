'''
File:
	my_portfolio.py

Description:
	The primary purpose of this file is to extend the PortfolioGenerator
	class defined in portfolio.py for our purposes.
'''

import pandas as pd
import numpy as np
from portfolio import PortfolioGenerator

class Generator(PortfolioGenerator):
	
	def __init__(self):
		pass

	# for now this just returns random weights
	def build_signal(self, stock_features):
		ticker_vals = stock_features['ticker'].unique()
		weights = 200*np.random.rand(ticker_vals.shape[0])-100
		output = pd.Series(weights, index=ticker_vals)
		return output
		

#main for testing
if __name__=='__main__':
	gen = Generator()
	stock_df = gen.read_stock_data()
	'''
	ticker_vals = stock_df['ticker'].unique()
	weights = 200*np.random.rand(ticker_vals.shape[0])-100
	output = pd.Series(weights, index=ticker_vals)
	print output
	'''
	gen.build_signal(stock_df)
	print gen.simulate_portfolio()
