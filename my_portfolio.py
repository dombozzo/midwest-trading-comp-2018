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
	
	#def build_signal(self, stock_features):
		

#main for testing
if __name__=='__main__':
	gen = Generator()
	gen.read_stock_data()
