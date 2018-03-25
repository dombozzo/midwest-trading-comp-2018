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
	
	#gets ticker values for small cap firms
	def get_small_cap_inds(self, stock_features):
		curr_cap = stock_features[['ticker','market_cap']].tail(1000)
		median = curr_cap['market_cap'].median()
		lower_half = curr_cap[curr_cap['market_cap'] < median]['ticker']
		return lower_half.values


	def oil_signal(self, stock_features):
		last2 = stock_features[['ticker','industry','OIL']].tail(2000)
		yest = last2.head(1000).set_index('ticker')
		today = last2.tail(1000).set_index('ticker')
		pct_chg = today[['industry']].join((today.OIL - yest.OIL)/yest.OIL)

		#now mitigate changes based on industry
		tech = pct_chg[pct_chg['industry'] == 'TECH'].OIL * .2
		agriculture = pct_chg[pct_chg['industry'] == 'AGRICULTURE'].OIL * .35
		finance = pct_chg[pct_chg['industry'] == 'FINANCE'].OIL * .25
		consumer = pct_chg[pct_chg['industry'] == 'CONSUMER'].OIL * .3
		other = pct_chg[pct_chg['industry'] == 'OTHER'].OIL * .25

		result = pd.concat([tech, agriculture, finance, consumer, other])
		return result
	
	def vix2_signal(self, stock_features):
		last2 = stock_features[['ticker','industry','VIX']].tail(2000)
		yest = last2.head(1000).set_index('ticker')
		today = last2.tail(1000).set_index('ticker')
		#TODO - make quadratic
		pct_chg = today[['industry']].join(((today.VIX - yest.VIX)/yest.VIX) ** 3)

		#now mitigate changes based on industry
		tech = pct_chg[pct_chg['industry'] == 'TECH'].VIX * .4
		agriculture = pct_chg[pct_chg['industry'] == 'AGRICULTURE'].VIX * .25
		finance = pct_chg[pct_chg['industry'] == 'FINANCE'].VIX * .5
		consumer = pct_chg[pct_chg['industry'] == 'CONSUMER'].VIX * .24
		other = pct_chg[pct_chg['industry'] == 'OTHER'].VIX * .3

		result = pd.concat([tech, agriculture, finance, consumer, other])
		return result
	
	def temp_signal(self, stock_features):
		last25 = stock_features[['ticker','industry','TEMP']].tail(25000)
		today = last25.tail(1000).set_index('ticker')
		avg_temp = last25['TEMP'].mean()
		diff_today_avg = today.TEMP - avg_temp

		#threshold value of 50 to see whether an increase in temp is good or bad
		#note - we are using a cubic relationship
		if avg_temp > 50:
			diff_today_avg *= -1
		cubed_diff = today[['industry']].join(diff_today_avg ** 3)
		
		#now mitigate changes based on industry
		tech = cubed_diff[cubed_diff['industry'] == 'TECH'].TEMP * .1
		agriculture = cubed_diff[cubed_diff['industry'] == 'AGRICULTURE'].TEMP * .4
		finance = cubed_diff[cubed_diff['industry'] == 'FINANCE'].TEMP * .25
		consumer = cubed_diff[cubed_diff['industry'] == 'CONSUMER'].TEMP * .3
		other = cubed_diff[cubed_diff['industry'] == 'OTHER'].TEMP * .25
		
		result = pd.concat([tech, agriculture, finance, consumer, other])
		return result
	
	#TODO - include notion of market cap
	def ix_signal(self, stock_features, ix_type, ind_weights):
		last2 = stock_features[['ticker','industry',ix_type]].tail(2000)
		yest = last2.head(1000).set_index('ticker')
		today = last2.tail(1000).set_index('ticker')
		pct_chg = today[['industry']].join((today[ix_type] - yest[ix_type])/yest[ix_type])

		#now mitigate changes based on industry
		tech = pct_chg[pct_chg['industry'] == 'TECH'][ix_type] * ind_weights[0]
		agriculture = pct_chg[pct_chg['industry'] == 'AGRICULTURE'][ix_type] * ind_weights[1]
		finance = pct_chg[pct_chg['industry'] == 'FINANCE'][ix_type] * ind_weights[2]
		consumer = pct_chg[pct_chg['industry'] == 'CONSUMER'][ix_type] * ind_weights[3]
		other = pct_chg[pct_chg['industry'] == 'OTHER'][ix_type]

		result = pd.concat([tech, agriculture, finance, consumer, other])
		return result

	# for now this just returns random weights
	def build_signal(self, stock_features):
		small_inds = self.get_small_cap_inds(stock_features)
		small_boost = np.zeros(1000)
		small_boost[small_inds] += 5

		vix_2 = self.vix2_signal(stock_features)
		small_ix = self.ix_signal(stock_features, 'SMALL_IX', [1.4, .74, .8, 1.1]) 
		big_ix = self.ix_signal(stock_features, 'BIG_IX', [1.1, .74, .8, 1.1]) 
		oil = self.oil_signal(stock_features)
		return small_ix + big_ix + oil + small_boost
		temp =  .01*self.temp_signal(stock_features)
		return other
		

#main for testing
if __name__=='__main__':
	gen = Generator()
	print gen.simulate_portfolio()
