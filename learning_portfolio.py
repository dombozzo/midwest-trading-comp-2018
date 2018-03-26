'''
File:
	learning_portfolio.py

Description:
	The primary purpose of this file is to attempt to create a neural
	network that learns the weights to be used.
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from portfolio import PortfolioGenerator

# Constants
MAX_LOOKBACK = 50
Const = tf.placeholder(tf.float32, [1])

class Generator():
	
	def __init__(self):
		pass
	
	# copied directly from PortfolioGenerator class
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

	
	#gets ticker values for small cap firms
	def get_small_cap_inds(self, stock_features):
		curr_cap = stock_features[['ticker','market_cap']].tail(1000)
		median = curr_cap['market_cap'].median()
		lower_half = curr_cap[curr_cap['market_cap'] < median]['ticker']
		return lower_half.values

	#gets percent change signal for oil prices
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
	
	#TODO
	# def vix_signal(self, stock_features):

	# second vix signal
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
	
	#TODO - need something better
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
	
	#TODO -- need something better
	#signal for 3 monthly t-bill rates - inversely related to market movements
	def get_3mr_signal(self, stock_features):
		last2 = stock_features[['ticker','industry','3M_R']].tail(2000)
		yest = last2.head(1000).set_index('ticker')
		today = last2.tail(1000).set_index('ticker')
		inv_pct_chg = today[['industry']].join((yest['3M_R'] - today['3M_R'])/yest['3M_R'])
		return inv_pct_chg['3M_R']


	# this is the function that actually builds the final signals based on the
	# all the individual signal functions above
	def build_signal(self, stock_features, W):
		
		x = self.get_3mr_signal(stock_features)
		
		#provide boost to small-cap firms
		small_inds = self.get_small_cap_inds(stock_features)
		small_boost = np.zeros(1000)
		small_boost[small_inds] += 5

		vix_2 = self.vix2_signal(stock_features)
		small_ix = self.ix_signal(stock_features, 'SMALL_IX', [1.4, .74, .8, 1.1]) 
		big_ix = self.ix_signal(stock_features, 'BIG_IX', [1.1, .74, .8, 1.1]) 
		oil = self.oil_signal(stock_features)

		# multiply all this by tensorflow weights
		result = W[0] * small_ix.values + W[1] * big_ix.values + W[2] * oil.values + W[3] * small_boost

		return result

	
	# copied directly from PortfolioGenerator Class
	def simulate_portfolio(self, W):
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
			if idx > 100:
				break
			stock_features = stock_df.loc[idx-MAX_LOOKBACK:idx-1]
			returns = stock_df.loc[idx:idx].set_index('ticker')['returns']
			signal = self.build_signal(stock_features, W)
			signal_return = returns.values * signal
			daily_returns.append(tf.reduce_mean(signal_return))
			#daily_returns.append(np.mean(signal_return))
		
		m = tf.reduce_mean(daily_returns, axis=None, keep_dims=True)
		devs_squared = tf.square(daily_returns - m)
		var = tf.reduce_mean(devs_squared, axis=None, keep_dims=True)
		std = tf.sqrt(var)

		sharpe_ratio = np.sqrt(252) * (m / std)
		print sharpe_ratio
		#sharpe_ratio = np.sqrt(252) * (np.mean(daily_returns) / np.std(daily_returns))
		return sharpe_ratio * Const
	

## Constants and Placeholders
NumFeaturesUsed = 4
NumStocks = 1000
Input = tf.placeholder(tf.float32,[NumFeaturesUsed, NumStocks])

# create tensorflow graph -- currently just one matrix multiplication
def MakeNetwork(Input):
	
	W = tf.get_variable('W', [NumStocks,NumStocks])
	FC = tf.matmul(Input, W)
	return FC

#construct model
#Weights = MakeNetwork(Input)
Weights = tf.get_variable('W', [NumFeaturesUsed, NumStocks])

# define loss
with tf.name_scope('loss'):
	gen = Generator()
	Loss = 4 - tf.reduce_mean(gen.simulate_portfolio(Weights))
print Loss

# define optimizer
with tf.name_scope('optimizer'):
	Optimizer = tf.train.AdamOptimizer(.01).minimize(Loss)

# initialize variables
Init = tf.global_variables_initializer()

# launch session with default graph
conf = tf.ConfigProto(allow_soft_placement=True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.2

with tf.Session(config=conf) as Sess:
	Sess.run(Init)
	Step = 1
	shape = [NumStocks, NumStocks, NumStocks, NumStocks]
	#W = tf.random_normal(shape, mean=0.0, stddev=1.0,dtype=tf.float32,seed=None,name=None)
	while Step < 5000:
		const = np.ones(1)
		_,L,W = Sess.run([Optimizer, Loss, Weights], feed_dict={Const:const})
		print Loss
		Step += 1
