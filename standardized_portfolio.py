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
import collections

#TODO - remove -- global variable for stat collection
#stats = collections.defaultdict(list)
thresholds = {'senti': 3.87765524879, \
			  'temp': 652.887038887, \
			  'vix_2': 0.126085094522, \
			  '3mr': 0.0356048721667, \
			  'rain': 0.030983282445, \
			  'copp': 0.0577264226424, \
			  'oil': 0.119131372099, \
			  'vix': 0.0144128819314,
			  'mod_ix': 0.0644891866728 \
			  }


class Generator(PortfolioGenerator):
	
	def __init__(self):
		pass
	
	#get ticker values for high pb firms
	def get_high_pb_inds(self, stock_features):
		curr_cap = stock_features[['ticker','pb']].tail(1000)
		median = curr_cap['pb'].median()
		upper_half = curr_cap[curr_cap['pb'] > median]['ticker']
		return upper_half
		curr_cap = stock_features[['ticker','pb','industry']].tail(1000)

		#tech
		tech = curr_cap[curr_cap['industry'] == 'TECH']
		median = tech['pb'].median()
		upper_inds = tech[tech['pb'] > median]['ticker'].values

		#agriculture
		ag = curr_cap[curr_cap['industry'] == 'AGRICULTURE']
		median = ag['pb'].median()
		upper_inds = np.concatenate((upper_inds, ag[ag['pb'] > median]['ticker'].values))

		#finance
		fin = curr_cap[curr_cap['industry'] == 'FINANCE']
		median = fin['pb'].median()
		upper_inds = np.concatenate((upper_inds, fin[fin['pb'] > median]['ticker'].values))


		#consumer
		cons = curr_cap[curr_cap['industry'] == 'CONSUMER']
		median = cons['pb'].median()
		upper_inds = np.concatenate((upper_inds, cons[cons['pb'] > median]['ticker'].values))

		#other
		other = curr_cap[curr_cap['industry'] == 'OTHER']
		median = other['pb'].median()
		upper_inds = np.concatenate((upper_inds, other[other['pb'] > median]['ticker'].values))
		
		return upper_inds

	#gets ticker values for small cap firms
	def get_small_cap_inds(self, stock_features):
		curr_cap = stock_features[['ticker','market_cap']].tail(1000)
		median = curr_cap['market_cap'].median()
		lower_half = curr_cap[curr_cap['market_cap'] < median]['ticker']
		return lower_half.values
	
	#gets percent change signal for copper prices
	def copper_signal(self, stock_features):
		last2 = stock_features[['ticker','industry','COPP']].tail(2000)
		yest = last2.head(1000).set_index('ticker')
		today = last2.tail(1000).set_index('ticker')
		pct_chg = today[['industry']].join(-1*(today.COPP - yest.COPP)/yest.COPP)

		#now mitigate changes based on industry
		tech = pct_chg[pct_chg['industry'] == 'TECH'].COPP * .2
		agriculture = pct_chg[pct_chg['industry'] == 'AGRICULTURE'].COPP * .35
		finance = pct_chg[pct_chg['industry'] == 'FINANCE'].COPP * .25
		consumer = pct_chg[pct_chg['industry'] == 'CONSUMER'].COPP * .3
		other = pct_chg[pct_chg['industry'] == 'OTHER'].COPP * .25

		result = pd.concat([tech, agriculture, finance, consumer, other])
		return result

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
	
	#TODO - figure out why not working
	#mitigated linear - negative % deviation from average
	def vix_signal(self, stock_features):
		avg_vix = 15.5305463609
		today = stock_features[['ticker', 'industry', 'VIX']].tail(1000)
		today = today.set_index('ticker')
		diff_today_avg = today[['industry']].join(-1*(today.VIX-avg_vix)/avg_vix)

		#mitigate changes based on industry
		tech = diff_today_avg[diff_today_avg['industry'] == 'TECH'].VIX * 0.02
		agriculture = diff_today_avg[diff_today_avg['industry'] == 'AGRICULTURE'].VIX * 0.02
		finance = diff_today_avg[diff_today_avg['industry'] == 'FINANCE'].VIX * 0.02
		consumer = diff_today_avg[diff_today_avg['industry'] == 'CONSUMER'].VIX * 0.10
		other = diff_today_avg[diff_today_avg['industry'] == 'OTHER'].VIX * 0.02
	
		result = pd.concat([tech, agriculture, finance, consumer, other])
		return result		
		
	#TODO
	# second vix signal
	def vix2_signal(self, stock_features):
		last2 = stock_features[['ticker','industry','VIX']].tail(2000)
		yest = last2.head(1000).set_index('ticker')
		today = last2.tail(1000).set_index('ticker')
		#TODO - make quadratic
		pct_chg = today[['industry']].join((-1*(today.VIX - yest.VIX)/yest.VIX) ** 3)

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

	#TODO - implement mitigated linear distribution based on deviance from average	
	def rain_signal(self, stock_features):	
		avg_rain = 0.377809102452
		today = stock_features[['ticker', 'industry', 'RAIN']].tail(1000)
		today = today.set_index('ticker')
		diff_today_avg = today[['industry']].join(today.RAIN-avg_rain)

		#mitigate changes based on industry
		tech = diff_today_avg[diff_today_avg['industry'] == 'TECH'].RAIN * 0.02
		agriculture = diff_today_avg[diff_today_avg['industry'] == 'AGRICULTURE'].RAIN * 0.02
		finance = diff_today_avg[diff_today_avg['industry'] == 'FINANCE'].RAIN * 0.02
		consumer = diff_today_avg[diff_today_avg['industry'] == 'CONSUMER'].RAIN * 0.10
		other = diff_today_avg[diff_today_avg['industry'] == 'OTHER'].RAIN * 0.02
	
		result = pd.concat([tech, agriculture, finance, consumer, other])
		return result		

	#TODO - value as compared to mean
	def senti_signal(self, stock_features):
		avg_senti = 68.9302780531
		today = stock_features[['ticker','industry', 'SENTI']].tail(1000)
		today = today.set_index('ticker')
		diff = today[['industry']].join(today.SENTI-avg_senti)

		#changes based on industry
		tech = diff[diff['industry'] == 'TECH'].SENTI * 0.25
		agriculture = diff[diff['industry'] == 'AGRICULTURE'].SENTI * 0.2
		finance = diff[diff['industry'] == 'FINANCE'].SENTI * 0.3
		consumer = diff[diff['industry'] == 'CONSUMER'].SENTI * 0.20
		other = diff[diff['industry'] == 'OTHER'].SENTI * 0.2
		
		result = pd.concat([tech,agriculture, finance, consumer, other])
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
		#yest = last2.head(1000).set_index('ticker')
		today = last2.tail(1000).set_index('ticker')
		#inv_pct_chg = today[['industry']].join((yest['3M_R'] - today['3M_R'])/yest['3M_R'])
		#TODO - this makes no sense -- bond rates should be inversely correlated
		avg3m_r = 0.092660508209
		diff_today_avg = today['3M_R'] - avg3m_r
		diff = today[['industry']].join(diff_today_avg)
		return diff['3M_R']


		return inv_pct_chg['3M_R']


	# this is the function that actually builds the final signals based on the
	# all the individual signal functions above
	def build_signal(self, stock_features):

		#penalty for high pb ratio firms #TODO - make more fine-tuned
		high_pb = self.get_high_pb_inds(stock_features)
		high_pb_penalty = np.zeros(1000)
		high_pb_penalty[high_pb] -= 5

		#copper signal
		copp = self.copper_signal(stock_features)
		#stats['copp'].append(copp)
		copp.clip_upper(thresholds['copp'])
		copp.clip_lower(-1*thresholds['copp'])
		copp = (10 / thresholds['copp']) * copp
		
		#sentiment signal
		senti = self.senti_signal(stock_features)
		#stats['senti'].append(senti)
		senti.clip_upper(thresholds['senti'])
		senti.clip_lower(-1*thresholds['senti'])
		senti = (10 / thresholds['senti']) * senti
		
		#rain signal
		rain = self.rain_signal(stock_features)
		#stats['rain'].append(rain)
		rain.clip_upper(thresholds['rain'])
		rain.clip_lower(-1*thresholds['rain'])
		rain = (10 / thresholds['rain']) * rain
		
		#t-bill signal
		sig_3mr = self.get_3mr_signal(stock_features)
		#stats['3mr'].append(sig_3mr)
		sig_3mr.clip_upper(thresholds['3mr'])
		sig_3mr.clip_lower(-1*thresholds['3mr'])
		sig_3mr = (10 / thresholds['3mr']) * sig_3mr

		#provide boost to small-cap firms
		small_inds = self.get_small_cap_inds(stock_features)
		small_boost = np.zeros(1000)
		large_boost = np.zeros(1000) + 5
		small_boost[small_inds] += 5
		large_boost[small_inds] -= 5

		#calculate ix signal
		small_ix = self.ix_signal(stock_features, 'SMALL_IX', [1.4, .74, .8, 1.1]) 
		big_ix = self.ix_signal(stock_features, 'BIG_IX', [1.1, .74, .8, 1.1]) 
		modified_ix = small_ix * small_boost + big_ix * large_boost
		#stats['mod_ix'].append(modified_ix)
		modified_ix = (10 / thresholds['mod_ix']) * modified_ix
		
		#calculate vix2 signal
		vix_2 = self.vix2_signal(stock_features)
		#stats['vix_2'].append(vix_2)
		vix_2.clip_upper(thresholds['vix_2'])
		vix_2.clip_lower(-1*thresholds['vix_2'])
		vix_2 = (10 / thresholds['vix_2']) * vix_2
		
		#vix signal
		vix = self.vix_signal(stock_features)
		#stats['vix'].append(vix)
		vix.clip_upper(thresholds['vix'])
		vix.clip_lower(-1*thresholds['vix'])
		vix = (10 / thresholds['vix']) * vix
		
		#oil signal
		oil = self.oil_signal(stock_features)
		#stats['oil'].append(oil)
		oil.clip_upper(thresholds['oil'])
		oil.clip_lower(-1*thresholds['oil'])
		oil = (10 / thresholds['oil']) * oil
		
		#temperature signal
		temp =	self.temp_signal(stock_features)
		#stats['temp'].append(temp)
		temp.clip_upper(thresholds['temp'])
		temp.clip_lower(-1*thresholds['temp'])
		temp = (10 / thresholds['temp']) * temp

		result = 1.5*copp + .01*senti + .3*rain + 1.2*sig_3mr + .3*vix_2 + .8*vix + .6*oil + .01*temp + 2.75*small_boost + 3*high_pb_penalty
		return result
		
	
		

#main for testing
if __name__=='__main__':
	gen = Generator()
	print gen.simulate_portfolio()
	
	'''
	for cat, values in stats.iteritems():
		arr = np.asarray(values)
		mean = np.mean(arr)
		std = np.std(arr)
		threshold = max(abs(mean-2*std), abs(mean+2*std))
		print cat, threshold
	'''
