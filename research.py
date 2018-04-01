#run through data files to calculate means
import numpy as np

textfile = open("./stock_data/factor_data.csv")
tickerfile = open("./stock_data/ticker_data.csv")
lines =textfile.readlines()
tickers =tickerfile.readlines()

#get average rainfall
#rainfall is 9th col
rainfalls= []
vix=[]
sig_3mr = []
senti = []
for line in lines[1:]:
	senti.append(float(line.split(',')[6]))
	vix.append(float(line.split(',')[0]))
	sig_3mr.append(float(line.split(',')[2]))
	rainfalls.append(float(line.split(',')[8]))

#get mean, std
pb = []
for line in tickers[1:]:
	pb.append(float(line.split(',')[3]))
pb = np.asarray(pb)
print("average pb value: {}".format(np.mean(pb)))
print("std of pb values: {}".format(np.median(pb)))

#get average rainfall value
totalrainfall =0
timeperiods =0
for day in rainfalls:
	totalrainfall += (day)
	timeperiods += 1
	

average = totalrainfall / timeperiods
print("average rainfall per timestep: {}".format(average))

# get average VIX value. Vix is col 1
totalvix =0
ct =0

for VIX in vix:
	totalvix += (VIX)
	ct +=1
print("average VIX per timestep {}".format(totalvix/ct))

# get average 3MR value. 3MR is col 3
total3mr =0
ct =0

for r in sig_3mr:
	total3mr += (r)
	ct +=1
print("average 3M_R per timestep {}".format(total3mr/ct))
	
#average senti. senti is col 7
totalsenti =0
ct =0
for sent in senti:
	totalsenti += sent
	ct += 1
print("average senti per timestep {}".format(totalsenti/ct))


