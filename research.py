#run through data files to calculate means

textfile = open("./stock_data/factor_data.csv")
lines =textfile.readlines()

#get average rainfall
#rainfall is 9th col
rainfalls= []
vix=[]
senti = []
for line in lines[1:]:
	senti.append(float(line.split(',')[6]))
	vix.append(float(line.split(',')[0]))
	rainfalls.append(float(line.split(',')[8]))

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
	
#average senti. senti is col 7
totalsenti =0
ct =0
for sent in senti:
	totalsenti += sent
	ct += 1
print("average senti per timestep {}".format(totalsenti/ct))


