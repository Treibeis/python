import numpy as np
import csv
import sys
import time
import argparse

month = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul': 7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

def date_trans(s):
	dd = int(s[:2])
	mm = month[s[2:5]]
	yy = int(s[5:])
	return yy*10000 + mm*100 + dd
	

def readdata(fname, nmax = 1e16, red = 100, head = 1):
	data = [[], [], []]
	with open(fname, 'r') as f:
		reader = csv.reader(f)
		count = 0
		i = 0
		for row in reader:
			i += 1
			if i<=head:
				continue
			lst = row[0].split('\t')
			if lst[4]!='':
				data[0].append(int(date_trans(lst[1])/red))
				data[1].append(lst[8])
				data[2].append(float(lst[4]))
				count += 1
				if count>nmax:
					break
	print('Number of valid samples: {}'.format(count))
	return data

def func(data, date, industry, percentile, red = 100):
	t0 = time.time()
	data[0] = np.array(data[0])
	data[1] = np.array(data[1])
	data[2] = np.array(data[2])
	date_ = int(date_trans(date)/red)
	select = (data[0] == date_) * (data[1] == industry)
	new_data = data[2][select]
	t1 = time.time()
	print('Time: taken {} s'.format(t1-t0))
	return np.percentile(new_data, percentile)

def read_new(s, k = 0):
	out = [[], [], []]
	j = 0
	with open(s, 'r') as f:
		for line in f:
			lst = line.split(', ')
			if j>=k:
				out[0].append(int(lst[0]))
				out[1].append(lst[1])
				out[2].append(float(lst[2]))
			j = j+1
	return out

def write_new(s, l, k = 0):
	j = 0
	with open(s, 'w') as f:
		for i in range(len(l[0])):
			if j<k:
				print (l[0][i])
			else:
				f.write(str(l[0][i]))
				f.write(', ')
				f.write(l[1][i])
				f.write(', ')
				f.write(str(l[2][i]))
				f.write('\n')
			j = j+1

if __name__ == "__main__":
	title = 'Mein Schatz, \n'+ \
	'The function that really does the work here is func (line 37)，\n'+ \
	'while other functions are meant for data format transformation and file input/output. \n'+ \
	'The meanings of the parameters for func are apparent by their names, except for red, \n'+ \
	'which sets the range of statistics, i.e. red = 100 for the month, red = 10000 for the year. \n'+ \
	'You may write a circulation that goes through all the months and industries that you are interested at to call the function func, and write the results into a file. This will solve your problem (See read_new and write_new for examples of file I/O with python). '
	parser = argparse.ArgumentParser(description=title)
	parser.add_argument('--f', help='input file name', required=False)
	parser.add_argument('--mode', help='specifies the type of the input file (csv/txt)', required=False)
	parser.add_argument('--d', help='date, e.g. 13oct1995', required=False)
	parser.add_argument('--i', help='industry, e.g. "Utilities"', required=False)
	parser.add_argument('--red', help='sets the range of statistics, red = 100 for the month, red = 10000 for the year', type=int, required=False)
	parser.add_argument('--nf', help='new file name', required=False)
	parser.add_argument('--p', help='percentile', type=float, required=False)
	args = parser.parse_args()
	args = vars(args)
	if args['f']==None:
		fname = 'month.csv'
	else:
		fname = args['f']
	if args['nf']==None:
		nfname = 'month.txt'
	else:
		nfname = args['nf']
	if args['mode']=='txt':
		mode = 1
	else:
		mode = 0
	if args['d']==None:
		date = '30jun1988'
	else:
		date = args['d']
	if args['i']==None:
		industry = '"Utilities"'
	else:
		ind = args['i']
		if args['i'][0]!='"':
			ind = '"'+ind
		if args['i'][-1]!='"':
			ind = ind+'"'
		industry = ind
	if args['red']==None:
		red = 100
	else:
		red = 10000*(args['red']>100) + 100*(args['red']<=100)
	if args['p']==None:
		p = 50.0
	else:
		p = args['p']

	if mode==0:
		data = readdata(fname, red=red)
		write_new(nfname,data,0)
	else:
		data = read_new(nfname)
	out = func(data, date, industry, p, red)
	print('The '+str(p)+'% percentile at '+date[2:]+' for '+industry+' is '+str(out)+'.')




