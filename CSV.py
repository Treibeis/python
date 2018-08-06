import csv
import numpy as np
#import matplotlib.pyplot as plt
#import mpl_toolkits.mplot3d


def tocsv(d, s = '-v-x.csv', j = 'x', k = 'v', l = 1):
	n=d['n']
	with open(s,'wb') as csvfile:
		spamwriter = csv.writer(csvfile, dialect = 'excel')
		for i in range(n):
			if l==0:
				a, b = str(d[j][i]), str(d[k][i])
			else:
				a, b = str(d[j][i]), str(-d[k][i])
			spamwriter.writerow([a, b])

def tocsv3D(d, s = 'z(x,y).csv', i = 'x', j = 'y', k = 'z', l = 0):
	n = d['n']*d['m']
	with open(s, 'wb') as csvfile:
		spamwriter = csv.writer(csvfile, dialect = 'excel')
		for m in range(n):
			if l==0:
				a, b, c = str(d[i][m]), str(d[j][m]), str(d[k][m])
			else:
				a, b, c = str(d[i][m]), str(d[j][m]), str(-d[k][m])
			spamwriter.writerow([a, b, c])
			
			
def recsv(s, n, k = 1, t = 1):
	out = []
	j = 0
	for i in range(n):
		out.append([])
	with open(s, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			if j<k:
				print (row)
			else:
				for i in range(n):
					out[i].append(float(row[i]))
			j = j+1
	if t!=0:
		for i in range(n):
			out[i].reverse()
			out[i] = np.array(out[i])
	return np.array(out)
	
def recsv3D(s, i = 'x', j = 'y', k = 'z', l = 0):
	l1, l2, l3 = [], [], []
	n = 0
	with open(s, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			l1.append(float(row[0]))
			l2.append(float(row[1]))
			if l==0:
				l3.append(float(row[2]))
			else:
				l3.append(-float(row[2]))
			n = n+1
	t = 0
	while l1[t]==l1[t+1]:
		t = t+1
	
	d = {i:l1, j:l2, k:l3, 'm':t+2 ,'n':n/(t+2)}
	return d
