import math 
import random 
import numpy as np
import matplotlib.pyplot as plt

def source(number = 5000, n = 5, size = [32, 32], ratio = [1.0, 2.0, 3.0, 2.0, 1.0], pdf = [2.0, 3.0, 2.0, 3.0, 2.0]):
	"""Generate point suorces and simulate the photon distribution with gauss PDF as a random sampling process"""
	sum = np.sum(ratio)
	subn = [int(number*x/sum) for x in ratio]
	pos = []
	out = []
	for i in range(n):
		x = random.uniform(0.0,size[0])
		y = random.uniform(0.0,size[1])
		pos.append([x,y])
	for k in range(n):
		for l in range(subn[k]):
			dx = random.gauss(0.0,pdf[k])
			dy = random.gauss(0.0,pdf[k])
			out.append([pos[k][0]+dx, pos[k][1]+dy])
	d = {'p':out, 'n':np.sum(subn), 's':size}
	return d
	
def img(input, move = [0.0, 0.0], rotate = [0.0, 1.0]):
	p = input['p']
	p1 = []
	p2 = []
	if (move[0]!=0) or (move[1]!=0):
		for i in range(input['n']):
			x1 = p[i][0]+move[0]
			y1 = p[i][1]+move[1]
			p1.append([x1,y1])
	else: 
		p1 = p
	if (rotate[0]!=0) or (rotate[1]!=1):
		for k in range(input['n']):
			x2 = ((p1[k][0]-float(input['s'][0])/2.0)*math.cos(rotate[0])+(p1[k][1]-float(input['s'][0])/2.0)*math.sin(rotate[0]))*rotate[1]+float(input['s'][0])/2.0
			y2 = ((p1[k][1]-float(input['s'][1])/2.0)*math.cos(rotate[0])-(p1[k][0]-float(input['s'][1])/2.0)*math.sin(rotate[0]))*rotate[1]+float(input['s'][1])/2.0
			p2.append([x2,y2])
	else:
		p2 = p1
	img = []
	t = 0
	for l in range(input['s'][1]):
		row = []
		for m in range(input['s'][0]):
			for j in range(input['n']):
				if ((m<p2[j][0])and(p2[j][0]<=m+1)) and ((l<p2[j][1])and(p2[j][1]<=l+1)):
					t = t+1
			row.append(t)
			t = 0
		img.append(row)
	d = {'o':input, 'i':img}
	return d
			
def noise_uni(d, a = 0.001):
	f = a*d['o']['n']/d['o']['s'][0]/d['o']['s'][1]
	i_ = []
	for i in range(d['o']['s'][1]):
		row = []
		for l in range(d['o']['s'][0]):
			k = f*random.uniform(0.0,1.0)
			row.append(d['i'][l][i]+k)
		i_.append(row)
	d_ = {'o':d['o'], 'i':i_}
	return d_
