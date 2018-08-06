import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time

def dla(k = 50, R = 40, p = 0.6, n = 1000):
	t1 = time.time()
	m = k*2+1
	f = np.zeros((m,m))
	count = 1
	num = 0
	f[k][k] = 1
	t = 0
	particle = []
	l1, l2 = [], []
	l1.append(k)
	l2.append(k)
	while (count<n) and (t==0):
		#x = int(random.random()*2*k)
		#y = int(random.random()*2*k)
		r1 = random.random()
		x = int (math.cos(r1*2*math.pi)*R+k)
		y = int (math.sin(r1*2*math.pi)*R+k)
		if ((x-k)**2+(y-k)**2<k**2) and (f[x][y]!=1):
			num += 1
			lx, ly = [], []
			lx.append(x)
			ly.append(y)
			s = 0
			if x+1<=2*k:
				if f[x+1][y]==1:
					s = 1
			if x-1>=0:
				if f[x-1][y]==1:
					s = 1
			if y+1<=2*k:
				if f[x][y+1]==1:
					s = 1
			if y-1>=0:
				if f[x][y-1]==1:
					s = 1
			if s==1:
				zz = random.random()
				if zz<p:
					count += 1
					f[x][y] = 1
					l1.append(x)
					l2.append(y)
					particle.append([lx,ly])
				else:
					s = 0
			out = 0
			while (s==0) and (out==0):
				z = random.random()
				if (0<=z) and (z<0.25):
					if f[x+1][y]==0:
						x += 1
				elif (0.25<=z) and (z<0.5):
					if f[x-1][y]==0:
						x -= 1
				elif (0.5<=z) and (z<0.75):
					if f[x][y+1]==0:
						y += 1
				else:
					if f[x][y-1]==0:
						y -= 1
				lx.append(x)
				ly.append(y)
				if x+1<=2*k:
					if f[x+1][y]==1:
						s = 1
				if x-1>=0:
					if f[x-1][y]==1:
						s = 1
				if y+1<=2*k:
					if f[x][y+1]==1:
						s = 1
				if y-1>=0:
					if f[x][y-1]==1:
						s = 1
				if s==1:
					zz = random.random()
					if zz<p:
						if (x-k)**2+(y-k)**2<R**2:
							f[x][y] = 1
							l1.append(x)
							l2.append(y)
							count += 1
							particle.append([lx,ly])
						else:
							t = 1
					else:
						s = 0
				else:
					if (x-k)**2+(y-k)**2>=k**2:
						out = 1
	d =  {}
	d['f'], d['trace'], d['x'], d['y'], d['num'], d['count'] = f, particle, l1, l2, num, count
	t2 = time.time()
	print (t2-t1)
	return d  


				
