import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time

def nearest(x, y, m = 10, n = 10):
	p = []
	if x==m:
		p.append([0,y])
	else:
		p.append([x+1,y])
	if x==0:
		p.append([m,y])
	else:
		p.append([x-1,y])
	if y==n:
		p.append([x,0]) 
	else:
		p.append([x,y+1])
	if y==0:
		p.append([x,n])
	else:
		p.append([x,y-1])
	return p

def energy(x, y, s, f, m = 10, n = 10, J = 1):
	p = nearest(x, y, m, n)
	e = 0
	for i in range(len(p)):
		e += s*f[p[i][0]][p[i][1]]*J
	return -e/2.0

def Etotal(H, f, m = 10, n = 10, J = 1):
	s = 0
	for i in range(m+1):
		for j in range(n+1):
			s += energy(i, j, f[i][j], f, m, n, J) - H*f[i][j]
	return s

def locusT(mc, lt = [1.5,3.0], step = 15, num = 1, H = 0.0, ini = 1, pre = 2*10**2, N = 400, m = 24, n = 24, J = 1):
	t = lt[0]
	b = (lt[1]-lt[0])/step
	lt = []
	lM = []
	le = []
	lr = []
	lMe, lEe, lre = [], [], []
	#ld = []
	for i in range(step+1):
		d = statistic(mc, num,t,H,ini,pre,N,m,n,J)
		#ld.append(d)
		lM.append(d['M'])
		lMe.append(d['sm'])
		le.append(d['E'])
		lEe.append(d['se'])
		lr.append(d['rate'])
		lre.append(d['sr'])
		lt.append(t)
		t += b
	out = {}
	out['M'], out['E'], out['T'], out['rate'], out['me'], out['ee'], out['re'] = lM, le, lt, lr, lMe, lEe, lre
	return out
		
def statistic(mc, num = 50, T = 2.0, H = 0.0, ini = 1, pre = 1*10**4, N = 2*10**3, m = 10, n = 10, J = 1):
	t1 = time.time()
	lM, le, lr, ld = [], [], [], []
	lvarM, lvarE = [], []
	for i in range(num):
		d = mc(T,H,0,0,ini,pre,N,m,n,J)
		ld.append(d)
		lM.append(d['M'])
		le.append(d['E'])
		lr.append(d['rate'])
		lvarM.append(d['varM'])
		lvarE.append(d['varE'])
	am = sum(lM)/float(num)
	ae = sum(le)/float(num)
	ar = sum(lr)/float(num)
	varM = sum(lvarM)/float(num)
	varE = sum(lvarE)/float(num)
	sm, se, sr = 0.0, 0.0, 0.0
	for i in range(num):
		sm += (lM[i]-am)**2
		se += (le[i]-ae)**2
		sr += (lr[i]-ar)**2
	sm = (sm/float(num))**0.5
	se = (se/float(num))**0.5
	sr = (sr/float(num))**0.5
	out = {}
	out['M'], out['E'], out['rate'], out['sm'], out['se'], out['sr'], out['lM'], out['lE'], out['lr'], out['d']  = am, ae, ar, sm, se, sr, lM, le, lr, ld
	t2 = time.time()
	print (t2-t1)
	return out

def MC2(T = 3.0, H = 0.0, mode = 0, cut = 2000, ini = 0, pre = 0, N = 1000, m = 10, n = 10, J = 1):
	t1 = time.time()
	if ini==0:
		f = np.array([[random.randrange(-1,3,2) for x in range(m+1)] for y in range(n+1)])
	else: 
		f = np.ones((m+1,n+1))
	lM, le, lf = [], [], []
	lf.append(f)
	for i in range(pre):
		lff = []
		lff.append(lf[i])
		for j in range(m+1):
			for k in range(n+1):
				f1 = np.copy(lff[j*(m+1)+k])
				x = j
				y = k
				if lff[j*(m+1)+k][x][y]==-1:
					f1[x][y] = 1
				else:
					f1[x][y] = -1
				de = 2*(energy(x, y, f1[x][y], f1, m, n, J)-energy(x, y, lff[j*(m+1)+k][x][y], lff[j*(m+1)+k], m, n, J)) -H*(f1[x][y]-lff[j*(m+1)+k][x][y])
				if de<=0:
					lff.append(f1)
				else:
					p = math.e**(-de/T)
					r = random.random()
					if r<=p:
						lff.append(f1)
					else:
						lff.append(lff[j*(m+1)+k])
		lf.append(lff[(m+1)*(n+1)])
	count = 0
	f0 = lf[pre]
	lf = []
	lf.append(f0)
	lM.append(f0.sum()/(m+1.0)/(n+1.0))
	le.append(Etotal(H, f0, m, n)/(m+1.0)/(n+1.0))
	for i in range(N-1):
		lff = []
		lff.append(lf[i])
		for j in range(m+1):
			for k in range(n+1):
				f1 = np.copy(lff[j*(m+1)+k])
				x = j
				y = k
				if lff[j*(m+1)+k][x][y]==-1:
					f1[x][y] = 1
				else:
					f1[x][y] = -1
				de = 2*(energy(x, y, f1[x][y], f1, m, n, J)-energy(x, y, lff[j*(m+1)+k][x][y], lff[j*(m+1)+k], m, n, J)) -H*(f1[x][y]-lff[j*(m+1)+k][x][y])
				if de<=0:
					lff.append(f1)
					count += 1
				else:
					p = math.e**(-de/T)
					r = random.random()
					if r<=p:
						lff.append(f1)
					else:
						lff.append(lff[j*(m+1)+k])
		lf.append(lff[(m+1)*(n+1)])
		lM.append(lf[i+1].sum()/(m+1.0)/(n+1.0))
		le.append(Etotal(H, lf[i+1], m, n)/(m+1.0)/(n+1.0))
	aM, aE = sum(lM)/float(N), sum(le)/float(N)
	varM, varE = 0.0, 0.0
	for i in range(N):
		varM += (lM[i]-aM)**2
		varE += (le[i]-aE)**2
	varM, varE = varM/float(N), varE/float(N)
	t2 = time.time()
	print (t2-t1)
	if mode ==1: 
		# It seems that this method of estimating the std can be really time-consuming as well, since we need a large N to obtain a good approximation of t and the complexity is O(N^2).
		# Besides, the efficient method for computing t is slow as well, in that the number of steps took for equilibration is large (~0.5-1.0*10^4).
		lfM, lfE = [], []
		for j in range(min(cut-1,N-1)):
			sm, se = 0.0, 0.0
			t = j+1
			for i in range(N):
				sm += lM[i-N]*lM[i+t-N]
				se += le[i-N]*le[i+t-N]
			sm, se = sm/float(N), se/float(N)
			lfM.append((sm-aM**2)/varM)
			lfE.append((se-aE**2)/varE)
		tM = 1+2*sum(lfM)
		tE = 1+2*sum(lfE)
		sm, se = (abs(tM)*varM/float(N))**0.5, (abs(tE)*varE/float(N))**0.5
		d = {}
		d['lf'], d['lM'], d['lE'], d['M'], d['E'], d['rate'], d['sm'], d['se'], d['tM'], d['tE'], d['fM'], d['fE'] = lf, lM, le, abs(aM), aE, count/float(N)/(m+1)/(n+1), sm, se, tM, tE, lfM, lfE
		t3 = time.time()
		print (t3-t2)
	else:
		d = {}
		d['lf'], d['lM'], d['lE'], d['M'], d['E'], d['rate'], d['varM'], d['varE'] = lf, lM, le, aM, aE, count/float(N)/(m+1)/(n+1), varM, varE
	return d



def MC1(T = 3.0, H = 0.0, mode = 0, cut = 2000, ini = 1, pre = 10**4, N = 2025, m = 10, n = 10, J = 1):
	t1 = time.time()
	if ini==0:
		f = np.array([[random.randrange(-1,3,2) for x in range(m+1)] for y in range(n+1)])
	else: # The rate of equilibration highly depends on the initial condition.
		f = np.ones((m+1,n+1))
	lM, le, lf = [], [], []
	lf.append(f)
	for i in range(pre):
		f1 = np.copy(lf[i])
		x = random.randrange(0,m+1,1)
		y = random.randrange(0,n+1,1)
		if lf[i][x][y]==-1:
			f1[x][y] = 1
		else:
			f1[x][y] = -1
		de = 2*(energy(x, y, f1[x][y], f1, m, n, J)-energy(x, y, lf[i][x][y], lf[i], m, n, J)) -H*(f1[x][y]-lf[i][x][y])
		if de<=0:
			lf.append(f1)
		else:
			p = math.e**(-de/T)
			r = random.random()
			if r<=p:
				lf.append(f1)
			else:
				lf.append(lf[i])
	count = 0
	lff = []
	lff.append(lf[pre])
	lM.append(lff[0].sum()/(m+1.0)/(n+1.0))
	le.append(Etotal(H, lff[0], m, n)/(m+1.0)/(n+1.0))
	for i in range(N-1):
		f1 = np.copy(lff[i])
		x = random.randrange(0,m+1,1)
		y = random.randrange(0,n+1,1)
		if lff[i][x][y]==-1:
			f1[x][y] = 1
		else:
			f1[x][y] = -1
		de = 2*(energy(x, y, f1[x][y], f1, m, n, J)-energy(x, y, lff[i][x][y], lff[i], m, n, J)) -H*(f1[x][y]-lff[i][x][y])
		if de<=0:
			count += 1
			lff.append(f1)
		else:
			p = math.e**(-de/T)
			r = random.random()
			if r<=p:
				lff.append(f1)
			else:
				lff.append(lff[i])
		lM.append(lff[i+1].sum()/(m+1.0)/(n+1.0))
		le.append(Etotal(H, lff[i+1], m, n)/(m+1.0)/(n+1.0))
	aM, aE = sum(lM)/float(N), sum(le)/float(N)
	varM, varE = 0.0, 0.0
	for i in range(N):
		varM += (lM[i]-aM)**2
		varE += (le[i]-aE)**2
	varM, varE = varM/float(N), varE/float(N)
	t2 = time.time()
	print (t2-t1)
	if mode ==1: 
		# It seems that this method of estimating the std can be really time-consuming as well, since we need a large N to obtain a good approximation of t and the complexity is O(N^2).
		# Besides, the efficient method for computing t is slow as well, in that the number of steps took for equilibration is large (~0.5-1.0*10^4).
		lfM, lfE = [], []
		for j in range(min(cut-1,N-1)):
			sm, se = 0.0, 0.0
			t = j+1
			for i in range(N):
				sm += lM[i-N]*lM[i+t-N]
				se += le[i-N]*le[i+t-N]
			sm, se = sm/float(N), se/float(N)
			lfM.append((sm-aM**2)/varM)
			lfE.append((se-aE**2)/varE)
		tM = 1+2*sum(lfM)
		tE = 1+2*sum(lfE)
		sm, se = (abs(tM)*varM/float(N))**0.5, (abs(tE)*varE/float(N))**0.5
		d = {}
		d['lf'], d['lM'], d['lE'], d['M'], d['E'], d['rate'], d['sm'], d['se'], d['tM'], d['tE'], d['fM'], d['fE'] = lff, lM, le, abs(aM), aE, count/float(N), sm, se, tM, tE, lfM, lfE
		t3 = time.time()
		print (t3-t2)
	else:
		d = {}
		d['lf'], d['lM'], d['lE'], d['M'], d['E'], d['rate'], d['varM'], d['varE'] = lff, lM, le, aM, aE, count/float(N), varM, varE
	return d


