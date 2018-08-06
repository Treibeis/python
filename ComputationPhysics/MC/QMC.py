import numpy as np
import matplotlib.pyplot as plt
import math
import random

def T(x):
	return math.e**(-x**2/2.0)/math.pi**0.5/2**0.5

def theory(n = 1200, l = [-6.0,6.0]):
	b = (l[1]-l[0])/n
	lx, lf = [], []
	x = l[0]
	lx.append(x)
	lf.append(T(x))
	for i in range(n):
		x += b
		lx.append(x)
		lf.append(T(x))
	return [lx, lf]
	

def trial(x, sig):
	return 1.0/math.e**(x**2/4/sig**2)/(2*math.pi)**0.25/sig**0.5

def dtrial(x, sig):
	return -x/2.0/math.e**(x**2/4)/(2*math.pi)**0.25/sig**2.5

def ddtrial(x, sig):
	return (-0.5/sig**2.5+x**2/4/sig**4.5)/math.e**(x**2/4)/(2*math.pi)**0.25

def V(x):
	return x**2/2.0

def H(x, sig):
	return -ddtrial(x, sig)/2.0+V(x)*trial(x, sig)

def walk(N0 = 50, ds = 0.1, sig = 2.0, E = 1.0, ad = 0.01, em = 500, pre = 200, N = 500):
	lx = []
	for i in range(N0):
		lx.append(random.normalvariate(0,sig))
	ln0 = []
	ln0.append(N0)
	for i in range(pre):
		lxx = []
		for j in range(len(lx)):
			x = lx[j]
			xx = x+random.normalvariate(0,1)*ds**0.5+dtrial(x, sig)*ds/trial(x, sig)
			M = int(random.random()+math.e**(-ds*((H(x, sig)/trial(x, sig)+H(xx, sig)/trial(xx, sig))/2.0-E)))
			if M!=0:
				for k in range(M):
					lxx.append(xx)
		lx = lxx
		ln0.append(len(lx))
		print (len(lx))
		if ln0[i]>em:
			if ln0[i+1]>ln0[i]:
				E = E-ad
			elif ln0[i+1]<ln0[i]:
				E = E+ad
	l = []
	ln = []
	lE = []
	l.append(lx)
	ln.append(len(lx))
	lE.append(E)
	for i in range(N-1):
		lxx = []
		for j in range(len(lx)):
			x = lx[j]
			xx = x+random.normalvariate(0,1)*ds**0.5+dtrial(x, sig)*ds/trial(x, sig)
			M = int(random.random()+math.e**(-ds*((H(x, sig)/trial(x, sig)+H(xx, sig)/trial(xx, sig))/2.0-E)))
			if M!=0:
				for k in range(M):
					lxx.append(xx)
		lx = lxx
		l.append(lx)
		ln.append(len(lx))
		print (len(lx))
		if ln[i+1]>ln[i]:
			E = E-ad
		elif ln[i+1]<ln[i]:
			E = E+ad
		lE.append(E)
	mi = min([min(lx) for lx in l])
	ma = max([max(lx) for lx in l])
	bi = sum([len(lx) for lx in l])/10/N
	lf = []
	for i in range(N):
		lf += l[i]
	lh = np.histogram(lf, bi, (mi,ma), normed = 1)	
	out = {}
	out['all'], out['final'], out['num'], out['E'], out['hist'] = l, lf, ln, lE, lh
	return out
			
