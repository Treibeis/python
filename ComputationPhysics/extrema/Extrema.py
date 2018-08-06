import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import CSV
import time

A0 = np.array([[6.0,2.0,1.0],[2.0,3.0,1.0],[1.0,1.0,1.0]])
d0 = {'A':A0, 'n':3}

def f0(d, l):
	x1, x2 = l[0], l[1]
	out = 100*(x2-x1**2)**2+(1-x1)**2
	return out

def to_1d(d, l):
	"""Transfering d['f'] to a 1-D function with a new independent variable l[0] as to_1d(l[0]) = d['f'](d['l1']-l[0]*d['l2'])"""
	l = np.dot(l[0], d['l2'])
	out = d['f'](d['d'], d['l1']-l)
	return out

def f1(d, l):
	""" xTAx/xTx """
	A = d['A']
	l1 = np.dot(A,l)
	l2 = np.dot(l,l1)
	out = l2/np.dot(l,l)
	return out

def f1_2d(d, l):
	"""f1 projected to the x1-x2 plane with fixed x3 """
	ll = [l[0], l[1], d['x3']]
	out = f1(d, ll)
	return out

def Lf1(d, l):
	""" xTAx/xTx-l[d['n']]*(xTx-1), where xT=[l[x] for x in range(d['n'])]: The generated function of f1 with a 1-D constraint"""
	A = d['A']
	n = d['n']
	l0 = []
	for i in range(n):
		l0.append(l[i])
	l1 = np.dot(A,l0)
	l2 = np.dot(l0,l1)
	out = l2/np.dot(l0,l0)+l[3]*(np.dot(l0,l0)-1)
	return out

def D(f, i, n, l, d = {}, h = 1.0/10**12):
	"""Calcuating the partial derivative of f with respect to the i+1th variable at l which is a n-dimensional vector"""
	l1, l2 = [], []
	for j in range(n):
		l1.append(l[j])
		l2.append(l[j])
		if j==i:
			l1[j] = l[j]-h
			l2[j] = l[j]+h
	out = (f(d, l2)-f(d, l1))/2/h
	return out

def DD(f, i, j, n, l, d = {}, h = 1.0/10**10):
	"""Calcuating the second partial derivative of f with respect to the i+1th and j+1th variables at l which is a n-dimensional vector"""
	l1, l2 = [], []
	for k in range(n):
		l1.append(l[k])
		l2.append(l[k])
		if k==i:
			l1[k] = l[k]-h
			l2[k] = l[k]+h
	l11, l12, l21, l22 = [], [], [], []
	for k in range(n):
		l11.append(l1[k])
		l12.append(l1[k])
		l21.append(l2[k])
		l22.append(l2[k])
		if k==j:
			l11[k] = l1[k]-h
			l12[k] = l1[k]+h
			l21[k] = l2[k]-h
			l22[k] = l2[k]+h
	df1 = (f(d, l12)-f(d, l11))/2/h
	df2 = (f(d, l22)-f(d, l21))/2/h
	out = (df2-df1)/2/h
	return out

def test(d, l):
	x, y = l[0], l[1]
	return (x**2+y**2)

def Newtonian(f, n, l0, d = {}, criterion = 1.0/10**4):
	i = 0
	l = []
	ls = []
	s = f(d, l0)
	ls.append(s)
	l.append(l0)
	while abs(s)>criterion:
		lx = []
		for j in range(n):		
			lx.append(l[i][j]-ls[i]/D(f, j, n, l[i], d))
		#if lx[0]<=0:
		#	lx[0] = -lx[0]
		l.append(lx)
		print (lx)
		s = f(d, l[i+1])
		ls.append(s)
		#print (i)
		i = i+1
	d = {'i':i+1, 'lf':ls, 'n':n, 'f':ls[i], 'x': l[i]}
	return d

def Fast(f, n, l0, d = {}, criterion = 1.0/10**4, a0 = 1.0/10**4):
	i = 0
	s = 0
	lv = []
	l = []
	lf = []
	ls = []
	for k in range(n):
		s = D(f, k, n, l0, d)**2+s
		lv.append([])
		lv[k].append(l0[k])
		l.append(l0[k])
	lf.append(f(d, l0))
	ls.append(s**0.5)
	while (s**0.5>criterion):
		i = i+1
		dl = []
		for j in range(n):
			dl.append(D(f, j, n, l, d))
		a = Newtonian(to_1d, 1, [a0], {'f':f, 'l1':l, 'l2':dl, 'd':d},1.0/10**8)['x'][0]
		l = l-np.dot(a, dl)
		#print ("a,l",a,l)
		s = 0
		for k in range(n):
			s = D(f, k, n, l, d)**2+s
			lv[k].append(l[k])
		lf.append(f(d, l))
		ls.append(s**0.5)
	if lf[i]>lf[i-1]:
		label = 'Max'
	else:
		label = 'Min'
	d = {'i':i+1, 'lv':lv, 'lf':lf, 'n':n, 'x':[lv[k][i] for k in range(n)], 'f':lf[i], 's':ls, 'l':label}
	return d

def Conjugate(f, n, l0, d = {}, criterion = 1.0/10**8, a0 = 1.0/10**4):
	i = 0
	s = 0
	lv = []
	l = []
	lf = []
	ls = []
	dl = []
	lS = []
	for k in range(n):
		s = D(f, k, n, l0, d)**2+s
		lv.append([])
		lv[k].append(l0[k])
		l.append(l0[k])
		dl.append(D(f, k, n, l0, d))
	lf.append(f(d, l0))
	ls.append(s**0.5)
	lS.append(np.dot(-1.0,dl))
	a00 = 1.0
	while (s**0.5>criterion):
		dl = []
		for j in range(n):
			dl.append(D(f, j, n, l, d))
		a = Newtonian(to_1d, 1, [a0*a00], {'f':f, 'l1':l, 'l2':np.dot(-1.0,lS[i]), 'd':d}, 1.0/10**8)['x'][0]
		l = l+np.dot(a, lS[i])
		#print ("a,l",a,l)
		g = []
		for j in range(n):
			g.append(D(f, j, n, l, d))
		S = np.dot(np.dot(g,g)/np.dot(dl,dl),lS[i])-g
		lS.append(S)
		a00 = np.dot(g,g)/np.dot(S,S)
		s = 0.0
		for k in range(n):
			s = D(f, k, n, l, d)**2+s
			lv[k].append(l[k])
		lf.append(f(d, l))
		ls.append(s**0.5)
		i = i+1
	if lf[i]>lf[i-1]:
		label = 'Max'
	else:
		label = 'Min'
	d = {'i':i+1, 'lv':lv, 'lf':lf, 'n':n, 'x':[lv[k][i] for k in range(n)], 'f':lf[i], 's':ls, 'l':label}
	return d

# The following fuctions are meant for plotting figures or evaluating the results obtained from the above functions.

def normalization(l):
	out = [x/np.dot(l,l)**0.5 for x in l]
	return out

def points(f, d = {}, lx = [-8.0, 8.0], ly = [-8.0, 8.0], m = 200, n = 200):
	l1, l2, l3 = [], [], []
	do = {}
	do['m'], do['n'] = m, n
	sx = (lx[1]-lx[0])/(m-1)
	sy = (ly[1]-ly[0])/(n-1)
	x_, y_ = lx[0], ly[0]
	for i in range(m):
		y_ = ly[0]
		for j in range(n):
			z_ = f(d, [x_, y_])
			l1.append(x_)
			l2.append(y_)
			l3.append(z_)
			y_ = y_+sy
		x_ = x_+sx
	do['x'], do['y'], do['z'] = l1, l2, l3
	return do

def trans(d):
	m, n = d['m'], d['n']
	lx = []
	ly = []
	lz = []
	k = 0
	for s in range(n):
		lx.append([])
		ly.append([])
		lz.append([])
	for i in range(n):
		for j in range(m):
			lx[i].append(d['x'][k])
			lz[i].append(d['z'][k])
			ly[i].append(d['y'][k])
			k = k+1
	dd = {}
	dd['x'], dd['y'], dd['z'] =  lx, ly, lz
	dd['m'], dd['n'] = m, n
	return dd



