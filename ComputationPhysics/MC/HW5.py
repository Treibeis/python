import random as r
import numpy as np
import matplotlib.pyplot as plt
import math
import mpl_toolkits.mplot3d

def dice():
	s = -1
	while s<0:
		s = 1	
		a = r.random()
		if (0<a) and (a<1.0/6):
			return 1
		elif (1.0/6<a) and (a<1.0/3):
			return 2
		elif (1.0/3<a) and (a<1.0/2):
			return 3
		elif (1.0/2<a) and (a<2.0/3):
			return 4
		elif (2.0/3<a) and (a<5.0/6):
			return 5
		elif (5.0/6<a):
			return 6
		else:
			s = -1

def playdice(n = 1000):
	l = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	for i in range(n):
		a = dice()
		if a==1:
			l[0] += 1
		elif a==2:
			l[1] += 1
		elif a==3:
			l[2] += 1
		elif a==4:
			l[3] += 1
		elif a==5:
			l[4] += 1
		else:
			l[5] += 1
	l = [x/n for x in l]
	error = [x-1.0/6 for x in l]
	std = 0
	for i in range(6):
		std += error[i]**2
	std = (std/6.0)**0.5
	d = {}
	d['result'], d['error'], d['std'], d['t'] = l, error, std, [1.0/6 for x in range(6)]
	return d
	
def sale(np1, np2, mu = 30.0, sigma = 3.0):
	b1 = -1
	b2 = -1
	while (b1<0) or (b2<0):
		a = r.normalvariate(mu, sigma)
		if np1<a:
			b1 = np1
		else:
			b1 = int(a)
		if np2<a:
			b2 = np2
		else:
			b2 = int(a)
	return [b1, b2]
	
def plan(n0 = 4000, mu = 3000.0, sigma = 300.0, day = 1000, cost = 2.5, price = 5.0):
	lp1, lp2, ls1, ls2 = [], [], [], []
	s0 = sale(n0, n0, mu, sigma)[0]
	s00 = sale(n0, n0, mu, sigma)[0]
	lp1.append(s0)
	lp2.append((s0+s00)/2)
	s1 = sale(lp1[0], lp2[0], mu, sigma)
	ls1.append(s1[0])
	ls2.append(s1[1])
	lp1.append(s1[0])
	lp2.append((s1[1]+s0)/2)
	s2 = sale(lp1[1], lp2[1], mu, sigma)
	ls1.append(s2[0])
	ls2.append(s2[1])
	for i in range(day-2):
		lp1.append(ls1[i+1])
		lp2.append((ls2[i]+ls2[i+1])/2)
		s = sale(lp1[i+2], lp2[i+2], mu, sigma)
		ls1.append(s[0])
		ls2.append(s[1])
	profit1, profit2 = [], []
	for i in range(day):
		profit1.append(ls1[i]*5.0-lp1[i]*2.5)
		profit2.append(ls2[i]*5.0-lp2[i]*2.5)
	sp1 = sum(profit1)
	sp2 = sum(profit2)
	a11 = sp1/sum(lp1)
	a12 = sp1/sum(ls1)
	a21 = sp2/sum(lp2)
	a22 = sp2/sum(ls2)
	d = {}
	d['p1'], d['s1'], d['profit1'], d['per1'], d['a11'], d['a12'] = lp1, ls1, profit1, sp1/day, a11, a12
	d['p2'], d['s2'], d['profit2'], d['per2'], d['a21'], d['a22'] = lp2, ls2, profit2, sp2/day, a21, a22
	return d

def f1(x, y, z):
	return A()*(x*y*z)**2
	
def f2(x, y, z):
	return x*y*z*8

def p1():
	x, y, z = r.normalvariate(0,1), r.normalvariate(0,1), r.normalvariate(0,1)
	return [x, y, z]
def w1(x, y, z):
	return math.e**(-(x**2+y**2+z**2)/2)/A()

def p2():
	a1, a2, a3 = r.random(), r.random(), r.random()
	x = (-2.0*math.log(a1))**0.5
	y = (-2.0*math.log(a2))**0.5
	z = (-2.0*math.log(a3))**0.5
	return [x, y, z]

def w2(x, y, z):
	return x*y*z*math.e**(-(x**2+y**2+z**2)/2)
	
def A():
	return (2*math.pi)**1.5
	
def int_3d1(n = 1000, g = f1, dis = p1, norm = A):
	s = 0
	lx, ly, lz = [], [], []
	for i in range(n):
		l = dis()
		x, y, z = l[0], l[1], l[2]
		lx.append(x)
		ly.append(y)
		lz.append(z)
		s += g(x,y,z)
	result = s/n
	d = {}
	d['x'], d['y'], d['z'], d['r'], d['e'] = lx, ly, lz, result, result-norm()
	return d

def int_3d2(n = 1000, g = f1, dis = p1, p = w1, h = 1.0, n0 = 10000, norm = A):
	s = 0.0
	lx, ly, lz = [], [], []
	a, b = 0.0, 0.0
	l = dis()
	x, y, z = l[0], l[1], l[2]
	lx.append(x)
	ly.append(y)
	lz.append(z)
	a += 1
	for i in range(n+n0-1):
		x = h*(2*r.random()-1)+lx[i]
		y = h*(2*r.random()-1)+ly[i]
		z = h*(2*r.random()-1)+lz[i]
		k = min(1, p(x, y, z)/p(lx[i], ly[i], lz[i]))
		R = r.random()
		if k>=R:
			b += 1
			lx.append(x)
			ly.append(y)
			lz.append(z)
		else:
			a += 1
			lx.append(lx[i])
			ly.append(ly[i])
			lz.append(lz[i])
	for j in range(n):
		s += g(lx[n0+j], ly[n0+j], lz[n0+j])
	result = s/n
	d = {}
	d['x'], d['y'], d['z'], d['r'], d['e'], d['a'] = lx, ly, lz, result, result-norm(), b/(n+n0)
	return d

def statistic2(g = f1, dis = p1, p = w1, m = 1000, n = 1000, h = 1.0, n0 = 10000, norm = A):
	s = 0.0
	error = 0.0
	la = []
	std = 0.0
	for i in range(m):
		a = int_3d2(n, g, dis, p, h, n0, norm)
		la.append(a['r'])
		s += a['r']
		error += a['e']**2
	average = s/m
	for i in range(m):
		std += (la[i]-average)**2
	std = (std/m)**0.5
	error = (error/m)**0.5
	d = {}
	d['a'], d['std'], d['error'], d['la'] = average, std, error, la
	return d
	
def statistic1(g = f1, dis = p1, m = 1000, n = 10000, norm = A):
	s = 0
	error = 0
	la = []
	std = 0
	for i in range(m):
		a = int_3d1(n, g, dis, norm)
		la.append(a['r'])
		s += a['r']
		error += a['e']**2
	average = s/m
	for i in range(m):
		std += (la[i]-average)**2
	std = (std/m)**0.5
	error = (error/m)**0.5
	d = {}
	d['a'], d['std'], d['error'], d['la'] = average, std, error, la
	return d
	
