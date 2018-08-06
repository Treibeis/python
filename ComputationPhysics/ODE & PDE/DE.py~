import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

A = 0.5

def fxx(x, y):
	return -x/(x**2+y**2)**1.5

def fyy(x, y):
	return -y/(x**2+y**2)**1.5

def E(l):
	return (l[1]**2+l[3]**2)/2-1/(l[0]**2+l[2]**2)**0.5

def I(l):
	return l[0]*l[3]-l[2]*l[1]

def ini(e = 0.5):
	return [1.0-e, 0.0, 0.0, ((1.0+e)/(1.0-e))**0.5]

def R_K(e = 0.0, N = 2, n = 100000, a1 = A):
	h = N*2.0*math.pi/n
	a2 = 1-a1 
	gam = 0.5/a2
	l0 = ini(e)
	lx, lxx, ly, lyy, lt, lE, lI , lp= [], [], [], [], [], [], [], []
	x, xx, y, yy = l0[0], l0[1], l0[2], l0[3]
	lx.append(l0[0])
	lxx.append(l0[1])
	ly.append(l0[2])
	lyy.append(l0[3])
	lt.append(0.0)
	lE.append(E(l0))
	lI.append(I(l0))
	t = 0.0
	k = 1
	for i in range(n):
		t = t+h
		kx1, kxx1, ky1, kyy1 = h*xx, h*fxx(x, y), h*yy, h*fyy(x, y)
		kx2, kxx2 = h*(xx+kxx1*gam), h*fxx(x+kx1, y+ky1*gam)
		ky2, kyy2 = h*(yy+kyy1*gam), h*fyy(x+kx1, y+ky1*gam)
		x = x+kx1*a1+kx2*a2
		y = y+ky1*a1+ky2*a2
		xx = xx+kxx1*a1+kxx2*a2
		yy = yy+kyy1*a1+kyy2*a2
		l = [x, xx, y, yy]
		lE.append(E(l))
		lI.append(I(l))
		lx.append(x)
		lxx.append(xx)
		ly.append(y)
		lyy.append(yy)
		lt.append(t)
		if (i+1)*N*2/n==k:
			lp.append([x, y])
			k = k+1
	d = {}
	d['x'], d['y'], d['t'], d['E'], d['I'], d['n'], d['e'] , d['p'] = lx, ly, lt, lE, lI, n+1, e, lp
	return d
		

def y_x(d1 = R_K(0.0), d2 = R_K(0.5), d3 = R_K(0.9), l = [-2.0,1.0,-9/8.0,9/8.0]):
	plt.plot(d1['x'],d1['y'])
	plt.plot(d2['x'],d2['y'])
	plt.plot(d3['x'],d3['y'])
	plt.axis(l)
	plt.xlabel(r'$x$',size=20.0)
	plt.ylabel(r'$y$',size=20.0)

def xy_t(d1 = R_K(0.0), d2 = R_K(0.5), d3 = R_K(0.9), l1 = [0,4*math.pi,-2.0,1.0], l2 = [0,4*math.pi,-1,1]):
	plt.subplot(211)
	plt.plot(d1['t'],d1['x'])
	plt.plot(d2['t'],d2['x'])
	plt.plot(d3['t'],d3['x'])
	plt.axis(l1)
	plt.xlabel(r'$t$')
	plt.ylabel(r'$x$')
	plt.subplot(212)
	plt.plot(d1['t'],d1['y'])
	plt.plot(d2['t'],d2['y'])
	plt.plot(d3['t'],d3['y'])
	plt.axis(l2)
	plt.xlabel(r'$t$')
	plt.ylabel(r'$y$')

def E_t(d1 = R_K(0.0), d2 = R_K(0.5), d3 = R_K(0.9)):
	plt.subplot(311)
	plt.plot(d1['t'],d1['E'],'b')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$E_{1}$')
	plt.subplot(312)
	plt.plot(d2['t'],d2['E'],'g')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$E_{2}$')
	plt.subplot(313)
	plt.plot(d3['t'],d3['E'],'r')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$E_{3}$')
	

def I_t(d1 = R_K(0.0), d2 = R_K(0.5), d3 = R_K(0.9)):
	plt.subplot(311)
	plt.plot(d1['t'],d1['I'],'b')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$I_{1}$')
	plt.subplot(312)
	plt.plot(d2['t'],d2['I'],'g')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$I_{2}$')
	plt.subplot(313)
	plt.plot(d3['t'],d3['I'],'r')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$I_{3}$')
