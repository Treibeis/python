import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import CSV
import time

def f(x, x_, y, y_):
	z = 1.0/((x-x_)**2+(y-y_)**2)**0.5
	return z
	
def integral(x_, y_, lx = [-1.0, 1.0], ly = [-1.0, 1.0] , m = 100, n = 100):
	sx = (lx[1]-lx[0])/m
	sy = (ly[1]-ly[0])/n
	tx = m%2
	ty = n%2
	x, y = lx[0], ly[0]
	l = []
	for i in range(n+1):
		s = 0
		x = lx[0]
		for j in range((m-tx)/2):
			f1 = f(x, x_, y, y_)
			f2 = f(x+sx, x_, y, y_)
			f3 = f(x+2*sx, x_, y, y_)
			s = (f1+4*f2+f3)*sx/3+s
			x = x+2*sx
		if tx==1:
			f1 = f(x-sx, x_, y, y_)
			f2 = f(x, x_, y, y_)
			f3 = f(x+sx, x_, y, y_)
			s = (-f1+8*f2+5*f3)*sx/12+s
		l.append(s)
		y = y+sy
	ss = 0
	for i in range((n-ty)/2):
		f1 = l[2*i]
		f2 = l[2*i+1]
		f3 = l[2*i+2]
		ss = ss+(f1+4*f2+f3)*sy/3
	if ty==1:
		f1 = l[n-3]
		f2 = l[n-2]
		f3 = l[n-1]
		ss = ss+(-f1+8*f2+5*f3)*sy/12
	return ss
	
def points(lx = [2.0, 10.0], ly = [2.0, 10.0], lx_ = [0, 1.0], ly_ = [0, 1.0], m = 25, n = 25):
	l1, l2, l3 = [], [], []
	d = {}
	d['m'], d['n'] = m, n
	sx = (lx[1]-lx[0])/(m-1)
	sy = (ly[1]-ly[0])/(n-1)
	x_, y_ = lx[0], ly[0]
	for i in range(m):
		y_ = ly[0]
		for j in range(n):
			if ((x_<=lx_[1]) and (x_>=lx_[0])) and ((y_<=ly_[1]) and (y_>=ly_[0])):
				z_ = 0
			else:			
				z_ = integral(x_, y_)
			l1.append(x_)
			l2.append(y_)
			l3.append(z_)
			y_ = y_+sy
		x_ = x_+sx
	d['x'], d['y'], d['z'] = l1, l2, l3
	return d
	

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
		
