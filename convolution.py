import math
import random


def createx(p1 = 0.0, p2 = 1.0, a = 1.0, b = 1.0):
	c = a/(a+b)
	f = random.uniform(0.0,1.0)
	if f>c:
		return p2
	else:
		return p1

def gauss(mu = 0.0, sigma = 1.0, n = 200, r = 10, a = 1.0):
	l1 = []
	step, y = float(r/n), mu-float(r/2)
	for i in range(n):
		l1.append(a*math.exp(-(y-mu)**2/2/sigma**2)/sigma/math.pow(2*math.pi, 0.5))
		y = y+step
	return l1

def convolution(n = 1000, mu = 0.0, sigma = 1.0, p1 = 0.0, p2 = 1.0, a = 1.0, b = 1.0):
	l1 = []
	l2 = []
	for i in range(n):
		x = createx(p1,p2,a,b)
		y = random.gauss(mu, sigma)
		z = x+y
		l1.append(y)
		l2.append(z)
	out = {'x':l1, 'y':l2, 'n':n}
	return out

def distribution(d1, mu = 0.0, sigma = 1.0, x1 = -100.0, x2 = 100.0, n = 50, l = 0):
	y1, y2 = x1+mu-5*sigma, x2+mu+5*sigma
	r, step, a = (y2-y1), (y2-y1)/n, y1
	l1 = []
	l2 = []
	if l==0:
		for i in range(n):
			l2.append(0)
			l1.append(a)
			for x in d1['y']:
				if (x<a+step) and (x>=a):
					l2[i] = l2[i]+1.0
			l2[i] = l2[i]/float(d1['n'])/step
			a = a+step
	elif l==1:
		for i in range(n):
			l2.append(0)
			l1.append(a)
			for x in d1['x']:
				if (x<a+step) and (x>=a):
					l2[i] = l2[i]+1.0
			l2[i] = l2[i]/float(d1['n'])/step
			a = a+step
	else:
		print('Input Error.\n')
		return d1
	d = {'x':l1, 'y':l2, 'n':n, 'r':r}
	return d

def uni_noise(d1, a = 1.0):
	l1 = []
	l2 = []
	for i in range(d1['n']):
		f = a*random.uniform(-0.5,0.5)
		l2.append(d1['y'][i] + f)
		l1.append(d1['x'][i])
	d = {'x':l1, 'y':l2, 'n':d1['n']}
	return d
