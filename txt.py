import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import mpl_toolkits.mplot3d

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))

plt.style.use('test2')

def restr(s, k = 0):
	out = []
	j = 0
	with open(s, 'r') as f:
		for line in f:
			lst = line.split()
			if j<k:
				a=1#print (lst[0])
			else:
				out.append(lst)
			j = j+1
	return out

def refloat(s, k = 0):
	out = []
	j = 0
	with open(s, 'r') as f:
		for line in f:
			lst = line.split()
			if j<k:
				a=1#print (lst[0])
			else:
				out.append(np.array([float(x) for x in lst]))
			j = j+1
	return out

def retxt(s, n, k = 0, t = 0):
	out = []
	for i in range(n):
		out.append([])
	j = 0
	with open(s, 'r') as f:
		for line in f:
			lst = line.split()
			if j<k:
				a=1#print (lst[0])
			else:
				for i in range(n):
					out[i].append(float(lst[i]))
			j = j+1
	if t!=0:
		for i in range(n):
			out[i].reverse()
	out[i] = np.array(out[i])
	return out

def retxt_nor(s, n, k = 1, t = 1, ind = 0, pre0 = 0):
	out = []
	for i in range(n):
		out.append([])
	j = 0
	pre = pre0
	with open(s, 'r') as f:
		for line in f:
			lst = line.split()
			if j<k:
				a=1#print (lst[0])
			else:
				if float(lst[ind]) >= pre:
					for i in range(n):
						out[i].append(float(lst[i]))
					pre = float(lst[ind])
			j = j+1
	if t!=0:
		for i in range(n):
			out[i].reverse()
	out[i] = np.array(out[i])
	return out

def totxt(s, l, ls = 0, t = 0, k = 0):
	j = 0
	with open(s, 'w') as f:
		if t!=0:
			for r in range(len(ls)):
				f.write(ls[r])
				f.write(' ')
			f.write('\n')
		for i in range(len(l[0])):
			if j<k:
				print (l[0][i])
			else:
				for s in range(len(l)):
					f.write(str(l[s][i]))
					f.write(' ')
				f.write('\n')
			j = j+1

def torow(s, l, k=0):
	j = 0
	with open(s, 'w') as f:
		for row in l:
			if j<k:
				print(row)
			else:
				for d in row:
					f.write(str(d))
					f.write(' ')
				f.write('\n')
			j = j+1

"""
def plot3d(t=[],v=[],title=1,l = [0,162,0,162],x='x',y='y',z='z'):
	ax=plt.subplot(111,projection='3d')
	ax.scatter(t[0],t[1],t[2],marker='*',color='r')
	ax.scatter(v[0],v[1],v[2],marker='.',color='b')
	ax.set_xlabel(r(x),size=18.0)
	ax.set_ylabel(r(y),size=18.0)
	ax.set_zlabel(r(z),size=18.0)
	if title==1:
		ax.set_title('Distribution of tracers and voids in Lambda')
	else:
		ax.set_title('Distribution of tracers and voids in R-P')
	ax.axis(l)
	

def read(s1 = 'L', i = 0, n = 8, d = 2, s2='NF', s3='.txt', k = 0, t = 0, fill = 0, fn = 4):
	if (fill==0):
		l1 = np.array([retxt(s2+'_'+s1+'_'+str(x+1)+s3, d, k, t)[i] for x in range(n)])
	else:
		s4 = str(i+1).zfill(fn)
		l1 = np.array([retxt(s2+'_'+s1+'_'+str(x+1)+s3, d, k, t)[i] for x in range(n)])
	#l2 = np.array([np.cumsum(l) for l in l1])
	return l1


def r(s=r'R_{V}'):
	return r'$' + s + r'[\mathrm{Mpc}\cdot h^{-1}]$'

def n1(b = 1):
	b = str(b)
	return r'$N_{V}\left(bin='+b+r'\ \mathrm{Mpc}\cdot h^{-1}\right)$'

def n2():
	return r'$N_{V}\left(R_{V}\leq R\right)$'

def nd(l, V = 1296.0**3):
	n = len(l[0])
	ly = []
	lx = []
	for i in range(n-2):
		a = l[1][i+3]-l[1][i+1]/(l[0][i+3]-l[0][i+1])
		b = l[0][i+2]*a
		ly.append(b/V)
		lx.append(l[0][i+2])
	return [lx, ly]


def std(l):
	m = len(l)
	n = len(l[0])
	average = []
	for i in range(n):
		s = 0
		for j in range(m):
			s += l[j][i]
		average.append(s/float(m))
	std = []
	for i in range(n):
		s = 0
		for j in range(m):
			s += (l[j][i]-average[i])**2
		std.append((s/(m))**0.5)
	lu, ll, lSN = [], [], []
	for i in range(n):
		lu.append(average[i]+std[i])
		ll.append(average[i]-std[i])
	#	print (ll[i])
		if std[i]!=0:
			lSN.append(abs(average[i]/std[i]))
		else:
			lSN.append(0.0)
	Aa = np.add.accumulate(average)
	Aup = np.add.accumulate(lu)
	Alow = np.add.accumulate(ll)
	d = {}
	d['a'], d['up'], d['low'], d['e'], d['std'], d['SN'], d['l'] = np.array(average), np.array(lu), np.array(ll), std, sum(std)/n, lSN, np.array([average, ll, lu, Aa, Alow, Aup, std, lSN])
	return d


def ratio(lr, l1, l2):
	m, n = len(l1), len(l2)
	out = [[], []]
	if n>m:
		a, b = n, m
	else:
		a, b = m, n
	for i in range(b):
		if (l2[i]!=0) and (l1[i]!=0):
			out[1].append(l1[i]/l2[i])
			out[0].append(lr[i])
	return out

def extract(l1, l2):
	la1, ll1, lu1, la2, ll2, lu2 = l1[0], l1[1], l1[2], l2[0], l2[1], l2[2]
	m, n = len(la1), len(la2)
	out = [[], [], []]
	if n>m:
		a, b = n, m
	else:
		a, b = m, n	
	for i in range(b):
		out[0].append(la1[i]-la2[i])
		out[1].append(ll1[i]-lu2[i])
		out[2].append(lu1[i]-ll2[i])
	return out


def stdio():
	return ['radius', 'aver', 'low', 'up', 'A_aver', 'A_low', 'A_up', 'std', 'S/N']
"""

"""
UL0 = 3.085678e24
Myr = 365.0*24*3600*10**6
tz0 = retxt('/home/friede/python/t_z_Planck.txt',2,0,0)
tcos = tz0[0][0]

tz = interp1d(tz0[1],tz0[0])
zt = interp1d(tz0[0],tz0[1])
D_Z = retxt('/home/friede/python/d_z_Planck.txt',2,0,0)
dz = interp1d(D_Z[1],D_Z[0])
zd = interp1d(D_Z[0],D_Z[1])


def f1(x):
	return x-1296.0

def modify(l, lm, f = f1):
	out = []
	for k in range(len(l)):
		out.append([])
	for i in range(len(l[0])):
		for j in range(len(l)):
			if lm[j]==1:
				out[j].append(f(l[j][i]))
			else:
				out[j].append(l[j][i])
	return out
"""
