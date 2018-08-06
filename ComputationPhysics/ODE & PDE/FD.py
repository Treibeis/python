import numpy as np
import matplotlib.pyplot as plt
import math
import mpl_toolkits.mplot3d
from matplotlib import animation

def ini(x):
	if (x>=0) and (x<=0.5):
		out = 2.0*x
	elif (x>=0.5) and (x<=1):
		out = 2.0-2.0*x
	else:
		#print("Error: out of range.\n")
		out = 0.0
	return out

def FD(Nt = 50, lt = [0.0, 0.06], Nx = 20, lx = [0.0, 1.0]):
	nt, nx = Nt+1, Nx+1
	hx, ht = (lx[1]-lx[0])/Nx, (lt[1]-lt[0])/Nt
	llx, llt = [], []
	t = 0
	for i in range(nt):
		x = 0
		for j in range(nx):
			llx.append(x)
			x = x+hx
			llt.append(t)
		t = t+ht
	u = np.zeros(nt*nx)
	a = np.zeros((Nx-1,Nx-1))
	k = 1
	while (k>=1) and (k<=Nx-1):
		u[k] = ini(llx[k])
		k = k+1
	for i in range(Nt):
		u[nx*i], u[nx*i+Nx] = 0.0, 0.0
		k = 1
		while (k>=1) and (k<=Nx-1):
			u[(i+1)*nx+k] = ht*(u[i*nx+k+1]+u[i*nx+k-1]-2*u[nx*i+k])/hx**2 + u[nx*(i)+k]
			if k>1:
				a[k-1][k-2] = ht/hx**2
			if k<Nx-1:
				a[k-1][k] = ht/hx**2
			a[k-1][k-1] = 1-2*ht/hx**2
			k = k+1
	d = {}
	d['x'], d['y'], d['z'], d['m'], d['n'], d['a'], d['eig'] = llx, llt, u, nx, nt, a, np.linalg.eig(a)
	return d

def IFD(r = 0, Nt = 12, lt = [0.0, 0.06], Nx = 20, lx = [0.0, 1.0]):
	nt, nx = Nt+1, Nx+1
	hx, ht = (lx[1]-lx[0])/Nx, (lt[1]-lt[0])/Nt
	llx, llt = [], []
	t = 0
	for i in range(nt):
		x = 0
		for j in range(nx):
			llx.append(x)
			x = x+hx
			llt.append(t)
		t = t+ht
	u = np.zeros(nt*nx)
	k = 1
	while (k>=1) and (k<=Nx-1):
		u[k] = ini(llx[k])
		k = k+1
	a = np.zeros((Nx-1,Nx-1))
	aa = np.zeros((Nx-1,Nx-1))
	for k in range(Nx-1):
		if k>0:
			a[k][k-1] = -ht/hx**2
			aa[k][k-1] = ht/hx**2
		if k<Nx-2:
			a[k][k+1] = -ht/hx**2
			aa[k][k+1] = ht/hx**2
		if r==0:
			a[k][k] = 2.0*ht/hx**2+1.0
		elif r==1:
			a[k][k] = 2.0*ht/hx**2+2.0
		else:
			print('Input Error.\n')
		aa[k][k] = 2.0-2.0*ht/hx**2
	for i in range(Nt):
		b = np.zeros(Nx-1)
		for k in range(Nx-1):
			if r==0:
				b[k] = u[i*nx+k+1]
			elif r==1:
				b[k] = 2.0*u[i*nx+k+1] + (u[i*nx+k+2]+u[i*nx+k])*ht/hx**2 - 2*u[i*nx+k+1]*ht/hx**2
			else:
				print('Input Error.\n')
		c = np.dot(np.linalg.inv(a),b)
		for k in range(Nx-1):
			u[(i+1)*nx+k+1] = c[k]
	a1 = np.linalg.inv(a)
	a2 = np.dot(np.linalg.inv(a),aa)
	d = {}
	d['x'], d['y'], d['z'], d['m'], d['n'], d['a1'], d['a2'], d['eig1'], d['eig2'] = llx, llt, u, nx, nt, a1, a2, np.linalg.eig(a1), np.linalg.eig(a2)
	return d

# The codes below are meant for graphics
				
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

def plot2d(d = FD(), r = 1):
		m, n = d['m'], d['n']-1
		k = n/r
		print(k)
		for i in range(k):
			if (i==0) or (i==n-1):
				plt.plot(d['x'][i*m*r:i*m*r+m], d['z'][i*m*r:i*m*r+m],ls='--',lw=2)
			else:
				plt.plot(d['x'][i*m*r:i*m*r+m], d['z'][i*m*r:i*m*r+m])
		if k!=n:
			plt.plot(d['x'][(n-1)*m:n*m], d['z'][(n-1)*m:n*m],ls='--',lw=2)
		plt.xlabel(r'$x$',size=18.0)
		plt.ylabel(r'$u$',size=18.0)
	
def plot3d(d = FD()):
	dd = trans(d)
	ax = plt.subplot(111,projection='3d')
	ax.plot_surface(dd['x'],dd['y'],dd['z'],rstride=5,cstride=2,cmap=plt.cm.coolwarm, alpha=0.8)
	ax.set_xlabel(r'$x$',size=18.0)
	ax.set_ylabel(r'$t$',size=18.0)
	ax.set_zlabel(r'$u$',size=18.0)

fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot([],[])

def init():
	line.set_data([], [])
	return line,

def animate0(i):
	d = FD()
	m, n = d['m'], d['n']
	x = d['x'][i*m:(i+1)*m]
	y = d['z'][i*m:(i+1)*m]
	line.set_data(x, y)
	return line,

def ani(f = 10, i = 51):
	animation.FuncAnimation(fig, animate0, init_func=init, frames=f, interval=i)
	



				
