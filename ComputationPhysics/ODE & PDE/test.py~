import numpy as np
import matplotlib.pyplot as plt
import math
import mpl_toolkits.mplot3d

def FD(Nx = 20, Ny = 20, lx = [0.0, 1.0], ly = [0.0, 1.0]):
	nx, ny = Nx+1, Ny+1
	mx, my = Nx-1, Ny-1
	hx, hy = (lx[1]-lx[0])/Nx, (ly[1]-ly[0])/Ny
	llx, lly = [], []
	a = np.zeros((mx*my,mx*my))
	y = 0
	for i in range(ny):
		x = 0
		for j in range(nx):
			llx.append(x)
			x = x+hx
			lly.append(y)
		y = y+hy
	for i in range(my):
		for k in range(mx):
			a[i*mx+k][i*mx+k] = 2.0/hx**2+2.0/hy**2
			if i>0:
				a[i*mx+k][(i-1)*mx+k] = -1.0/hy**2
			if i<my-1:
				a[i*mx+k][(i+1)*mx+k] = -1.0/hy**2
			if k>0:
				a[i*mx+k][i*mx+k-1] = -1.0/hx**2
			if k<mx-1:
				a[i*mx+k][i*mx+k+1] = -1.0/hx**2
	b = np.linalg.eig(a)
	d = {}
	d['a'], d['b'] = a, b
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

def plot3d(d = {}):
	dd = trans(d)
	ax = plt.subplot(111,projection='3d')
	ax.plot_surface(dd['x'],dd['y'],dd['z'],rstride=1,cstride=1,cmap=plt.cm.coolwarm, alpha=0.8)
	ax.set_xlabel(r'$x$',size=18.0)
	ax.set_ylabel(r'$y$',size=18.0)
	ax.set_zlabel(r'$\psi$',size=18.0)



				
