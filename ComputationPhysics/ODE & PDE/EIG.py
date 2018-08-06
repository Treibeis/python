import numpy as np
import matplotlib.pyplot as plt
import math
import mpl_toolkits.mplot3d

def vec_T(l, x, y):
	out = 2*np.sin(l[0]*math.pi*x)*np.sin(l[1]*math.pi*y)
	return out

def E_T(mx = 19, my = 19):
	ls = []
	for i in range(mx):
		for j in range(my):
			o = ((i+1)**2+(j+1)**2)*math.pi**2
			ls.append((o, i+1, j+1))
	raw = sorted(ls, key=lambda ls:ls[0])
	eig, I_J = [], []
	for i in range(mx*my):
		eig.append(raw[i][0])
		I_J.append([raw[i][1],raw[i][2]])
	out = {}
	out['eig'], out['ij'] = eig, I_J
	return out

def norm(l, n):
	s = 0
	for i in range(n):
		s = s+l[i]**2
	return [x/s**0.5 for x in l]

def regular(b, mx, my):
	bt = b[1].T
	ls = []
	n = mx*my
	for i in range(n):
		if bt[i][mx+1]<0:
			bt[i] = [-x for x in bt[i]]
		bt[i] = norm(bt[i], mx*my)
		ls.append((b[0][i], bt[i]))
	out = sorted(ls, key=lambda ls:ls[0])
	return out

def EIG(Nx = 20, Ny = 20, to = 0.001, cut = 80, lx = [0.0, 1.0], ly = [0.0, 1.0]):
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
	lu = []
	ld = []
	out = {}
	eig = []
	c = regular(b, mx, my)
	for j in range(mx*my):
		u = np.zeros(ny*nx)
		for i in range(ny):
			u[nx*i], u[nx*i+Nx] = 0.0, 0.0
		for k in range(nx):
			u[k], u[nx*Ny+k] = 0.0, 0.0
		for i in range(my):
			for k in range(mx):
				u[(i+1)*nx+k+1] = c[j][1][i*mx+k]/(hx*hy)**0.5
		d = {}
		d['x'], d['y'], d['z'], d['m'], d['n'], d['eig'] = llx, lly, u, nx, ny, c[j][0]
		ld.append(d)
		eig.append(c[j][0])
	d0 = E_T(mx, my)
	error = 0
	num = 0
	lj = []
	ldt = []
	for j in range(mx*my):
		u0 = np.zeros(nx*ny)
		for i in range(ny):
			for k in range(nx):
				u0[i*nx+k] = vec_T(d0['ij'][j], k*hx, i*hy)
		er = np.dot(u0, ld[j]['z'])*hx*hy
		if (1-to<er) and (er<1+to):
			error = error+(er-1)**2
			num = num+1
			lj.append([d0['ij'][j],j])
		d = {}
		d['x'], d['y'], d['z'], d['m'], d['n'], d['eig'] = llx, lly, u0, nx, ny, d0['eig'][j]
		ldt.append(d)
	s = 0
	lre = []
	for j in range(mx*my):
		re = (d0['eig'][j]-eig[j])/d0['eig'][j]
		s = s+re
		if j<cut:
			lre.append(re)
	out['d_n'], out['d_t'], out['eig_n'], out['eig_t'], out['e1'], out['ij'], out['num'], out['e2'], out['re'] = ld, ldt, eig, d0['eig'], (error/num)**0.5, lj, num, s/mx/my, lre
	return out

				
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



				
