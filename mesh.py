from HMC import *
from com_red_HMC import *

Tlist = [7.4893436088, 7.34968676055, 7.17056916489, 6.92443239074, 6.85329808398, 6.65483994532, 6.49941956572, 6.29488971946, 6.07869598583, 5.83977424921]

DX_, DY_, DZ_ = 10, 16, 16

def integ_x(me, unit = UL, background = 0.0):
	l0 = me['head']
	x0, x1, xbin, y0, y1, ybin, z0, z1, zbin = l0[0], l0[1], l0[2], l0[3], l0[4], l0[5], l0[6], l0[7], l0[8]
	dx, dy, dz = (x1-x0)/(xbin), (y1-y0)/(ybin), (z1-z0)/(zbin)
	yref = np.linspace(y0+dy/2, y1-dy/2, ybin)
	zref = np.linspace(z0+dz/2, z1-dz/2, zbin)
	Y, Z = np.meshgrid(yref, zref)
	final = np.array([np.ones(ybin, dtype='float')*background for z in range(zbin)])
	for i in range(zbin):
		for j in range(ybin):
			result = np.sum(me['l'][3][i][j]*dx*unit)
			if result>=background:
				final[i][j] = result
	return [Y, Z, final]

def mesh3d(lx, ly, lz, ld, x0, x1, xbin, y0, y1, ybin, z0, z1, zbin, background = 0.0, style = 0):
	dx, dy, dz = (x1-x0)/(xbin), (y1-y0)/(ybin), (z1-z0)/(zbin)
	xref = np.linspace(x0+dx/2, x1-dx/2, xbin)
	yref = np.linspace(y0+dy/2, y1-dy/2, ybin)
	zref = np.linspace(z0+dz/2, z1-dz/2, zbin)
	X, Y, Z = np.meshgrid(xref, yref, zref)
	data = np.array(np.matrix([lx, ly, lz, ld]).transpose())
	out = [[[[] for x in range(xbin)] for y in range(ybin)] for z in range(zbin)]
	oz = [[] for x in range(zbin)]
	ozy = [[[] for x in range(ybin)] for y in range(zbin)]
	data = sorted(data, key=lambda data:data[2])
	cz = 0
	for i in range(len(data)):
		if (data[i][2]>=zref[cz]-dz/2)and(data[i][2]<zref[cz]+dz/2):
			oz[cz].append(data[i])
		else:
			if data[i][2]>=zref[cz]+dz/2:
				cz += 1
		if cz==zbin:
			break
	for j in range(zbin):
		buf = np.array(oz[j])
		buf = sorted(buf, key=lambda buf:buf[1])
		cy = 0
		for s in range(len(buf)):
			if (buf[s][1]>=yref[cy]-dy/2)and(buf[s][1]<yref[cy]+dy/2):
				ozy[j][cy].append(buf[s])
			else:
				if buf[s][1]>=yref[cy]+dy/2:
					cy += 1
			if cy==ybin:
				break
	for k in range(zbin):
		for l in range(ybin):
			buf = np.array(ozy[k][l])
			buf = sorted(buf, key=lambda buf:buf[0])
			cx = 0
			for s in range(len(buf)):
				if (buf[s][0]>=xref[cx]-dx/2)and(buf[s][0]<xref[cx]+dx/2):
					out[k][l][cx].append(buf[s])
				else:
					if buf[s][0]>=xref[cx]+dx/2:
						cx += 1
				if cx==xbin:
					break
	final = np.array([[np.ones(xbin, dtype='float')*background for y in range(ybin)] for z in range(zbin)])
	for z in range(zbin):
		for y in range(ybin):
			for x in range(xbin):
				if len(out[z][y][x])>0:
					if style==0:
						final[z][y][x] = np.sum([out[z][y][x][k][3] for k in range(len(out[z][y][x]))])/dz/dy/dx
					elif style==1:
						final[z][y][x] = np.average([out[z][y][x][k][3] for k in range(len(out[z][y][x]))])
					else:
						final[z][y][x] = np.std([out[z][y][x][k][3] for k in range(len(out[z][y][x]))])
	head = [x0, x1, xbin, y0, y1, ybin, z0, z1, zbin]
	return {'l':[X, Y, Z, final],'head':head}

def mesh2d(lx, ly, lz, x0, x1, xbin, y0, y1, ybin, background = 10.0, style = 0):
	dx, dy = (x1-x0)/(xbin), (y1-y0)/(ybin)
	xref = np.linspace(x0+dx/2, x1-dx/2, xbin)
	yref = np.linspace(y0+dy/2, y1-dy/2, ybin)
	X, Y = np.meshgrid(xref, yref)
	data = np.array(np.matrix([lx, ly, lz]).transpose())
	out = [[[] for x in range(xbin)] for y in range(ybin)]
	oy = [[] for y in range(ybin)]
	data = sorted(data, key=lambda data:data[1])
	cy = 0
	for i in range(len(data)):
		if (data[i][1]>=yref[cy]-dy/2)and(data[i][1]<yref[cy]+dy/2):
			oy[cy].append(data[i])
		else:
			if data[i][1]>=yref[cy]+dy/2:
				cy += 1
		if cy==ybin:
			break
	for j in range(ybin):
		buf = np.array(oy[j])
		buf = sorted(buf, key=lambda buf:buf[0])
		cx = 0
		for s in range(len(buf)):
			if (buf[s][0]>=xref[cx]-dx/2)and(buf[s][0]<xref[cx]+dx/2):
				out[j][cx].append(buf[s])
			else:
				if buf[s][0]>=xref[cx]+dx/2:
					cx += 1
			if cx==xbin:
				break
	final = np.array([np.ones(xbin, dtype='float')*background for y in range(ybin)])
	for y in range(ybin):
		for x in range(xbin):
			if len(out[y][x])>0:
				if style==0:
					final[y][x] = np.sum([out[y][x][k][2] for k in range(len(out[y][x]))])/dx/dy
				elif style==1:
					final[y][x] = np.average([out[y][x][k][2] for k in range(len(out[y][x]))])
				else:
					final[y][x] = np.std([out[y][x][k][2] for k in range(len(out[y][x]))])
	return [X, Y, final]

def mesh4(d, sq, scale = 1, fac = 3, res = [19, 19, 19]): #size = [2.5, 2.5, 2.5], centre = [BOXSIZE/2, BOXSIZE/2, BOXSIZE/2]):
	gas = d['part'][sq][0]
	dark = d['part'][sq][1]
	velc = (np.array([np.nanmean(d['part'][sq][1][3+x]) for x in range(3)])*(OMEGA0-OMEGA_B)*len(d['part'][sq][1][0])+ np.array([np.nanmean(d['part'][sq][0][3+x]) for x in range(3)])*OMEGA_B*len(d['part'][sq][0][0]))/((OMEGA0-OMEGA_B)*len(d['part'][sq][1][0])+OMEGA_B*len(d['part'][sq][0][0]))
	centre = [d['info'][sq][3+x]+Shift[x] for x in range(3)]
	rv = Virial(d['info'][sq][2], 1/scale-1)
	size = [fac*rv for x in range(3)]
	print(centre, size)
	v0 = [mesh3d(gas[0], gas[1], gas[2], gas[3+x]-velc[x], centre[0]-size[0],centre[0]+size[0],res[0],centre[1]-size[1],centre[1]+size[1],res[1],centre[2]-size[2],centre[2]+size[2],res[2],0,1) for x in range(3)]
	v1 = [mesh3d(dark[0], dark[1], dark[2], dark[3+x]-velc[x], centre[0]-size[0],centre[0]+size[0],res[0],centre[1]-size[1],centre[1]+size[1],res[1],centre[2]-size[2],centre[2]+size[2],res[2],0,1) for x in range(3)]
	base = [v0[0]['l'][0], v0[0]['l'][1], v0[0]['l'][2]]
	out = [base+[v0[x]['l'][3] for x in range(3)], base+[v1[x]['l'][3] for x in range(3)]]
	return out

def mesh3(d0, z = 0, T = 10, lim = 0.0, res = 40, dx=6.0, dy=6.0, dz=10.0):
	l0 = convert2(d0['l'][0][0])
	#l1 = convert1(d0['l'][0][1])
	bac = SZfac((1+z)**3*CRITICAL(0)*OMEGA_B/1000, T/1000)
	#bac = 0
	m3SZ = mesh3d(DZ(l0['l'][2]),l0['l'][0],l0['l'][1],SZfac(l0['l'][9],l0['l'][11]),NORM-dz/2,NORM+dz/2,res,360-dx+PHI,360+PHI+dx,60,-dy+THETA,THETA+dy,60,bac,1)
	m2SZ = integ_x(m3SZ, UL, lim)
	return m2SZ
	
def mesh2(d0, dx=1.0, dy=1.0):
	l0 = convert2(d0['l'][0][0])
	l1 = convert2(d0['l'][0][1])
	mv0 = mesh2d(l0['l'][0],l0['l'][1],projection(l0['l'][0],l0['l'][1],l0['l'][3],l0['l'][4],l0['l'][5]),360-dx+PHI,360+PHI+dx,60,-dy+THETA,THETA+dy,60,0,1)
	mv0d = mesh2d(l0['l'][0],l0['l'][1],projection(l0['l'][0],l0['l'][1],l0['l'][3],l0['l'][4],l0['l'][5]),360-dx+PHI,360+PHI+dx,60,-dy+THETA,THETA+dy,60,0,2)
	mv1 = mesh2d(l1['l'][0],l1['l'][1],projection(l1['l'][0],l1['l'][1],l1['l'][3],l1['l'][4],l1['l'][5]),360-dx+PHI,360+PHI+dx,60,-dy+THETA,THETA+dy,60,0,1)
	mv1d = mesh2d(l1['l'][0],l1['l'][1],projection(l1['l'][0],l1['l'][1],l1['l'][3],l1['l'][4],l1['l'][5]),360-dx+PHI,360+PHI+dx,60,-dy+THETA,THETA+dy,60,0,2)
	mB = mesh2d(l0['l'][0],l0['l'][1],Bremsstrahlung(l0['l'][9],l0['l'][11],l0['l'][7]/l0['l'][9])/(np.pi/180)**2,360-dx+PHI,360+dx+PHI,60,-dy+THETA,dy+THETA,60,0,0)
	mB[2] = mB[2]/np.cos(mB[1]*np.pi/180)
	mt = mesh2d(l0['l'][0],l0['l'][1],l0['l'][11],360-dx+PHI,360+PHI+dx,60,-dy+THETA,THETA+dy,60,10,1)
	mtd = mesh2d(l0['l'][0],l0['l'][1],l0['l'][11],360-dx+PHI,360+PHI+dx,60,-dy+THETA,THETA+dy,60,10,2)
	out = [mv0, mv0d, mv1, mv1d, mt, mtd, mB]
	for i in range(7):
		out[i][0] = out[i][0]-39
	return out

def mesh1(d0, z, dx=5.0, dy=8.0):
	md0=mesh2d(d0['l'][0][0][0],d0['l'][0][0][2],d0['l'][0][0][7],-dx,dx,80,-dy,dy,80,(1+z)**3*DY_*OMEGA_B*CRITICAL(z)*matter(z)/OMEGA0,0)
	md1=mesh2d(d0['l'][0][1][0],d0['l'][0][1][2],d0['l'][0][1][7],-dx,dx,80,-dy,dy,80,(1+z)**3*DY_*(OMEGA0-OMEGA_B)*matter(z)*CRITICAL(z)/OMEGA0,0)
	mx0=mesh2d(d0['l'][0][0][0],d0['l'][0][0][2],d0['l'][0][0][3],-dx,dx,40,-dy,dy,40,0,1)
	mx1=mesh2d(d0['l'][0][1][0],d0['l'][0][1][2],d0['l'][0][1][3],-dx,dx,40,-dy,dy,40,0,1)
	my0=mesh2d(d0['l'][0][0][0],d0['l'][0][0][2],d0['l'][0][0][4],-dx,dx,40,-dy,dy,40,0,1)
	mz0=mesh2d(d0['l'][0][0][0],d0['l'][0][0][2],d0['l'][0][0][5],-dx,dx,40,-dy,dy,40,0,1)
	my1=mesh2d(d0['l'][0][1][0],d0['l'][0][1][2],d0['l'][0][1][4],-dx,dx,40,-dy,dy,40,0,1)
	mz1=mesh2d(d0['l'][0][1][0],d0['l'][0][1][2],d0['l'][0][1][5],-dx,dx,40,-dy,dy,40,0,1)
	mt=mesh2d(d0['l'][0][0][0],d0['l'][0][0][2],d0['l'][0][0][11],-dx,dx,80,-dy,dy,80,10,1)
	mB=mesh2d(d0['l'][0][0][0],d0['l'][0][0][2],Bremsstrahlung(d0['l'][0][0][9],d0['l'][0][0][11],d0['l'][0][0][7]/d0['l'][0][0][9])/UL**2,-dx,dx,80,-dy,dy,80,1/10**16,0)
	a1 = [md0, md1, mx0, my0, mz0, mx1, my1, mz1, mt, mB]
	md0=mesh2d(d0['l'][0][0][0],d0['l'][0][0][1],d0['l'][0][0][7],-dx,dx,80,-dy,dy,80,(1+z)**3*DZ_*OMEGA_B*CRITICAL(z)*matter(z)/OMEGA0,0)
	md1=mesh2d(d0['l'][0][1][0],d0['l'][0][1][1],d0['l'][0][1][7],-dx,dx,80,-dy,dy,80,(1+z)**3*DZ_*(OMEGA0-OMEGA_B)*matter(z)*CRITICAL(z)/OMEGA0,0)
	mx0=mesh2d(d0['l'][0][0][0],d0['l'][0][0][1],d0['l'][0][0][3],-dx,dx,40,-dy,dy,40,0,1)
	mx1=mesh2d(d0['l'][0][1][0],d0['l'][0][1][1],d0['l'][0][1][3],-dx,dx,40,-dy,dy,40,0,1)
	my0=mesh2d(d0['l'][0][0][0],d0['l'][0][0][1],d0['l'][0][0][4],-dx,dx,40,-dy,dy,40,0,1)
	mz0=mesh2d(d0['l'][0][0][0],d0['l'][0][0][1],d0['l'][0][0][5],-dx,dx,40,-dy,dy,40,0,1)
	my1=mesh2d(d0['l'][0][1][0],d0['l'][0][1][1],d0['l'][0][1][4],-dx,dx,40,-dy,dy,40,0,1)
	mz1=mesh2d(d0['l'][0][1][0],d0['l'][0][1][1],d0['l'][0][1][5],-dx,dx,40,-dy,dy,40,0,1)
	mt=mesh2d(d0['l'][0][0][0],d0['l'][0][0][1],d0['l'][0][0][11],-dx,dx,80,-dy,dy,80,10,1)
	mB=mesh2d(d0['l'][0][0][0],d0['l'][0][0][1],Bremsstrahlung(d0['l'][0][0][9],d0['l'][0][0][11],d0['l'][0][0][7]/d0['l'][0][0][9])/UL**2,-dx,dx,80,-dy,dy,80,1/10**16,0)
	a2 = [md0, md1, mx0, my0, mz0, mx1, my1, mz1, mt, mB]
	msp0=mesh2d(d0['l'][0][0][0],d0['l'][0][0][3],np.ones(len(d0['l'][0][0][3])),-dx,dx,80,-2000,2000,80,0,0)
	msp1=mesh2d(d0['l'][0][1][0],d0['l'][0][1][3],np.ones(len(d0['l'][0][1][3])),-dx,dx,80,-2000,2000,80,0,0)
	return [a2, a1, [msp0, msp1]]

def mesh0(d0, z, dx=8.0, dy=8.0):
	md0=mesh2d(d0['l'][0][0][1],d0['l'][0][0][2],d0['l'][0][0][7],-dx,dx,80,-dy,dy,80,(1+z)**3*DX_*OMEGA_B*CRITICAL(z)*matter(z)/OMEGA0,0)
	md1=mesh2d(d0['l'][0][1][1],d0['l'][0][1][2],d0['l'][0][1][7],-dx,dx,80,-dy,dy,80,(1+z)**3*DX_*(OMEGA0-OMEGA_B)*matter(z)*CRITICAL(z)/OMEGA0,0)
	mx0=mesh2d(d0['l'][0][0][1],d0['l'][0][0][2],-d0['l'][0][0][3],-dx,dx,40,-dy,dy,40,0,1)
	mx1=mesh2d(d0['l'][0][1][1],d0['l'][0][1][2],-d0['l'][0][1][3],-dx,dx,40,-dy,dy,40,0,1)
	my0=mesh2d(d0['l'][0][0][1],d0['l'][0][0][2],d0['l'][0][0][4],-dx,dx,40,-dy,dy,40,0,1)
	mz0=mesh2d(d0['l'][0][0][1],d0['l'][0][0][2],d0['l'][0][0][5],-dx,dx,40,-dy,dy,40,0,1)
	my1=mesh2d(d0['l'][0][1][1],d0['l'][0][1][2],d0['l'][0][1][4],-dx,dx,40,-dy,dy,40,0,1)
	mz1=mesh2d(d0['l'][0][1][1],d0['l'][0][1][2],d0['l'][0][1][5],-dx,dx,40,-dy,dy,40,0,1)
	mt=mesh2d(d0['l'][0][0][1],d0['l'][0][0][2],d0['l'][0][0][11],-dx,dx,80,-dy,dy,80,10,1)
	mB=mesh2d(d0['l'][0][0][1],d0['l'][0][0][2],Bremsstrahlung(d0['l'][0][0][9],d0['l'][0][0][11],d0['l'][0][0][7]/d0['l'][0][0][9])/UL**2,-dx,dx,80,-dy,dy,80,1/10**16,0)
	return [md0, md1, mx0, my0, mz0, mx1, my1, mz1, mt, mB]
