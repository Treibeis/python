from cosmology import *
from scipy.optimize import fsolve, curve_fit
from scipy.special import erf
from scipy.integrate import odeint, ode
from scipy.interpolate import interp1d
from txt import *
import multiprocessing as mp
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

foralpha = lambda x: ((x-np.sin(x))/(2.*np.pi))**(2./3)

def delta(z, zvir, dmax = 200.):
	f0 = (1.+zvir)/(1.+z)
	f = lambda x: foralpha(x)-f0
	alpha = fsolve(f, 1.)[0]
	d = 9.*(alpha-np.sin(alpha))**2. / (2.*(1.-np.cos(alpha))**3)
	return min(d-1, dmax)

def rho_z(z, vir, dmax = 200):
	return delta(z, vir, dmax) * rhom(1/(1+z))

def T_b(z, a1=1./119, a2=1./115, T0=2.726):
	a = 1./(1+z)
	return T0/(a*(1+a/(a1*(1+(a2/a)**1.5))))

def T_dm(z, m = 1., T0=2.726):
	zc = m*1e9*eV / ((3./2)*BOL*T0) - 1
	Tc = T0*(1+zc)
	return T0*(1+z) * (z>zc) + Tc*((1+z)/(1+zc))**2 * (z<=zc)

def Tdot_adiabatic(z, Om = 0.315, Ob = 0.048, OR = 9.54e-5, h = 0.6774, X = 0.76):
	mu = 4/(1+3*X)
	a = 1/(1+z)
	return -2*T_b(z)*H(a, Om, h, OR) #* 1.5*BOL*rhom(a, Om, h)*Ob/(PROTON*mu)

def Tdot(z, Om = 0.315, Ob = 0.048, OR = 9.54e-5, h = 0.6774, X = 0.76, a1=1./119, a2=1./115, T0=2.726):
	mu = 4/(1+3*X)
	a = 1/(1+z)
	Denorm = a**2 * (a+a1+a1*(a2/a)**1.5)**2
	Norm = a1*(1.5*a2*(a2/a)**0.5 + 2*a*(1+(a2/a)**1.5) \
		 + a1*(1+2*(a2/a)**1.5+(a2/a)**3))
	return - T0 * Norm/Denorm * a*H(a, Om, h, OR)  #* 1.5*BOL*rhom(a, Om, h)*Ob/(PROTON*mu)

GeV_to_mass = eV*1e9/SPEEDOFLIGHT**2	

def GammaC(z, Om = 0.315, Ob = 0.048, OR = 9.54e-5, h = 0.6774, X = 0.76, a1=1./119, a2=1./115, T0=2.726):
	a = 1/(1+z)
	dT0 = Tdot(z, Om, Ob, OR, h, X, a1, a2, T0)/(a*H(a, Om, h, OR))
	dT1 = -2*T_b(z, a1, a2, T0)/a
	gamma = (dT0-dT1)*a*H(a, Om, h, OR)/(T0/a-T_b(z, a1, a2, T0))
	return gamma

def vbdm_z(z, v0 = 30., z0 = 1100.):
	return v0*1e5*(1+z)/(1+z0)

def uthf(mb, mdm, Tb, Tdm):
	return (Tb*BOL/mb+Tdm*BOL/mdm)**0.5

def drag(rho, v, Tb, Tdm, mb = PROTON, mdm = 0.3*GeV_to_mass, sigma = 8e-20):
	uth = (Tb*BOL/mb+Tdm*BOL/mdm)**0.5
	r = v/uth
	return rho*sigma*1e20/(mb+mdm)*(erf(r/2**0.5)-(2/np.pi)**0.5*np.exp(-r**2/2)*r)/v**2

def Q_IDMB(rho, v, Tb, Tdm, mb = PROTON, mdm = 0.3*GeV_to_mass, sigma = 8e-20):
	uth = (Tb*BOL/mb+Tdm*BOL/mdm)**0.5
	r = v/uth
	c = (Tdm-Tb)/uth**3 * ((2/np.pi)**0.5*np.exp(-r**2/2))
	d = mdm/v * (erf(r/2**0.5)-(2/np.pi)**0.5*np.exp(-r**2/2)*r)/BOL
	out = mb*rho*sigma*1e20/(mdm+mb)**2 * (c + d)
	return out/1.5
	
def dv_z(z, v0 = 30., Mdm = 0.3, sigma = 8e-20, Om = 0.315, Ob = 0.048, OR = 9.54e-5, h = 0.6774, X = 0.76):
	a = 1/(1+z)
	mdm = Mdm*GeV_to_mass
	Tb = T_b(z)
	Tdm = T_dm(z, Mdm)
	rho = rhom(a, Om, h)
	xh = 4*X/(1+3*X)
	v = vbdm_z(z, v0)
	accH = drag(rho, v, Tb, Tdm, PROTON, mdm, sigma)*xh
	accHe = drag(rho, v, Tb, Tdm, 4*PROTON, mdm, sigma)*(1-xh)
	return (accH + accHe) * 1/H(a, Om, h, OR)

def Q_z(z, v, Mdm = 0.3, sigma = 8e-20, Om = 0.315, Ob = 0.048, OR = 9.54e-5, h = 0.6774, X = 0.76):
	a = 1/(1+z)
	mdm = Mdm*GeV_to_mass
	Tb = T_b(z)
	Tdm = T_dm(z, Mdm)
	rho = (Om-Ob)/Om * rhom(a, Om, h)
	xh = 4*X/(1+3*X)
	QH = Q_IDMB(rho, v, Tb, Tdm, PROTON, mdm, sigma)*xh
	QHe = Q_IDMB(rho, v, Tb, Tdm, 4*PROTON, mdm, sigma)*(1-xh)
	return QH + QHe

def vcrit(z, Mdm, sigma, frac = 0.1, vi = 0.2, Om = 0.315, Ob = 0.048, OR = 9.54e-5, h = 0.6774, X = 0.76):
	f = lambda x: frac*Tdot(z, Om, Ob, OR, h, X) - Q_z(z, x, Mdm, sigma, Om, Ob, OR, h, X)
	vc = fsolve(f, vi * 1e5)[0]
	vin = vi*1e5
	while vc>vbdm_z(z, 90.):
		vin = vin*5.0
		vc = fsolve(f, vin)[0]
	return vc

TS0 = 1420e6*PLANCK/BOL

def xalpha(T0, z, xa0, Om = 0.315, Ob = 0.048, h = 0.6774, Tse = 0.402):
	T = T0*(T0>1) + 1.0*(T0<=1)
	Sa = np.exp(-2.06*((Ob*h)/0.0327)**(1/3)*(Om/0.307)**(-1/6)*((1+z)/10)**0.5*(T/Tse)**(-2/3))
	return xa0 * (1+Tse/T)**-1 * Sa

def T21_IGM(z, TS, Om = 0.315, Ob = 0.048, h = 0.6774, X = 0.76, T0=2.726):
	Tcmb = T0*(1+z)
	#xa = xalpha(TS, z, xa0, Om, Ob, h)
	return 26.8 * X * (Ob*h/0.0327)*(Om/0.307)**-0.5*((1+z)/10)**0.5*(TS-Tcmb)/TS #* xa/(1+xa)

def TS_T21(z, T, Om = 0.315, Ob = 0.048, h = 0.6774, X = 0.76, T0=2.726):
	Tcmb = T0*(1+z)
	A = 26.8 * X * (Ob*h/0.0327)*(Om/0.307)**-0.5*((1+z)/10)**0.5
	T_T = 1 - T/A
	return Tcmb/T_T

def T21_pred(v0 = 30., Mdm = 0.3, sigma = 8e-20, xa0 = 1.0, z1 = 17.0, z0 = 1000., Om = 0.315, Ob = 0.048, OR = 9.54e-5, h = 0.6774, X = 0.76, a1=1./119, a2=1./115, T0=2.726, nb = 100000):
	d = main(z0, z1, v0, Mdm, sigma, Om, Ob, OR, h, X, a1, a2, T0, nb)
	#TS = max(d['Tb'][-1], TS0)
	Tb = d['Tb'][-1] #TS_Tb(d['Tb'][-1], z1, xa0, Om, Ob, h, T0)
	TS = TS_Tb(Tb, z1, xa0)
	return T21_IGM(z1, TS, Om, Ob, h, X, T0), Tb

def parasp(v0 = 30., m1 = -4, m2 = 2, s1 = -1, s2 = 4, nbin = 10, xa0 = 1.0, ncore=4):
	lm = np.logspace(m1, m2, nbin)
	ls = np.logspace(s1, s2, nbin)
	X, Y = np.meshgrid(lm, ls, indexing = 'ij')
	lT = np.zeros(X.shape)
	lTb = np.zeros(X.shape)
	np_core = int(nbin/ncore)
	lpr = [[i*np_core, (i+1)*np_core] for i in range(ncore-1)] + [[(ncore-1)*np_core, nbin]]
	print(lpr)
	manager = mp.Manager()
	def sess(pr0, pr1, j):
		out = []
		for i in range(pr0, pr1):
			sol = T21_pred(v0, lm[i], ls[j]*1e-20, xa0)
			out.append(sol)
		output.put((pr0, np.array(out).T))
	for i in range(nbin):
		output = manager.Queue()
		pros = [mp.Process(target=sess, args=(lpr[k][0], lpr[k][1], i)) for k in range(ncore)]
		for p in pros:
			p.start()
		for p in pros:
			p.join()
		out = [output.get() for p in pros]
		out.sort()
		lT[:,i] = -np.hstack([x[1][0] for x in out])
		lTb[:,i] = np.hstack([x[1][1] for x in out])
		#for j in range(nbin):
		#	sol = T21_pred(v0, lm[i], ls[j]*1e-20, xa0)
		#	lT[i,j] = -sol[0]
		#	lTb[i,j] = sol[1]
	return X, Y*1e-20, lT, lTb

def main(z0 = 1000., z1 = 9.0, v0 = 30., Mdm = 0.3, sigma = 8e-20, Om = 0.315, Ob = 0.048, OR = 9.54e-5, h = 0.6774, X = 0.76, a1=1./119, a2=1./115, T0=2.726, nb = 100000, Tmin = 0.1, vmin = 1e-10):
	xh = 4*X/(1+3*X)
	def func(y, a):
		if y[1]<Tmin: #y[1]<=y[0]:
			dTdm = -2*y[0]/a
			dTb = -2*y[1]/a
		else:
			rhob = Ob/Om * rhom(a, Om, h)
			QH = Q_IDMB(rhob, y[2], y[0], y[1], Mdm*GeV_to_mass, PROTON, sigma)*xh
			QHe = Q_IDMB(rhob, y[2], y[0], y[1], Mdm*GeV_to_mass, 4*PROTON, sigma)*(1-xh)
			dTdm = -2*y[0]/a + (QH+QHe)/ (a*H(a, Om, h, OR))
			rhodm = (Om-Ob)/Om * rhom(a, Om, h)
			QH = Q_IDMB(rhodm, y[2], y[1], y[0], PROTON, Mdm*GeV_to_mass, sigma)*xh
			QHe = Q_IDMB(rhodm, y[2], y[1], y[0], 4*PROTON, Mdm*GeV_to_mass, sigma)*(1-xh)
			dTb = -2*y[1]/a + (GammaC(1/a-1, Om, Ob, OR, h, X, a1, a2, T0)*(T0/a-y[1]) + (QH+QHe))/ (a*H(a, Om, h, OR))
		if y[2]<vmin:
			dv = -y[2]/a
		else:
			DH = drag(rhom(a, Om, h), y[2], y[1], y[0], PROTON, Mdm*GeV_to_mass, sigma)
			DHe = drag(rhom(a, Om, h), y[2], y[1], y[0], 4*PROTON, Mdm*GeV_to_mass, sigma)	
			dv = -y[2]/a - (xh*DH + (1-xh)*DHe)/(a*H(a, Om, h, OR))
		return [dTdm, dTb, dv]
	ai = 1/(1+z0)
	af = 1/(1+z1)
	#lz = np.linspace(z0, z1, nb)
	la = np.logspace(np.log10(ai), np.log10(af), nb)
	#la = 1/(lz+1)
	#print(la)
	lz = 1/la - 1
	y0 = [T_dm(z0, Mdm), T_b(z0), vbdm_z(z0, v0)]
	sol = odeint(func, y0, la)#, mxstep = 1000, hmin = 1e-30)
	d = {}
	sol = sol.T
	d['lz'] = lz
	d['la'] = la
	d['Tb'] = sol[1]
	d['Tdm'] = sol[0]
	d['v'] = sol[2]
	d['u'] = uthf(PROTON, Mdm*GeV_to_mass, d['Tb'], d['Tdm'])
	return d

def main_(z0 = 500., z1 = 9.0, v0 = 30., Mdm = 0.3, sigma = 8e-20, Om = 0.315, Ob = 0.048, OR = 9.54e-5, h = 0.6774, X = 0.76, a1=1./119, a2=1./115, T0=2.726, nb = 1000, tol = 1e-20):
	xh = 4*X/(1+3*X)
	def func(y, t):
		Tdm, Tb, v = max(y[0], tol), max(y[1], tol), max(y[2], tol)
		a = 1/(1+ZT(np.log10(t/1e9/YR)))
		if y[1]<=Tdm:
			dTdm = -2*Tdm/a * (a*H(a, Om, h, OR))
			dTb = -2*Tb/a * (a*H(a, Om, h, OR))
		else:
			rhob = Ob/Om * rhom(a, Om, h)
			QH = Q_IDMB(rhob, v, Tdm, Tb, Mdm*GeV_to_mass, PROTON, sigma)*xh
			QHe = Q_IDMB(rhob, v, Tdm, Tb, Mdm*GeV_to_mass, 4*PROTON, sigma)*(1-xh)
			dTdm = -2*Tdm/a *(a*H(a, Om, h, OR)) + (QH+QHe)#/ (a*H(a, Om, h, OR))
			rhodm = (Om-Ob)/Om * rhom(a, Om, h)
			QH = Q_IDMB(rhodm, v, Tb, Tdm, PROTON, Mdm*GeV_to_mass, sigma)*xh
			QHe = Q_IDMB(rhodm, v, Tb, Tdm, 4*PROTON, Mdm*GeV_to_mass, sigma)*(1-xh)
			dTb = -2*Tb/a *(a*H(a, Om, h, OR)) + (GammaC(1/a-1, Om, Ob, OR, h, X, a1, a2, T0)*(T0/a-Tb) + (QH+QHe))#/ (a*H(a, Om, h, OR))
		DH = drag(rhom(a, Om, h), v, Tb, Tdm, PROTON, Mdm*GeV_to_mass, sigma)
		DHe = drag(rhom(a, Om, h), v, Tb, Tdm, 4*PROTON, Mdm*GeV_to_mass, sigma)	
		dv = -v/a *(a*H(a, Om, h, OR)) - (xh*DH + (1-xh)*DHe)#/(a*H(a, Om, h, OR))
		return [dTdm, dTb, dv]
	lz = np.linspace(z0, z1, nb)
	la = 1/(1+lz)
	lt = np.array([TZ(x) for x in lz])
	y0 = [T_dm(z0, Mdm), T_b(z0), vbdm_z(z0, v0)]
	d = {}
	d['la'] = la
	d['lz'] = lz
	d['Tb'] = np.zeros(nb)
	d['Tdm'] = np.zeros(nb)
	d['v'] = np.zeros(nb)
	sol = odeint(func, y0, lt)#, rtol=1e-2)
	sol = sol.T
	d['Tdm'] = sol[0]
	d['Tb'] = sol[1]
	d['v'] = sol[2]
	#r = ode(func).set_integrator('dopri5', rtol = 1e-2)
	#r.set_initial_value(y0, TZ(z0))
	#d['Tdm'][0], d['Tb'][0], d['v'][0] = y0[0], y0[1], y0[2]
	#count = 1
	#while r.successful() and count<nb:
	#	r.integrate(la[count])
	#	d['Tdm'][count] = r.y[0]
	#	d['Tb'][count] = r.y[1]
	#	d['v'][count] = r.y[2]
	#d['u'] = uthf(PROTON, Mdm*GeV_to_mass, d['Tb'], d['Tdm'])
	return d

def TS_Tb(T, z, xa0 = 1.85, Om = 0.315, Ob = 0.048, h = 0.6774, T0 = 2.726, Tse = 0.402):
	xa = xalpha(T, z, xa0, Om, Ob, h, Tse)
	Tcmb = T0*(1+z)
	A = (1+xa)*T/((1+xa)*T-xa*(T-Tcmb))
	TS = Tcmb*A
	return TS * (TS>Tse) + Tse * (TS<=Tse)
	

#lTb = np.array([20, 6.790859696667654, 0.5629660674927683, 0.0019257427490605513, 1e-3])
#lTS = np.array([1.0, 1.5, 4.437998143755263, 275.0427210371158, 450.])*lTb

#def ratioT(logT, a, b):
#	return a*logT + b
#ratioT = interp1d(np.log(lTb), np.log(lTS))

#para = curve_fit(ratioT, np.log(lTb), np.log(lTS))
#print('a = {}, b = {}'.format(para[0][0], para[0][1]))

xa0 = 1.73

lx = 10**np.linspace(-3, 1, 100)
ly = TS_Tb(lx, 17.0, xa0)#, para[0][0], para[0][1]))
plt.plot(lx, ly)
#plt.plot(lTb[1:], lTS[1:], 'o')
plt.xscale('log')
plt.yscale('log')
plt.savefig('TS_Tb.pdf')

#plt.show()

TS_edges = TS_T21(17, -500)
print('Spin temperature implied by EDGES: {} K'.format(TS_edges))
Tb = T21_pred(1e-10)[1]
print('Tb and TS with DMBS: {} K, {} K'.format(Tb, TS_Tb(Tb, 17.0, xa0)))
print('Tb and TS in CDM: {} K, {} K'.format(T_b(17), TS_Tb(T_b(17), 17.0, xa0)))
T210 = T21_IGM(17, TS_Tb(T_b(17), 17.0, xa0))
print('T21 in CDM : {} mK'.format(T210))
#print('Ratio of TS = {}'.format(TS_edges/TS))

def vdis(v, sigma = 30):
	return v**2*np.exp(-3*(v/sigma)**2/2)

def stack(lv, lZ):
	Z = np.zeros(lZ[0].shape)
	nx, ny = lZ[0].shape[0], lZ[0].shape[1]
	lw = vdis(lv)
	norm = np.trapz(lw, lv)
	for i in range(nx):
		for j in range(ny):
			lT21 = np.array([x[i, j] for x in lZ])
			Z[i, j] = np.trapz(lT21*lw, lv)/norm
	return Z
	

if __name__=="__main__":
	mode = 1
	v0 = 0.1
	nbin = 48
	ncore = 4
	"""
	lf = ['1e-10', '10.0', '21.0', '30.0', '40.0', '50.0', '60.0', '70.0', '80.0', '90.0']
	lZ = [-T21_IGM(17.0, TS_Tb(np.array(retxt('Tb_'+v0+'.txt',nbin,0,0)),17.0, xa0)) for v0 in lf]
	lv = np.array([0, 10, 21, 30, 40, 50, 60, 70, 80, 90])
	print(lv)
	Z = stack(lv, lZ)
	X = np.array(retxt('X_'+lf[0]+'.txt',nbin,0,0))
	Y = np.array(retxt('Y_'+lf[0]+'.txt',nbin,0,0))
	plt.figure()
	ctf = plt.contourf(X, Y, -np.log10(Z), 1000, cmap=plt.cm.jet)
	for c in ctf.collections:
		c.set_edgecolor('face')
	cb = plt.colorbar()
	cb.set_label(r'$-\log(-T_{21}\ [\mathrm{mK}])$',size=12)
	#plt.contourf(X, Y, -np.log10(Z), np.linspace(-3.4, -2, 100), cmap=plt.cm.jet)
	plt.contour(X, Y, np.log10(Z), [np.log10(231)], colors='k')
	plt.contour(X, Y, np.log10(Z), [np.log10(300)], colors='k')
	plt.contour(X, Y, np.log10(Z), [np.log10(500)], colors='k')
	plt.contour(X, Y, np.log10(Z), [np.log10(-T210)], colors='k', linestyles='--')
	plt.plot([0.3], [8e-20], '*', color='purple')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$m_{\mathrm{DM}}c^{2}\ [\mathrm{Gev}]$')
	plt.ylabel(r'$\sigma_{1}\ [\mathrm{cm^{2}}]$')
	plt.tight_layout()
	plt.savefig('T21map.pdf')

	Z0 = -T21_IGM(17.0, TS_Tb(np.array(retxt('Tb_'+lf[0]+'.txt',nbin,0,0)),17.0, xa0))
	plt.figure()
	ctf = plt.contourf(X, Y, Z/Z0, 1000, cmap=plt.cm.Blues)
	for c in ctf.collections:
		c.set_edgecolor('face')
	cb = plt.colorbar()
	cb.set_label(r'$\langle T_{21}\rangle/T_{21}(v_{\mathrm{bDM,0}}=0)$',size=12)
	#plt.contourf(X, Y, -np.log10(Z), np.linspace(-3.4, -2, 100), cmap=plt.cm.jet)
	#plt.contour(X, Y, np.log10(Z), [np.log10(231)], colors='k')
	#plt.contour(X, Y, np.log10(Z), [np.log10(300)], colors='k')
	#plt.contour(X, Y, np.log10(Z), [np.log10(500)], colors='k')
	#plt.contour(X, Y, np.log10(Z), [np.log10(-T210)], colors='k', linestyles='--')
	plt.plot([0.3], [8e-20], '*', color='purple')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$m_{\mathrm{DM}}c^{2}\ [\mathrm{Gev}]$')
	plt.ylabel(r'$\sigma_{1}\ [\mathrm{cm^{2}}]$')
	plt.tight_layout()
	plt.savefig('T21Ratio.pdf')
	"""
	"""
	if mode==0:
		X, Y, Z, Tb = parasp(v0, m1 = -4, m2 = 2, s1 = -1, s2 = 2, nbin = nbin, xa0 = xa0, ncore = ncore)
		totxt('X_'+str(v0)+'.txt',X,0,0,0)
		totxt('Y_'+str(v0)+'.txt',Y,0,0,0)
		totxt('Z_'+str(v0)+'.txt',Z,0,0,0)
		totxt('Tb_'+str(v0)+'.txt',Tb,0,0,0)
	else:
		X = np.array(retxt('X_'+str(v0)+'.txt',nbin,0,0))
		Y = np.array(retxt('Y_'+str(v0)+'.txt',nbin,0,0))
		#Z = np.array(retxt('Z_'+str(v0)+'.txt',nbin,0,0))
		Tb = np.array(retxt('Tb_'+str(v0)+'.txt',nbin,0,0))
		TS = TS_Tb(Tb, 17.0, xa0)
		Z = -T21_IGM(17, TS)
	#print(Z)
	Tref = TS_T21(17, -10**3.5)
	print('Minimum of Tb: {} K'.format(np.min(Tb)))
	print('Possible minimum of TS: {} K'.format(Tref))
	print('Maximum -T21: {} mK, with TS = {} K:'.format(np.max(Z), TS_T21(17, -np.max(Z))))
	print('Ratio of TS: {}'.format(Tref/TS_T21(17, -np.max(Z))))
	plt.figure()
	ctf = plt.contourf(X, Y, -np.log10(Z), 100, cmap=plt.cm.jet)
	for c in ctf.collections:
		c.set_edgecolor('face')
	cb = plt.colorbar()
	cb.set_label(r'$-\log(-T_{21}\ [\mathrm{mK}])$',size=12)
	#plt.contourf(X, Y, -np.log10(Z), np.linspace(-3.4, -2, 100), cmap=plt.cm.jet)
	plt.contour(X, Y, np.log10(Z), [np.log10(231)], colors='k')
	plt.contour(X, Y, np.log10(Z), [np.log10(300)], colors='k')
	plt.contour(X, Y, np.log10(Z), [np.log10(500)], colors='k')
	plt.contour(X, Y, np.log10(Z), [np.log10(-T210)], colors='k', linestyles='--')
	plt.plot([0.3], [8e-20], '*', color='purple')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$m_{\mathrm{DM}}c^{2}\ [\mathrm{Gev}]$')
	plt.ylabel(r'$\sigma_{1}\ [\mathrm{cm^{2}}]$')
	plt.tight_layout()
	plt.savefig('T21map_vbDM'+str(v0)+'.pdf')
	#plt.show()
	"""
	"""
	mdm = 0.3
	sig = 8e-20 #-19
	if mode==0:
		lv = 10**np.linspace(0, 5, 1000)
		lTb = np.array([T21_pred(x, mdm, sig)[1] for x in lv])
		lT21 = T21_IGM(17, TS_Tb(lTb, 17.0, xa0))
		totxt('T21_v.txt',[lv, lT21],0,0,0)
	else:
		lf = np.array(retxt('T21_v.txt',2,0,0))
		lv = lf[0]
		lT21 = lf[1]*(lf[1]>-500) + -500*(lf[1]<=-500)
	#print(TS_T21(17, lT21[0]), lTb[0])
	plt.figure()
	plt.plot(lv, lT21, label=r'$m_{\mathrm{DM}}c^{2}='+str(mdm)+r'\ \mathrm{GeV}$, $\sigma_{1}='+str(sig)+r'\ \mathrm{cm^{2}}$')
	plt.plot(lv, [T21_IGM(17, T_b(17)) for x in lv], 'k--', label='CDM (fully coupled)')
	plt.plot(lv, [T21_IGM(17, TS_Tb(T_b(17), 17.0, xa0)) for x in lv], 'k:', label=r'CDM ($x_{a,0}=1.73$)')
	plt.legend(loc=4)
	plt.ylim(-550, 0)
	plt.xlim(np.min(lv), np.max(lv))
	plt.xscale('log')
	plt.xlabel(r'$v_{\mathrm{bDM},0}\ [\mathrm{km\ s^{-1}}]$')
	plt.ylabel(r'$T_{21}\ [\mathrm{mK}]$')
	plt.tight_layout()
	plt.savefig('T21_v_mdm'+str(mdm)+'GeV_sigma_1'+str(sig)+'_.pdf')
	T21 = np.trapz(vdis(lv)*lT21, lv)/np.trapz(vdis(lv), lv)
	print('Averaged T21 with streaming motions : {} mK'.format(T21))
	"""
	
	lls = ['-', '--', '-.', ':']
	llc = ['k', 'b', 'g', 'r']#['b', 'g', 'orange', 'r']
	lv0 = [1e-10, 30, 60, 90]
	llb = [r'$v_{b\chi,0}=0$', r'$v_{b\chi,0}=1\sigma_{\mathrm{rms}}$', r'$v_{b\chi,0}=2\sigma_{\mathrm{rms}}$', r'$v_{b\chi,0}=3\sigma_{\mathrm{rms}}$']
	mdm = 0.3# 0.003 #3e-1
	sig = -19
	zmax = 1000
	z0, z1 = 1100, 9
	fig = plt.figure(figsize=(12,5))
	ax1 = plt.subplot(121)
	ax2 = plt.subplot(122)
	down1, up1 = 1e-1, 1e3
	down2, up2 = 1e-2, 1e2
	ax1.text(z1+2, up1*0.4, r'$m_{\mathrm{\chi}}c^{2}='+str(mdm)+r'\ \mathrm{GeV}$, $\sigma_{1}=10^{'+str(sig)+r'}\ \mathrm{cm^{2}}$')
	#ax2.text(z1+15, up2*0.75, r'$m_{\mathrm{DM}}c^{2}='+str(mdm)+r'\ \mathrm{GeV}$, $\sigma_{1}=10^{'+str(sig)+r'}\ \mathrm{cm^{2}}$')
	for v, c, l, ls in zip(lv0, llc, llb, lls):
		if c is not 'k':
			d = main(z0, z1, v0 = v, Mdm=mdm, sigma=10**sig, Tmin = 1e-5)
			ax1.plot(d['lz']+1, d['Tb'], color=c, label=r'$T_{b}$, '+l)
			ax1.plot(d['lz']+1, d['Tdm'], color=c, label=r'$T_{\chi}$, '+l, ls='--')
		if c is not 'k':
			ax2.plot(d['lz']+1, d['v']/1e5, label=l, color=c)
			ax2.plot(d['lz']+1, d['u']/1e6, ls='--', color=c, label=r'$0.1u_{\mathrm{th}}$, '+l)
			ax2.plot(d['lz']+1, vbdm_z(d['lz'], v)/1e5, ls = '-.', color=c, label=l+', CDM')
	ax1.plot(d['lz']+1, T_b(d['lz']), 'k-.', label=r'$T_{b}$, CDM')
	#ax1.plot(d['lz']+1, T_dm(d['lz'], mdm), 'k:', label=r'$T_{\mathrm{DM}}$, CDM')
	ax1.fill_between([15, 20],[up1, up1],[down1, down1],label='EDGES',facecolor='gray')
	ax1.set_xlabel(r'$1+z$')
	ax1.set_ylabel(r'$T\ [\mathrm{K}]$')
	ax1.legend(loc=4)
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.set_xlim(z1+1, zmax)
	ax1.set_ylim(down1, up1)
	#ax2.plot(d['lz'],np.zeros(len(d['lz'])), 'k', lw=0.5)
	ax2.set_xlabel(r'$1+z$')
	ax2.set_ylabel(r'$v_{b\chi}\ [\mathrm{km\ s^{-1}}]$')
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.legend(loc=4)
	ax2.set_xlim(z1+1, zmax)
	ax2.set_ylim(down2, up2)
	plt.tight_layout()
	plt.savefig('T_z.pdf')#_mdm'+str(mdm)+'GeV_logsigma1'+str(sig)+'_.pdf')

	"""
	m_dm = 0.3
	#z0, z1 = 1e3, 9
	z0, z1 = 500, 9
	lz = np.linspace(z0,z1,500)
	llc = ['yellow', 'orange', 'r']
	frac = 1
	down, up = 0.1, 10
	lsigma = [-19.5, -19., -18.5, -18]#, -18]
	plt.figure()
	plt.text(z1+11, up*0.6, r'$m_{\mathrm{DM}}c^{2}='+str(m_dm)+r'\ \mathrm{GeV}$, $\frac{\dot{\Lambda}_{\mathrm{scat}}}{\dot{\Lambda}_{\mathrm{adia}}}='+str(frac)+'$')
	for s, i in zip(lsigma, lls):
		lv = [vcrit(x, m_dm, 10**s, frac, 0.2)/1e5 for x in lz]
		plt.plot(lz+1, lv, label=r'$v_{\mathrm{crit}}$, $\sigma_{1}=10^{'+str(s)+'}\ \mathrm{cm^{2}}$',ls=i)
		plt.plot(lz+1, dv_z(lz, Mdm=m_dm, sigma=10**s)/1e5, color=llc[0], ls=i, lw=3, alpha=0.5)#, label=r'$\dot{v}t_{\mathrm{H}}$, $\sigma_{0}='+str(s)+'\ \mathrm{cm^{2}}$')
		plt.plot(lz+1, dv_z(lz, 60., Mdm=m_dm, sigma=10**s)/1e5, color=llc[1], ls=i, lw=3, alpha=0.5)
		plt.plot(lz+1, dv_z(lz, 90., Mdm=m_dm, sigma=10**s)/1e5, color=llc[2], ls=i, lw=3, alpha=0.5)
	a = [plt.plot(lz+1, vbdm_z(lz, i*30)/1e5, label=r'$v_{\mathrm{bDM}}$, $'+str(i)+'\sigma$', color = llc[i-1], lw=1) for i in range(1, 4)]
	plt.fill_between([16, 19],[up, up],[down, down],label='EDGES',facecolor='gray')
	plt.plot([], [], 'k', lw=3, alpha=0.3, label=r'$-\dot{v}t_{\mathrm{H}}$')
	plt.xlabel(r'$1+z$')
	plt.ylabel(r'$v\ [\mathrm{km\ s^{-1}}]$')
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(down, up)
	plt.xlim(z1+1, z0+1)
	plt.legend(loc=4)
	plt.tight_layout()
	plt.savefig('v_sigma.pdf')

	sig = -19
	lm = [0.2, 0.5, 0.7, 1.0]
	plt.figure()
	plt.text(z1+11, up*0.6, r'$\sigma_{1}=10^{'+str(sig)+r'}\ \mathrm{cm^{2}}$, $\frac{\dot{\Lambda}_{\mathrm{scat}}}{\dot{\Lambda}_{\mathrm{adia}}}='+str(frac)+'$')
	for m, i in zip(lm, lls):
		lv = [vcrit(x, m, 10**sig, frac)/1e5 for x in lz]
		plt.plot(lz+1, lv, label=r'$v_{\mathrm{crit}}$, $m_{\mathrm{DM}}c^{2}='+str(m)+r'\ \mathrm{GeV}$', ls=i)
		plt.plot(lz+1, dv_z(lz, Mdm=m, sigma=10**sig)/1e5, color=llc[0], ls=i, lw=3, alpha=0.5)#, label=r'$\dot{v}t_{\mathrm{H}}$, $\sigma_{0}='+str(s)+'\ \mathrm{cm^{2}}$')
		plt.plot(lz+1, dv_z(lz, 60., Mdm=m, sigma=10**sig)/1e5, color=llc[1], ls=i, lw=3, alpha=0.5)
		plt.plot(lz+1, dv_z(lz, 90., Mdm=m, sigma=10**sig)/1e5, color=llc[2], ls=i, lw=3, alpha=0.5)
	a = [plt.plot(lz+1, vbdm_z(lz, i*30)/1e5, label=r'$v_{\mathrm{bDM}}$, $'+str(i)+'\sigma$', color = llc[i-1], lw=1) for i in range(1, 4)]
	plt.fill_between([16, 19],[up, up],[down, down],label='EDGES',facecolor='gray')
	plt.plot([], [], 'k', lw=3, alpha=0.3, label=r'$-\dot{v}t_{\mathrm{H}}$')
	plt.xlabel(r'$1+z$')
	plt.ylabel(r'$v\ [\mathrm{km\ s^{-1}}]$')
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(down, up)
	plt.xlim(z1+1, z0+1)
	plt.legend(loc=4)
	plt.tight_layout()
	plt.savefig('v_mDM.pdf')

	v = 0.2
	dT1 = Tdot(lz)
	Q0 = Q_z(lz, v, m_dm)
	plt.figure()
	plt.plot(lz+1, -dT1, label='adiabatic cooling')
	plt.plot(lz+1, -Q0, label=r'scattering cooling, $v_{\mathrm{bDM}}='+str(v)+'\ \mathrm{km\ s^{-1}}$', ls = '-.')
	plt.xlabel(r'$1+z$')
	plt.ylabel(r'$\dot{T}\ [\mathrm{K\ s^{-1}}]$')
	#plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	plt.savefig('CoolingRate_mdm'+str(m_dm)+'GeV_v'+str(v)+'.pdf')
	"""

	"""
	#z0, z1 = 1e3, 9
	m_dm = mdm
	z0, z1 = 1000, 9
	lz = np.linspace(z0,z1,1000)
	T0 = T_dm(lz, m_dm)
	T1 = [T_cosmic(x) for x in lz]
	T2 = T_b(lz)
	Tcmb = 2.726*(lz+1)
	plt.figure()
	plt.plot(1+lz, T0, label=r'$T_{\mathrm{DM}}$, $m_{\chi}c^{2}='+str(m_dm)+r'\ \mathrm{GeV}$')
	plt.plot(1+lz, T1, '--', label=r'$T_{\mathrm{b}}$, Mirocha (2018)')
	plt.plot(1+lz, T2, '-.', label=r'$T_{\mathrm{b}}$, Tseliakhovich (2010)')
	plt.plot(1+lz, Tcmb, 'k:', label=r'$T_{\mathrm{CMB}}$')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel(r'$1+z$')
	plt.ylabel(r'$T\ [\mathrm{K}]$')
	plt.xlim(z0+1, z1+1)
	plt.legend()
	plt.tight_layout()
	plt.savefig('IGM_T_z_mdm'+str(m_dm)+'GeV.pdf')
	"""
	#plt.show()



