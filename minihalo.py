from cosmology import *
from scipy.optimize import fsolve
from scipy.special import erf
from scipy.integrate import odeint
from txt import *

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

def main(z0 = 500., z1 = 9.0, v0 = 30., Mdm = 0.3, sigma = 8e-20, Om = 0.315, Ob = 0.048, OR = 9.54e-5, h = 0.6774, X = 0.76, a1=1./119, a2=1./115, T0=2.726, nb = 100000):
	xh = 4*X/(1+3*X)
	def func(y, a):
		if False:#y[1]<=y[0]:
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
		DH = drag(rhom(a, Om, h), y[2], y[1], y[0], PROTON, Mdm*GeV_to_mass, sigma)
		DHe = drag(rhom(a, Om, h), y[2], y[1], y[0], 4*PROTON, Mdm*GeV_to_mass, sigma)	
		dv = -y[2]/a - (xh*DH + (1-xh)*DHe)/(a*H(a, Om, h, OR))
		return [dTdm, dTb, dv]
	lz = np.linspace(z0, z1, nb)
	la = 1/(1+lz)
	y0 = [T_dm(z0, Mdm), T_b(z0), vbdm_z(z0, v0)]
	sol = odeint(func, y0, la)
	d = {}
	sol = sol.T
	d['lz'] = lz
	d['la'] = la
	d['Tb'] = sol[1]
	d['Tdm'] = sol[0]
	d['v'] = sol[2]
	d['u'] = uthf(PROTON, Mdm*GeV_to_mass, d['Tb'], d['Tdm'])
	return d

if __name__=="__main__":
	#"""
	lls = ['-', '--', '-.', ':']
	llc = ['b', 'g', 'orange', 'r']#['g', 'yellow', 'orange', 'r']
	lv0 = [1e-10, 30, 60, 90]
	llb = [r'$v_{\mathrm{bDM},0}=0$', r'$v_{\mathrm{bDM},0}=1\sigma$', r'$v_{\mathrm{bDM},0}=2\sigma$', r'$v_{\mathrm{bDM},0}=3\sigma$']
	mdm = 3e-6
	sig = -19
	zmax = 1000
	z0, z1 = 1100, 9
	fig = plt.figure(figsize=(12,6))
	ax1 = plt.subplot(121)
	ax2 = plt.subplot(122)
	down1, up1 = 0.1, 1e3
	down2, up2 = 1e-2, 1e2
	ax1.text(z1+2, up1*0.6, r'$m_{\mathrm{DM}}c^{2}='+str(mdm)+r'\ \mathrm{GeV}$, $\sigma_{1}=10^{'+str(sig)+r'}\ \mathrm{cm^{2}}$')
	#ax2.text(z1+15, up2*0.75, r'$m_{\mathrm{DM}}c^{2}='+str(mdm)+r'\ \mathrm{GeV}$, $\sigma_{1}=10^{'+str(sig)+r'}\ \mathrm{cm^{2}}$')
	for v, c, l, ls in zip(lv0, llc, llb, lls):
		d = main(z0, z1, v0 = v, Mdm=mdm, sigma=10**sig)
		ax1.plot(d['lz']+1, d['Tb'], color=c, label=r'$T_{\mathrm{b}}$, '+l)
		ax1.plot(d['lz']+1, d['Tdm'], color=c, label=r'$T_{\mathrm{DM}}$, '+l, ls='--')
		if c is not 'b':
			ax2.plot(d['lz']+1, d['v']/1e5, label=l, color=c)
			ax2.plot(d['lz']+1, vbdm_z(d['lz'], v)/1e5, ls = '--', color=c, label=l+', CDM')
			ax2.plot(d['lz']+1, d['u']/1e6, ls='-.', color=c, label=r'$0.1u_{\mathrm{th}}$, '+l)
	ax1.plot(d['lz']+1, T_b(d['lz']), 'k-.', label=r'$T_{\mathrm{b}}$, CDM')
	ax1.plot(d['lz']+1, T_dm(d['lz'], mdm), 'k:', label=r'$T_{\mathrm{DM}}$, CDM')
	ax1.fill_between([16, 19],[up1, up1],[down1, down1],label='EDGEDS',facecolor='gray')
	ax1.set_xlabel(r'$1+z$')
	ax1.set_ylabel(r'$T\ [\mathrm{K}]$')
	ax1.legend(loc=4)
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.set_xlim(z1+1, zmax)
	ax1.set_ylim(down1, up1)
	#ax2.plot(d['lz'],np.zeros(len(d['lz'])), 'k', lw=0.5)
	ax2.set_xlabel(r'$1+z$')
	ax2.set_ylabel(r'$v_{\mathrm{bDM}}\ [\mathrm{km\ s^{-1}}]$')
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.legend(loc=4)
	ax2.set_xlim(z1+1, zmax)
	ax2.set_ylim(down2, up2)
	plt.tight_layout()
	plt.savefig('T_z_mdm'+str(mdm)+'GeV_logsigma1'+str(sig)+'_.pdf')

	
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
	plt.fill_between([16, 19],[up, up],[down, down],label='EDGEDS',facecolor='gray')
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
	plt.fill_between([16, 19],[up, up],[down, down],label='EDGEDS',facecolor='gray')
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

	#"""

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
	#"""
	#plt.show()



