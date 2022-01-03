from cosmology import *
from txt import *
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

gammaE = np.euler_gamma

def laxdot(eta, x, chi1=0, chi2=0, chi=0):
	la = np.zeros((8, x.shape[0]))
	la[0], la[1] = 1, 0
	la[2] = -743./336-11*eta/4
	la[3] = 4*np.pi-113./12*chi+19*eta/6*(chi1+chi2)
	la[4] = 34103./18144+5*chi**2+eta*(13661./2016-chi1*chi2/8)+59*eta**2/18
	la[5] = -np.pi*(4159./672+189./8*eta)-chi*(31571./1008-1165./24*eta)
	la[5] += (chi1+chi2)*(21863./1008*eta-79./6*eta**2)-3./4*chi**3+9*eta/4*chi1*chi2*chi
	la[6] = 16447322263./139708800-1712./105*gammaE+16*np.pi**2/3-856./105*np.log(16*x)
	la[6] += eta*(451*np.pi**2/48-56198689./217728)+541./896*eta**2-5605./2592*eta**3
	la[6] += -80*np.pi/3*chi+(20*np.pi/3-1135./36*chi)*eta*(chi1+chi2)
	la[6] += (64153./1008-457./36*eta)*chi**2-(787./144*eta-3037./144*eta**2)*chi1*chi2
	la[7] = -np.pi*(4415./4032-358675./6048*eta-91495./1512*eta**2)
	la[7] += -chi*(2529407./27216-845827./6048*eta+41551./864*eta**2)
	la[7] += (chi1+chi2)*(1580239./54432*eta-451597./6048*eta**2+2045./432*eta**3
			+107./6*eta*chi**2-5*eta**2/24*chi1*chi2) + 12*np.pi*chi**2
	la[7] += -chi**3*(1505./24+eta/8)+chi1*chi2*chi*(101./24*eta+3./8*eta**2)
	return la.T

def laphi(eta, x, chi1=0, chi2=0, chi=0):
	la = np.zeros((8, x.shape[0]))
	la[0], la[1] = 1, 0
	la[2] = 3715./756+55./9*eta
	la[3] = -16*np.pi+113./3*chi-38./3*eta*(chi1+chi2)
	la[4] = 15293365./508032-50*chi**2+eta*(27145./504+5./4*chi1*chi2)+3085./72*eta**2
	la[5] = (1+1.5*np.log(x))*(np.pi*(38645./756-65./9*eta)-chi*(735505./2268+130./9*eta)+
			(chi1+chi2)*(12850./81*eta+170./9*eta**2)-10./3*chi**3+10*eta*chi*chi1*chi2)
	la[6] = 11583231236531./4694215680-640./3*np.pi**2-6848./21*gammaE-6848./63*np.log(64*x**(3/2))
	la[6] += eta*(2255*np.pi**2/12-15737765635./3048192)+76055./1728*eta**2
	la[6] += -127825./1296*eta**3+2920*np.pi/3*chi-(175-1490*eta)/3*chi**2
	la[6] += -(1120*np.pi/3-1085./3*chi)*eta*(chi1+chi2)+(26945./336*eta-2365./6*eta**2)*chi1*chi2
	la[7] = np.pi*(77096675./254016+378515./1512*eta-74045./756*eta**2)
	la[7] += -chi*(20373952415./3048192+150935./224*eta-578695./432*eta**2)
	la[7] += (chi1+chi2)*(4862041225./1524096*eta+1189775./1008*eta**2
			-71705./216*eta**3-830./3*eta*chi**2+35./3*eta**2*chi1*chi2)-560*np.pi*chi**2
	la[7] += 20*np.pi*eta*chi1*chi2+chi**3*(94555./168-85*eta)
	la[7] += chi*chi1*chi2*(39665./168*eta+255*eta**2)
	return la.T

def laamp(eta, x, chi1=0, chi2=0, chi=0):
	la = np.zeros((7, x.shape[0]),dtype=complex)
	la[0], la[1] = 1, 0
	la[2] = -107./42+55./42*eta
	la[3] = 2*np.pi-4./3*chi+2./3*eta*(chi1+chi2)
	la[4] = -2173./1512-eta*(1069./216-2*chi1*chi2)+2047./1512*eta**2
	la[5] = -107*np.pi/21+eta*(34*np.pi/21-24j)
	la[6] = 27027409./646800-856*gammaE/105+428j*np.pi/105+2*np.pi**2/3
	la[4] += eta*(41*np.pi**2/96-278185./33264)-20261./2772*eta**2+114635./99792*eta**3
	la[6] += -428./105*np.log(16*x)
	return la.T

def xdot(x, eta, chi1=0, chi2=0, chi=0):
	lx = np.array([x**(k/2) for k in range(8)]).T
	la = laxdot(eta, x, chi1, chi2, chi)
	fac = np.array([np.dot(x,a) for x, a in zip(lx, la)])
	return 64./5*eta*x**5*fac

def PNphase(x, eta, chi1=0, chi2=0, chi=0, t0=0, phi0=0):
	lx = np.array([x**(k/2) for k in range(8)]).T
	la = laphi(eta, x, chi1, chi2, chi)
	fac = np.array([np.dot(x,a) for x, a in zip(lx, la)])
	f = x**(3/2)/np.pi
	return 2*np.pi*f*t0 - phi0 - np.pi/4 + 3./(128*eta)*x**(-5./2)*fac

def PNamplitude(x, eta, dl=1., chi1=0, chi2=0, chi=0):
	lx = np.array([x**(k/2) for k in range(7)]).T
	la = laamp(eta, x, chi1, chi2, chi)
	fac = np.array([np.dot(x,a) for x, a in zip(lx, la)])
	return 8*eta*x/dl*(np.pi/5)**0.5*fac

UTgw = lambda M: GRA*Msun/SPEEDOFLIGHT**3 * M
ULgw = lambda M: GRA*Msun/SPEEDOFLIGHT**2 * M

def PNwf(f, M1, M2, DL, chi1 = 0, chi2 = 0):
	M = M1 + M2
	delta = (M1 - M2)/M
	chi = (1+delta)/2*chi1 + (1-delta)/2*chi2
	eta = M1*M2/M**2
	UT = UTgw(M)
	UL = ULgw(M)
	x = (np.pi*f*UT)**(2/3)
	dl = DL/UL
	A = PNamplitude(x, eta, dl, chi1, chi2, chi)
	phi = PNphase(x, eta, chi1, chi2, chi)
	omdot = 3./2*x**0.5*xdot(x, eta, chi1, chi2, chi)
	fac = (np.pi/omdot)**0.5
	return np.abs(A*fac*np.exp(phi*1j))*UT
	
def wf0(sign, f, f0, d):
	return 0.5*(1+sign*np.tanh(4*(f-f0)/d))
	
def lorentz(f, f0, sig):
	return sig**2/((f-f0)**2+sig**2/4)
	
def map_param(eta, chi):
	a1 = -2.417e-3*chi - 1.093e-3*chi**2 - 1.917e-2*eta*chi + 7.267e-2*eta - 2.504e-1*eta**2
	a2 = 5.962e-1*chi - 5.6e-2*chi**2 + 1.52e-1*eta*chi - 2.97*eta + 1.312e1*eta**2
	a3 = -3.283e1*chi + 8.859*chi**2 + 2.931e1*eta*chi + 7.954e1*eta - 4.349e2*eta**2
	a4 = 1.619e2*chi - 4.702e1*chi**2 - 1.751e2*eta*chi - 3.225e2*eta + 1.587e3*eta**2
	a5 = -6.32e2*chi + 2.463e2*chi**2 + 1.048e3*eta*chi + 3.355e2*eta - 5.115e3*eta**2
	a6 = -4.809e1*chi - 3.643e2*chi**2 - 5.215e2*eta*chi + 1.87e3*eta + 7.354e2*eta**2
	
	g1 = 4.149*chi - 4.07*chi**2 - 8.752e1*eta*chi - 4.897e1*eta + 6.665e2*eta**2
	d1 = -5.472e-2*chi + 2.094e-2*chi**2 + 3.554e-1*eta*chi + 1.151e-1*eta + 9.64e-1*eta**2
	d2 = -1.235*chi + 3.423e-1*chi**2 + 6.062*eta*chi + 5.949*eta - 1.069e1*eta**2
	return np.array([a1, a2, a3, a4, a5, a6, g1, d1, d2])
	
s4, s5 = -0.129, -0.384
t0, t2, t3 = -2.686, -3.454, 2.353
def afin(a1, a2, M1, M2):
	M = M1 + M2
	eta = M1*M2/M**2
	q = M2/M1 * (M2<=M1) + M1/M2 * (M2>M1)
	a = (a1+a2*q**2)/(1+q**2)
	return a+s4*a**2*eta+s5*a*eta**2+t0*a*eta+2*3**0.5*eta+t2*eta**2+t3*eta**3
	
k1, k2, k3 = 1.5251, -1.1568, 0.1292
def fRD_a(a, M):
	UT = UTgw(M)
	return 1/(2*np.pi)/UT*(k1+k2*(1-a)**k3)
	
q1, q2, q3 = 0.7, 1.4187, -0.499
def Q_a(a):
	return q1+q2*(1-a)**q3
	
def PMphase(f, eta, lam):
	phi = lam[0]*f**(-5./3)+lam[1]/f+lam[2]/f**(-1./3)+lam[3]+lam[4]*f**(2./3)+lam[5]*f
	return phi/eta
	
def RDphase(f, fRD, eta, lam):
	beta1 = PMphase(fRD, eta, lam)
	beta2 = lam[0]*(-5./3)*fRD**(-8./3)-lam[1]/fRD**2-1./3*lam[2]*fRD**(-4./3)
	beta2 += 2./3*lam[4]*fRD**(-1./3)+lam[5]
	beta2 = beta2/eta
	return beta1 + beta2*f
	
def PMwf(fobs, M1, M2, DL, z = 0, chi1=0, chi2=0, cor = 100):
	f = fobs*(1+z)
	M = M1 + M2
	delta = (M1 - M2)/M
	chi = (1+delta)/2*chi1 + (1-delta)/2*chi2
	eta = M1*M2/M**2
	q = M2/M1 * (M2<=M1) + M1/M2 * (M2>M1)
	a = afin(chi1, chi2, M1, M2)
	#print(a)
	lam = map_param(eta, chi)
	UT = UTgw(M)
	UL = ULgw(M)
	dl = DL/UL
	x = (np.pi*f*UT)**(2/3)
	f_ = f*UT
	
	fRD = fRD_a(a, M)*UT
	f1, f2, d = 0.1*fRD, fRD, 0.005
	phi = PNphase(x, eta, chi1, chi2, chi)*wf0(-1,f_,f1,d)
	phi += PMphase(f_, eta, lam)*wf0(1,f_,f1,d)*wf0(-1,f_,f2,d)
	phi += RDphase(f_,fRD,eta,lam)*wf0(1,f_,f2,d)
	
	omdot = 3./2*x**0.5*xdot(x, eta, chi1, chi2, chi)
	fac = (np.pi/omdot)**0.5
	Q = Q_a(a)/cor
	APM = PNamplitude(x, eta, dl, chi1, chi2, chi)*fac + lam[6]*f_**(5./6)/dl
	ARD = lam[7]*lorentz(f_,fRD,lam[8]*Q)*f_**(-7./6)/dl
	f0, d = 0.98*fRD, 0.015
	A = APM*wf0(-1,f_,f0,d) + ARD*wf0(1,f_,f0,d)
	
	#return np.abs(A*np.exp(phi*1j))*UT*(1+z)**2, ARD*UT*(1+z)**2
	return np.abs(A*np.exp(phi*1j))*UT*(1+z), ARD*UT*(1+z)

def SNR(M1, M2, ref, z, a0 = PC, e0 = 0.99, fac=2):
	M = M1+M2
	DL = DZ(z)*(1+z)
	fRD = fRD_a(afin(0,0,M1,M2), M)/(1+z)
	lf = ref[0]
	fmin = fGW(M, a0, e0)/(1+z)
	sel = (lf<fRD*fac) * (lf>fmin)
	lf = lf[sel]
	S2 = np.trapz(PMwf(lf, M1, M2, DL, z)[0]**2/ref[1][sel]**2, lf)
	return S2**0.5*2
	
def da_de(t, y):
	e, a = t, y[0]
	out = 64*3/304*a/e/(1-e**2)*(1+73./24*e**2+37./96*e**4)/(1+121./304*e**2)
	return out

def de_dt(a, e, M1, M2):
	m1, m2 = M1*Msun, M2*Msun
	out = -304/15*e*GRA**3*m1*m2*(m1+m2)/(SPEEDOFLIGHT**5*a**4*(1-e**2)**2.5)
	out *= (1+121./304*e**2)
	return out

def agw3b(M1, M2, sig, rho, e, H=17.5):
	m = (M1 + M2)*Msun
	m1m2 = M1 * M2 * Msun**2
	Fe = (1-e**2)**(-3.5)*(1+(73./24)*e**2+(37./96)*e**4)
	A = GRA*H*rho/sig
	B = 64*GRA**3*m1m2*m*Fe/(5*SPEEDOFLIGHT**5)
	agw = (B/A)**(1/5) 
	return agw
	
def fGW_wen03(M, a, e, fint=0):
	forb = 0.5*(GRA*M*Msun/a**3)**0.5/np.pi
	npeak = 2*(1+e)**1.1954/(1-e**2)**1.5
	if fint>0:
		npeak = np.round(npeak+0.001)
	return forb*npeak

ck0 = [-1.01678, 5.57372, -4.9271, 1.68506]
def fGW(M, a, e, fint=0, ck=ck0):
	forb = 0.5*(GRA*M*Msun/a**3)**0.5/np.pi
	npeak = 2*(1+ck[0]*e+ck[1]*e**2+ck[2]*e**3+ck[3]*e**4)*(1-e**2)**-1.5
	#print(npeak)
	if fint>0:
		npeak = 2*(npeak<=2)+np.round(npeak)*(npeak>2)
		#print(npeak)
	return forb*npeak

from scipy.integrate import quad, solve_ivp
from miniquasar import rho_inf, sigma_inf, time_isobhb
	
if __name__=="__main__":
	M1, M2 = 125, 125
	de = -5
	e0 = 1-10**de
	gamma = 1.5
	sig, rho = sigma_inf(M1+M2), rho_inf(M1+M2, gamma)
	a0 = agw3b(M1, M2, sig, rho, e0)
	ef = 1e-5
	em = 1e-2
	nb = 500
	le1 = 1-np.geomspace(10**de, 1-em, nb)
	le2 = np.geomspace(em, ef, nb)
	le = np.hstack([le1, le2])
	sol = solve_ivp(da_de, (e0, ef), [a0], method='RK45', t_eval=le)
	la = sol.y[0]
	lfGW = fGW(M1+M2, la, le)
	ldtde = 1/de_dt(la, le, M1, M2)
	lde = le[1:]-le[:-1]
	l2in = (ldtde[1:]+ldtde[:-1])/2 * lde
	ltcol = np.cumsum(l2in)
	tcol1 = ltcol[-1]/YR #np.trapz(ldtde, le)/YR
	tcol2 = time_isobhb(a0, M1, M2, e0)
	print('Coalesence time: {} ({}) Myr'.format(tcol1/1e6, tcol2/1e6))
	f1, f2 = np.min(lfGW)/2, np.max(lfGW)*2
	a1, a2 = np.min(la)/PC, np.max(la)/PC
	#e1, e2 = np.min(le), np.max(le)
	lenU = GRA*(M1+M2)*Msun/SPEEDOFLIGHT**2/PC
	print('Geometric length unit: {} pc'.format(lenU))
	plt.figure(figsize=(10,7))
	plt.subplot(221)
	plt.loglog(la/PC, lfGW, label=r'$1-e_{0}=10^{'+str(de)+'}$,\n $M_{1}='+str(M1)
		+r'\ \mathrm{M_{\odot}}$, $M_{2}='+str(M2)+'\ \mathrm{M_{\odot}}$')
		#, label=r'$a_{0}='+'{:.1f}'.format(a0/lenU/PC)+'M$')
	plt.fill_between([10*lenU, 500*lenU], [f1, f1], [f2, f2], 
		facecolor='gray', alpha=0.5, label=r'$a\sim 10M-500M$')
	plt.legend()
	plt.xlabel(r'$a\ [\mathrm{pc}]$')
	plt.ylabel(r'$f_{\mathrm{GW}}\ [\mathrm{Hz}]$')
	plt.ylim(f1, f2)
	#plt.xlim(a1, a2)
	sel = (la/PC>10*lenU) * (la/PC<500*lenU)
	le_ = le[sel]
	e1, e2 = np.min(le_), np.max(le_)
	plt.subplot(222)
	plt.loglog(le, lfGW)
	plt.fill_between([e1, e2], [f1, f1], [f2, f2], 
		facecolor='gray', alpha=0.5, label=r'$a\sim 10M-500M$')
	#plt.legend()
	plt.xlabel(r'$e$')
	#plt.ylabel(r'$f_{\mathrm{GW}}\ [\mathrm{Hz}]$')
	plt.ylim(f1, f2)
	#plt.xlim(ef, 1)
	plt.tight_layout()
	plt.subplot(223)
	plt.plot(la[1:]/PC, ltcol/YR)
	plt.xlabel(r'$a\ [\mathrm{pc}]$')
	plt.ylabel(r'$t_{\mathrm{col}}\ [\mathrm{yr}]$')
	plt.yscale('log')
	plt.xlim(0, a0/PC)
	plt.tight_layout()
	plt.subplot(224)
	plt.plot(le[1:], ltcol/YR)
	#plt.legend()
	plt.xlabel(r'$e$')
	plt.yscale('log')
	#plt.ylabel(r'$f_{\mathrm{GW}}\ [\mathrm{Hz}]$')
	#plt.ylim(f1, f2)
	#plt.xlim(ef, 1)
	plt.xlim(0, 1.05)
	plt.tight_layout()
	plt.savefig('fGW_tcol_a_e.pdf')
	plt.close()

	cr = 'comp/'
	M1, M2 = 1.4e5, 3e4
	M = M1 + M2
	lab = 'Type2H'
	z = 8.22
	DL = DZ(z)*(1+z)
	fRD = fRD_a(afin(0,0,M1,M2), M)/(1+z)
	#f1, f2 = 1e-4, 1e3
	#y1, y2 = 1e-24, 1e-18
	f1, f2 = 1e-5, 1
	y1, y2 = 4e-22, 1e-16
	lf = np.geomspace(f1, fRD*2, 1000)
	#ET1 = np.array(retxt(cr+'ETbase.txt',2,0,0))
	#SNR1 = SNR(M1, M2, ET1, z)
	ET2 = np.array(retxt(cr+'ETxylophone.txt',2,0,0))
	SNR2 = SNR(M1, M2, ET2, z)
	deciHz = np.array(retxt(cr+'deciHz_op.txt',2,0,0))
	dhz = np.array([deciHz[0], deciHz[1]/deciHz[0]**0.5])
	SNR3 = SNR(M1, M2, dhz, z)
	LISA = np.array(retxt(cr+'LISA.txt',2,0,0))
	lisa = np.array([LISA[0], LISA[1]/LISA[0]**0.5])
	SNR4 = SNR(M1, M2, lisa, z)
	LIGO = np.array(retxt(cr+'AdLIGOdesign.txt',2,0,0))
	LIGO_ = np.array(retxt(cr+'AdLIGOcurrent.txt',2,0,0))
	h22 = PMwf(lf, M1, M2, DL, z)[0]
	plt.figure()
	plt.loglog(lf, 2*h22*lf, label=lab+r', $z_{\mathrm{GW}}='+str(z)+'$')#r'$M_{1}='+str(M1)+r'\ \rm{M}_{\odot}$, $M_{2}='+str(M2)+r'\ \rm{M}_{\odot}$, '+r'$z_{\mathrm{GW}}=2$')
	#plt.loglog(ET1[0], ET1[1]*ET1[0]**0.5, 'k--', label='ET baseline, SNR'+r'$={:.1f}$'.format(SNR1))
	plt.loglog(ET2[0], ET2[1]*ET2[0]**0.5, 'k--', label='ET_xylophone, SNR'+r'$={:.0f}$'.format(SNR2))
	plt.loglog(*deciHz, 'k-.', label='DO_optimal, SNR'+r'$={:.0f}$'.format(SNR3))
	plt.loglog(*LISA, 'k:', label='LISA, SNR'+r'$={:.0f}$'.format(SNR4))
	plt.text(f1*2, y1*2, r'$M_{1}='+str(M1)+r'\ \rm{M}_{\odot}$, $M_{2}='+str(M2)+r'\ \rm{M}_{\odot}$')
	plt.xlabel(r'$f\ [\rm{Hz}]$')
	plt.ylabel('Characteristic strain')#r'$2f|\tilde{h}(f)|\ (\sqrt{fS_{h}(f)})$')
	plt.legend(loc=1)
	plt.xlim(f1, f2)
	plt.ylim(y1, y2)
	plt.tight_layout()
	plt.savefig('h_f_'+lab+'.pdf')
	plt.close()
	
	z = 2
	plt.figure()
	s1 = PMwf(lf/(1+z), M1, M2, DL, z)[0]
	s2 = PMwf(lf, M1/(1+z), M2/(1+z), DL, z)[0]
	plt.loglog(s1/s2)
	plt.ylim(1e-1,10.0)
	#plt.show()
	plt.close()
	
	lm1 = [125, 5e2, 1e4, 1e4, 1e4]
	lm2 = [125, 5e2, 125, 5e2, 1e4]
	llab = ['Type0L', 'Type0H', 'Type1L', 'Type1H', 'Type2']
	z = 1
	DL = DZ(z)*(1+z)
	f1, f2 = 1e-4, 5e3
	y1, y2 = 1e-24, 1e-16
	plt.figure()
	for i in range(len(lm1)):
		M1, M2 = lm1[i], lm2[i]
		M = M1 + M2
		lab = llab[i]
		fRD = fRD_a(afin(0,0,M1,M2), M)/(1+z)
		lf = np.geomspace(f1, fRD*2, 1000)
		h22 = PMwf(lf*(1+z), M1, M2, DL)[0]
		plt.loglog(lf, 2*h22*lf, label=lab)#+r', $z_{\mathrm{GW}}='+str(z)+'$')
	plt.loglog(LIGO[0], LIGO[1]*LIGO[0]**0.5, 'k-', label='AdLIGO')
	plt.loglog(LIGO_[0], LIGO_[1]*LIGO_[0]**0.5, 'k-', label='LIGO-Livingston', lw=0.5)
	plt.loglog(ET2[0], ET2[1]*ET2[0]**0.5, 'k--', label='ET_xylophone')#, SNR'+r'$={:.0f}$'.format(SNR2))
	plt.loglog(*deciHz, 'k-.', label='DO_optimal')#, SNR'+r'$={:.0f}$'.format(SNR3))
	plt.loglog(*LISA, 'k:', label='LISA')#, SNR'+r'$={:.0f}$'.format(SNR4))
	plt.text(f1*2, y1*2, r'$z_{\mathrm{GW}}='+str(z)+'$')
	plt.xlabel(r'$f\ [\rm{Hz}]$')
	plt.ylabel('Characteristic strain')#r'$2f|\tilde{h}(f)|\ (\sqrt{fS_{h}(f)})$')
	#plt.legend(loc=1)
	plt.legend(bbox_to_anchor=(0., 0.72, 1., 1.), loc=3,
       ncol=3, mode="expand", borderaxespad=0.)
	plt.xlim(f1, f2)
	plt.ylim(y1, y2)
	plt.tight_layout()
	plt.savefig('wfdemo.pdf')
	plt.close()
	
	#"""
	z = 0
	norm = UTgw(M)
	#print(norm)
	DL = ULgw(M)
	fRD = fRD = fRD_a(afin(0,0,M1,M2), M)
	f1, f2 = 1e-2/UTgw(M), 0.2/UTgw(M)
	lf = np.geomspace(f1, fRD*2, 1000)
	h22PN = PNwf(lf, M1, M2, DL)
	h22PM, h22RD = PMwf(lf, M1, M2, DL, z, cor = 100)
	y1, y2 = np.min(h22PM/norm)*0.9, np.max(h22PM/norm)*1.1
	plt.figure()
	plt.loglog(lf/(1+z)*norm, h22PM/norm, label='PM')
	plt.loglog(lf/(1+z)*norm, h22PN/norm, '--', label='PN')
	plt.loglog(lf/(1+z)*norm, h22RD/norm, ':', label='RD')
	plt.plot([fRD/(1+z)*norm, fRD/(1+z)*norm], [y1, y2], 'k-.', label=r'$Mf_{\rm{RD}}$')
	plt.xlabel(r'$Mf$')
	plt.ylabel(r'$A(Mf)D_{L}/M$')
	plt.ylim(y1, y2)
	plt.legend()
	plt.tight_layout()
	plt.savefig('wf_test.pdf')
	#plt.show()
	plt.close()
	#"""
	
	M = 100
	a = 1e-5*AU
	x1, x2 = 1e-8, 1
	le = np.linspace(0, 1-x1, 1000) #1-np.geomspace(x1, x2, 100)
	f0 = fGW_wen03(M, a, le)
	f1 = fGW(M, a, le)
	f0_ = fGW_wen03(M, a, le, 1)
	f1_ = fGW(M, a, le, 1)
	y1, y2 = 10, 1e4
	plt.figure()
	plt.plot(le, f0, '--', label='Wen+2003')
	plt.plot(le, f1, label='Hamers+2021')
	plt.plot(le, f0_, ':', label='Wen+2003 (int)')
	plt.plot(le, f1_, '-.', label='Hamers+2021 (int)')
	plt.title(r'$M=100\ \rm M_{\odot}$, $a=10^{-5}\ \rm AU$')
	plt.xlabel(r'$e$')
	plt.ylabel(r'$f_{\rm GW,peak}\ [\rm Hz]$')
	plt.yscale('log')
	plt.xlim(0, 1)
	plt.ylim(y1, y2)
	plt.legend()
	plt.tight_layout()
	plt.savefig('fGW_e.pdf')
	plt.close()
