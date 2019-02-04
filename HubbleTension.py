from cosmology import *
from numba import njit
import matplotlib.pyplot as plt

@njit
def foralpha(x):
	return (x-np.sin(x))/(2.*np.pi)
@njit
def fordelta(alpha, lim = 1e-42): 
	Dnom = np.array([2.*(1.-np.cos(alpha))**3, lim]).max()
	return 9.*(alpha-np.sin(alpha))**2. / Dnom

lx = np.linspace(0, 2*np.pi, 10000)
lal = foralpha(lx)
lx = [x for x in lx]#+[2*np.pi]
lal = [x for x in lal]#+[1.0]
	
alpha_ap = interp1d(lal, lx)

def delta(a, a0, dmax = 200.):
	z = 1./a-1.
	zvir = 1./a0-1.
	f0 = (1.+zvir)/(1.+z) * (zvir<=z) + 1.0 * (zvir>z)
	alpha = alpha_ap(f0**1.5)
	d = fordelta(alpha)
	return min(d, dmax)

def init(zdec = 1100, Om = 0.315, Or = 9.54e-5, h = 0.6774):
	H0 = h*100*UV/UL/1e3
	a1 = 1./(1.+zdec)
	rhom1 = Om*H0**2*3/8/np.pi/GRA *(1.+zdec)**3
	rhor1 = Or*H0**2*3/8/np.pi/GRA *(1.+zdec)**4
	rhol1 = (1-Om-Or)*H0**2*3/8/np.pi/GRA
	return [a1, rhom1, rhor1, rhol1]

init0 = init()

def Hmod(a, D, init = init0):
	a1 = init[0]
	rhom = init[1]*delta(a, 1., D)*(a1/a)**3
	rhor = init[2]*(a1/a)**4
	rhol = init[3]
	rho = rhom + rhor + rhol
	return (rho * 8*np.pi*GRA/3)**0.5

def t_a(a, D, init = init0):
	def dt_da(a):
		return 1/a/Hmod(a, D, init)
	a1 = 0.0#init[0]
	I = quad(dt_da, a1, a, epsrel = 1e-8)
	return I[0]

def dC_a(a, D, init = init0):
	def integrand(a):
		return SPEEDOFLIGHT/(a**2)/Hmod(a, D, init)
	I = quad(integrand, a, 1, epsrel = 1e-8)
	return I[0]

if __name__ == '__main__':
	h1 = 0.6774
	h2 = 0.7352
	Om = 0.315
	D0 = ((h2/h1)**2.-(1.-Om))/Om
	print('Default Delta0 = {}'.format(D0))
	la = np.logspace(-3, 0, 1000)
	lz = 1./la - 1.

	plt.figure()
	plt.plot(lz, [delta(a, 1, D0) for a in la])
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\Delta$')
	plt.xscale(r'log')
	plt.xlim(1e-2, 1e3)
	plt.ylim(1, 1.6)
	plt.tight_layout()
	plt.savefig('Delta_z.pdf')
	plt.close()

	H_unit = (100*UV/UL/1e3)
	lH = np.array([Hmod(a, D0) for a in la])/H_unit
	lH0 = H(la, Om, h1)/H_unit
	print('H0_mod = {}, H0_FLRW = {} [100 km s^-1 Mpc^-1]'.format(lH[-1], lH0[-1]))

	plt.figure()
	plt.loglog(lz, lH, label=r'$\Delta_{0}='+str(int(D0*1000)/1000)+'$')
	plt.loglog(lz, lH0, '--', label=r'FLRW')
	plt.legend()
	plt.xlabel(r'$z$')
	plt.ylabel(r'$H\ [\mathrm{100\ km\ s^{-1}\ Mpc^{-1}}]$')
	plt.xlim(1e-3, 10)
	plt.ylim(0.1, 1e2)
	plt.tight_layout()
	plt.savefig('Hubble_z.pdf')
	plt.close()

	la = np.logspace(-3, 0, 100)
	lz = 1./la - 1.

	ldL = np.array([dC_a(a, D0) for a in la])/(UL*1e3)#/la
	ldL0 = np.array([DZ(z) for z in lz])/(UL*1e3)#/la
	plt.figure()
	plt.loglog(lz, ldL, label=r'$\Delta_{0}='+str(int(D0*1000)/1000)+'$')
	plt.loglog(lz, ldL0, '--', label=r'FLRW')
	plt.legend()
	plt.xlabel(r'$z$')
	plt.ylabel(r'$d_{C}\ [\mathrm{Mpc}]$')
	plt.xlim(1e-2, 1e3)
	plt.ylim(10, 1e5)
	plt.tight_layout()
	plt.savefig('dC_z.pdf')
	plt.close()

	lt = np.array([t_a(a, D0) for a in la])/(1e9*YR)
	lt0 = np.array([TZ(z) for z in lz])/(1e9*YR)
	print('t0_mod = {}, t0_FLRW = {} [Gyr]'.format(lt[-1], lt0[-1]))
	plt.figure()
	plt.loglog(lz, lt, label=r'$\Delta_{0}='+str(int(D0*1000)/1000)+'$')
	plt.loglog(lz, lt0, '--', label=r'FLRW')
	plt.legend()
	plt.xlabel(r'$z$')
	plt.ylabel(r'$t\ [\mathrm{Gyr}]$')
	plt.xlim(1e-2, 1e3)
	plt.ylim(1e-3, 20)
	plt.tight_layout()
	plt.savefig('t_z.pdf')
	plt.close()

