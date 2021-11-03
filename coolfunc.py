from onezone1 import *
import sys
lls = ['-', '--', '-.', ':']*2
llw = [1, 1, 1, 1, 2, 2, 2, 2]
from cosmology import *
PLANCK = 2*np.pi*HBAR
_C18O_nu = 109.782182
_13CO_nu = 110.201370
_12CO_nu = 115.271203

_C18O_A = 6.5e-8
_13CO_A = 6.5e-8
_12CO_A = 7.4e-8

def nu_J(J, nu0 = _12CO_nu):
	return nu0*J*(J+1)/2 * (J>=0) + 0.0

def A_J(J, nu0 = _12CO_nu, A0 = _12CO_A):
	return A0*((nu_J(J, nu0)-nu_J(J-1, nu0))/nu_J(1, nu0))**3*J/(2*J+1) * 3

def occupation_LTE(T, nu0 = _12CO_nu, nJ = 1000):
	lJ = np.linspace(0, nJ, nJ-1)
	lnu = nu_J(lJ, nu0)
	lfrac = (2*lJ+1)*np.exp(-lnu*1e9*PLANCK/(BOL*T))
	return lfrac/np.sum(lfrac)

def CO_cooling(T, J1, J2, xCO = 1e-4):
	lxJ = occupation_LTE(T)
	out = np.sum([lxJ[i+1]*A_J(i)*(nu_J(i+1)-nu_J(i))*1e9*PLANCK for i in range(J1, J2+1)])
	return out * xCO

if __name__ == "__main__":
	tmax = 7
	nb = 10**int(sys.argv[1])
	Tbase1 = 10**np.linspace(0.1, tmax, 100)
	Tbase2 = 10**np.linspace(0.1, 4+np.log10(2), 100)
	Tbase3 = 10**np.linspace(np.log10(5.1)+3, tmax, 100)
	Tbase0 = np.logspace(0.1, 3, 100)	

	lny1 = equilibrium_i(Tbase1, nb)
	lny2 = equilibrium_i(Tbase2, nb)
	lny3 = equilibrium_i(Tbase3, nb)
	lLam1 = np.array(np.matrix([coolfunc(Tbase1[i], lny1[i]) for i in range(len(lny1))]).transpose())
	lLam2 = np.array(np.matrix([coolfunc(Tbase2[i], lny2[i]) for i in range(len(lny2))]).transpose())
	lLam3 = np.array(np.matrix([coolfunc(Tbase3[i], lny3[i]) for i in range(len(lny3))]).transpose())
	lLCO = np.array([CO_cooling(x, 2, 5) for x in Tbase0])
	plt.figure(figsize=(8,6))
	#plt.plot(Tbase3, lLam3[4]/nb, label="atomic")
	plt.plot(Tbase1, lLam1[0]/nb, label='total', color='k', lw = 2)
	plt.plot(Tbase3[2:], lLam3[5][2:]/nb, '-', label=r"free-free", lw=1)#, color = 'r')
	plt.plot(Tbase3[11:], lLam3[5+9][11:]/nb, '--', label=r"$\mathrm{He}$", lw=1)#, color = 'orange')
	plt.plot(Tbase3, lLam3[5+8]/nb, '-.', label=r"$\mathrm{H}$", lw=1)#, color = 'g')
	plt.plot(Tbase2, lLam2[1]/nb, '-', label=r'$\mathrm{H_{2}}$, $x_{\mathrm{H_{2}}}=10^{-3}$')
	plt.plot(Tbase2, lLam2[2]/nb, '--', label=r'$\mathrm{HD}$, $x_{\mathrm{HD}}=4\times 10^{-6}$')
	plt.plot(Tbase2, lLam2[3]/nb, '-.', label=r'$\mathrm{LiH}$, $x_{\mathrm{LiH}}=4.6\times 10^{-10}$')
	plt.plot(Tbase0, lLCO, ':', label='$\mathrm{CO}$, $x_{\mathrm{CO}}=10^{-4}$')
	#[plt.plot(Tbase3, lLam3[5+i]/nb, label=str(i), lw=0.5) for i in range(11)]
	plt.legend()
	plt.yscale('log')
	plt.xscale('log')
	plt.xlim(1, 10**tmax)
	plt.ylim(1e-31, 1e-15)
	plt.xlabel(r'$T\ [\mathrm{K}]$')
	plt.ylabel(r'$W\ [\mathrm{erg\ s^{-1}}]$')
	plt.title('Cooling rates for primordial gas at $n=10^{'+sys.argv[1]+'}\ \mathrm{cm^{-3}}$')#, \n'+r'with $x_{\mathrm{H_{2}}}=10^{-3}$, $x_{\mathrm{HD}}=4\times 10^{-6}$, $x_{\mathrm{LiH}}=4.6\times 10^{-10}$')
	plt.tight_layout()
	#plt.savefig('coolfunc.pdf')
	plt.savefig('coolfunc_n'+sys.argv[1]+'.pdf')
	#plt.show()

	Tbase4 = 10**np.linspace(4,8,100)
	plt.figure()
	lnb = [1,2,3,4]
	for i in range(len(lnb)):
		lny4 = equilibrium_i(Tbase4, lnb[i])
		lLam4 = np.array(np.matrix([coolfunc(Tbase4[i], lny4[i]) for i in range(len(lny4))]).transpose())
		plt.plot(Tbase4, lLam4[5]/lLam4[0],ls=lls[i],lw=llw[i],label=r'$n=10^{'+str(lnb[i])+'}\ \mathrm{cm^{-3}}$')
	plt.yscale('log')
	plt.xscale('log')
	plt.legend()
	plt.xlabel(r'$T\ [\mathrm{K}]$')
	plt.ylabel(r'$W_{\mathrm{ff}}/W_{\mathrm{tot}}$')
	plt.tight_layout()
	plt.savefig('ff_ratio.pdf')
	plt.show()



