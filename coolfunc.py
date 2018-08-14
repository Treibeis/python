from onezone1 import *
import sys
lls = ['-', '--', '-.', ':']*2
llw = [1, 1, 1, 1, 2, 2, 2, 2]


if __name__ == "__main__":
	tmax = 6
	nb = 10**int(sys.argv[1])
	Tbase1 = 10**np.linspace(1, tmax, 100)
	Tbase2 = 10**np.linspace(1, 4+np.log10(2), 100)
	Tbase3 = 10**np.linspace(np.log10(5.1)+3, tmax, 100)

	lny1 = equilibrium_i(Tbase1, nb)
	lny2 = equilibrium_i(Tbase2, nb)
	lny3 = equilibrium_i(Tbase3, nb)
	lLam1 = np.array(np.matrix([coolfunc(Tbase1[i], lny1[i]) for i in range(len(lny1))]).transpose())
	lLam2 = np.array(np.matrix([coolfunc(Tbase2[i], lny2[i]) for i in range(len(lny2))]).transpose())
	lLam3 = np.array(np.matrix([coolfunc(Tbase3[i], lny3[i]) for i in range(len(lny3))]).transpose())
	plt.figure(figsize=(8,6))
	plt.plot(Tbase3, lLam3[4]/nb, label="atomic")
	plt.plot(Tbase2, lLam2[1]/nb, '--', label=r'$\mathrm{H_{2}}$')
	plt.plot(Tbase2, lLam2[2]/nb, '-.', label=r'$\mathrm{HD}$')
	plt.plot(Tbase2, lLam2[3]/nb, ':', label=r'$\mathrm{LiH}$')
	#plt.plot(Tbase3, lLam3[5+8]/nb, label=r"$\mathrm{H^{+}}$", lw=0.5)#, color = 'g')
	#plt.plot(Tbase3[10:], lLam3[5+9][10:]/nb, label=r"$\mathrm{He^{+}}$", lw=0.5)#, color = 'orange')
	#plt.plot(Tbase3[2:], lLam3[5][2:]/nb, label=r"Bremsstrahlung", lw=0.5)#, color = 'r')
	plt.plot(Tbase1, lLam1[0]/nb, label='total', color='k', lw = 2)
	#[plt.plot(Tbase3, lLam3[5+i]/nb, label=str(i), lw=0.5) for i in range(11)]
	plt.legend()
	plt.yscale('log')
	plt.xscale('log')
	plt.xlim(10, 10**tmax)
	plt.xlabel(r'$T\ [\mathrm{K}]$')
	plt.ylabel(r'$W\ [\mathrm{erg\ s^{-1}}]$')
	plt.title('Overall cooling rates for $n=10^{'+sys.argv[1]+'}\ \mathrm{cm^{-3}}$ with\n'+r'$[\mathrm{H_{2}/H}]=10^{-3}$, $[\mathrm{HD/H}]=4\times 10^{-6}$, $[\mathrm{LiH/H}]=4.6\times 10^{-10}$')
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



