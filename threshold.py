from cosmology import *
from coolingf import *
from onezone1 import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def MJeans(T, n, mu=1.22, gamma=5./3):
	cs = (gamma*BOL*T/(mu*PROTON))**0.5
	return 2*(cs/2e4)**3/(n/1e3)**0.5

def T_n_MJ(n, M, mu=1.22, gamma=5./3):
	cs = (M/2*(n/1e3)**0.5)**(1/3)*2e4
	T = cs**2*mu*PROTON/gamma/BOL
	return T

tFF = lambda rho: (3.*np.pi/(32.*GRA*rho))**0.5
Zsun = 0.014

mu = 1.0/Hfrac
gamma = 5./3.
T = 8e3
#T = 8.19e3
n = 2000.
#n = 1000.
xe = 1e-4
rho = n*mu*PROTON

du_dt = 1./(gamma-1.) * n * BOL * T / tFF(rho)
lam_atom = LambdaHI(T, n, n*xe) + LambdaHII(T, n*xe, n*xe)

ratio = du_dt/lam_atom

lam_H2 = LambdaH2(T, 1, n)
xh2_crit = du_dt/lam_H2/n

print('Gamma/Lambda = {}'.format(ratio))

print('Critical H2 abundance: {}'.format(xh2_crit))

lam_metal = JJ_metal_cooling(T, n, xe, 1.)
Z_crit = (du_dt-lam_atom) /lam_metal#/Zsun

print('Critical Z: {} [Zsun]'.format(Z_crit))

print('10^-7 lam_H2/lam_metal [Z/(10^7 xH2)] = {}'.format((lam_H2*n)/lam_metal/1e7))

ngb = 1#32
#ngb = 1
mgas = 1e4
nth = 1e2

Mdot = 0.04*19/18
print('Mdot = {:.4f} [Msun yr^-1]'.format(Mdot))
#Mdot = 0.04
tdyn = tFF(nth*mu*PROTON)

delta = Mdot * tdyn/YR / (ngb*mgas) + 1

print('delta = {:.0f}'.format(delta))

#"""
rep = './'#'OZmodels/'
#lab = 'onezone'
#lab = 'hot'
#lab = 'cold'
#lab = 'hot_LWNS1'
#lab = 'HighInflow'
lab = 'demo'
#Ti = 2e4
#Ti = 7e3
Ti = 1e2
#ni = 5e2
#ni = 1e2
ni = 0.01
boost = 1.0
z = 0.0
Zth = 1e-5 #* Zsun
xh2_crit = 1#1e-7
xh2 = 1e-10
J_21 = 3e4
evo0 = main1(Ti, ni/0.93, xh2, xe, z=z, mode=1, Z=0.0, boost=boost, H2_flag=xh2_crit, J_21 = 0.0, Li=0)#, D=0)
evo1 = main1(Ti, ni/0.93, xh2, xe, z=z, mode=1, Z=Zth*20, boost=boost, H2_flag=xh2_crit, J_21 = J_21, Li=0)#, D=0)
evo2 = main1(Ti, ni/0.93, xh2, xe, z=z, mode=1, Z=0, boost=boost, H2_flag=xh2_crit, J_21 = J_21, Li=0)#, D=0)
evo3 = main1(Ti, ni/0.93, xh2, xe, z=z, mode=1, Z=Zth, boost=boost, H2_flag=xh2_crit, J_21 = 0.0, Li=0)#, D=0)
evo4 = main1(Ti, ni/0.93, xh2, xe, z=z, mode=1, Z=Zth*1e1, boost=boost, H2_flag=xh2_crit, J_21 = 0.0, Li=0)#, D=0)
evo5 = main1(Ti, ni/0.93, xh2, xe, z=z, mode=1, Z=Zth*1e2, boost=boost, H2_flag=xh2_crit, J_21 = 0.0, Li=0)#, D=0)
evo6 = main1(Ti, ni/0.93, xh2, xe, z=z, mode=1, Z=Zth*1e3, boost=boost, H2_flag=xh2_crit, J_21 = 0.0, Li=0)#, D=0)
evo7 = main1(Ti, ni/0.93, xh2, xe, z=z, mode=1, Z=0, boost=boost, H2_flag=xh2_crit, J_21 = J_21/300, Li=0)#, D=0)
		
x1, x2 = ni, 1e15#1e10
y1, y2 = 10, 3e5 #7e2, 2e4
T0, T1, T2 = 1e3, 7e3, 1e4
lnx = np.linspace(x1, x2, 100)
plt.figure()
plt.loglog(evo0['n']*0.93, evo0['T'], 'k-', lw = 3, label=r'$J_{21}=Z=0$', alpha = 0.9)
plt.loglog(evo3['n']*0.93, evo3['T'], '-.', label=r'$J_{21}=0$, $Z=10^{-5}\ \mathrm{Z}_{\odot}$')
plt.loglog(evo4['n']*0.93, evo4['T'], ':', label=r'$J_{21}=0$, $Z=10^{-4}\ \mathrm{Z}_{\odot}$')
plt.loglog(evo5['n']*0.93, evo5['T'], ls=(0, (10, 5)), label=r'$J_{21}=0$, $Z=10^{-3}\ \mathrm{Z}_{\odot}$')
plt.loglog(evo6['n']*0.93, evo6['T'], ls=(0, (2,1)), label=r'$J_{21}=0$, $Z=10^{-2}\ \mathrm{Z}_{\odot}$')
plt.loglog(evo1['n']*0.93, evo1['T'], '--', label=r'$J_{21}=3\times 10^{4}$, $Z=2\times 10^{-4}\ \mathrm{Z}_{\odot}$')
plt.loglog(evo7['n']*0.93, evo7['T'], ls=(0, (2,1,1)), label=r'$J_{21}=300$, $Z=0$', color='brown')
plt.loglog(evo2['n']*0.93, evo2['T'], '-', label=r'$J_{21}=3\times 10^{4}$, $Z=0$', color='r')
plt.plot(lnx, T_n_MJ(lnx, 0.1), 'k--', color='gray', alpha=0.5, lw=3) #, label=r'$M_{\rm J}=0.1-10^4\ \rm M_{\odot}$')
[plt.plot(lnx, T_n_MJ(lnx, 10**(i)), 'k--', color='gray', alpha=0.5, lw=3) for i in range(5)]
plt.text(1.5e-2, 15, r'$M_{\rm J}$')
for i in range(6):
		plt.text(10*10**(i*(2+3./10)), 7e2*10**(i/10), '$10^{'+str(i-1)+r'}\rm\ M_{\odot}$', color='gray')
#plt.fill_between([x1, x2], [T1, T1], [T2, T2], facecolor='k', alpha=0.3)#, label=r'$T(\mathrm{DCBH})$')
#plt.plot([x1, x2], [T0, T0], 'k--', lw=0.5)#, label=r'$T_{\mathrm{SF}}$')
#plt.plot([n, n], [y1, y2], 'k-', lw=0.5)#, label=r'$n_{\mathrm{th}}(\mathrm{DCBH})$')
plt.xlabel(r'$n\ [\mathrm{cm^{-3}}]$')
plt.ylabel(r'$T\ [\mathrm{K}]$')
plt.legend(loc=1,ncol=2)
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.tight_layout()
plt.savefig(rep+'T_n_'+lab+'.pdf')
plt.close()

"""
#x1, x2 = 1e2, 1e6
plt.figure()
plt.loglog(evo0['n']*0.93, evo0['X'][3], 'k-', lw = 3, label=r'$J_{21}=Z=0$', alpha = 0.9)
plt.loglog(evo1['n']*0.93, evo1['X'][3], '--', label=r'$J_{21}=3\times 10^{3}$, $Z=0$')
plt.loglog(evo2['n']*0.93, evo2['X'][3], '-', label=r'$J_{21}=3\times 10^{4}$, $Z=0$')
plt.loglog(evo3['n']*0.93, evo3['X'][3], '-.', label=r'$J_{21}=3\times 10^{4}$, $Z=4\times 10^{-4}\ \mathrm{Z}_{\odot}$')
plt.loglog(evo4['n']*0.93, evo4['X'][3], ':', label=r'$J_{21}=3\times 10^{4}$, $Z=5\times 10^{-4}\ \mathrm{Z}_{\odot}$')
#plt.fill_between([x1, x2], [7e3, 7e3], [1e3, 1e3], facecolor='k', alpha=0.3)
plt.xlabel(r'$n\ [\mathrm{cm^{-3}}]$')
plt.ylabel(r'$[\mathrm{H_{2}/H}]$')
plt.legend()
plt.xlim(x1, x2)
plt.ylim(1e-12, 1e-2)
plt.tight_layout()
plt.savefig(rep+'xH2_n_'+lab+'.pdf')
plt.close()

plt.figure()
plt.loglog(evo0['n']*0.93, evo0['X'][4], 'k-', lw = 3, label=r'$J_{21}=Z=0$', alpha = 0.9)
plt.loglog(evo1['n']*0.93, evo1['X'][4], '--', label=r'$J_{21}=3\times 10^{3}$, $Z=0$')
plt.loglog(evo2['n']*0.93, evo2['X'][4], '-', label=r'$J_{21}=3\times 10^{4}$, $Z=0$')
plt.loglog(evo3['n']*0.93, evo3['X'][4], '-.', label=r'$J_{21}=3\times 10^{4}$, $Z=4\times 10^{-4}\ \mathrm{Z}_{\odot}$')
plt.loglog(evo4['n']*0.93, evo4['X'][4], ':', label=r'$J_{21}=3\times 10^{4}$, $Z=5\times 10^{-4}\ \mathrm{Z}_{\odot}$')
#plt.fill_between([x1, x2], [7e3, 7e3], [1e3, 1e3], facecolor='k', alpha=0.3)
plt.xlabel(r'$n\ [\mathrm{cm^{-3}}]$')
plt.ylabel(r'$[\mathrm{H_{2}^{+}/H}]$')
plt.legend()
plt.xlim(x1, x2)
#plt.ylim(1e-12, 1)
plt.tight_layout()
plt.savefig(rep+'xH2plus_n_'+lab+'.pdf')
plt.close()

plt.figure()
plt.loglog(evo0['n']*0.93, evo0['X'][2], 'k-', lw = 3, label=r'$J_{21}=Z=0$', alpha = 0.9)
plt.loglog(evo1['n']*0.93, evo1['X'][2], '--', label=r'$J_{21}=3\times 10^{3}$, $Z=0$')
plt.loglog(evo2['n']*0.93, evo2['X'][2], '-', label=r'$J_{21}=3\times 10^{4}$, $Z=0$')
plt.loglog(evo3['n']*0.93, evo3['X'][2], '-.', label=r'$J_{21}=3\times 10^{4}$, $Z=4\times 10^{-4}\ \mathrm{Z}_{\odot}$')
plt.loglog(evo4['n']*0.93, evo4['X'][2], ':', label=r'$J_{21}=3\times 10^{4}$, $Z=5\times 10^{-4}\ \mathrm{Z}_{\odot}$')
#plt.fill_between([x1, x2], [7e3, 7e3], [1e3, 1e3], facecolor='k', alpha=0.3)
plt.xlabel(r'$n\ [\mathrm{cm^{-3}}]$')
plt.ylabel(r'$[\mathrm{H^{-}/H}]$')
plt.legend()
plt.xlim(x1, x2)
#plt.ylim(1e-12, 1)
plt.tight_layout()
plt.savefig(rep+'xHminus_n_'+lab+'.pdf')
plt.close()

#x1, x2 = 1e2, 1e6
plt.figure()
plt.loglog(evo0['n']*0.93, evo0['X'][5], 'k-', lw = 3, label=r'$J_{21}=Z=0$', alpha = 0.9)
plt.loglog(evo1['n']*0.93, evo1['X'][5], '--', label=r'$J_{21}=3\times 10^{3}$, $Z=0$')
plt.loglog(evo2['n']*0.93, evo2['X'][5], '-', label=r'$J_{21}=3\times 10^{4}$, $Z=0$')
plt.loglog(evo3['n']*0.93, evo3['X'][5], '-.', label=r'$J_{21}=3\times 10^{4}$, $Z=4\times 10^{-4}\ \mathrm{Z}_{\odot}$')
plt.loglog(evo4['n']*0.93, evo4['X'][5], ':', label=r'$J_{21}=3\times 10^{4}$, $Z=5\times 10^{-4}\ \mathrm{Z}_{\odot}$')
#plt.fill_between([x1, x2], [7e3, 7e3], [1e3, 1e3], facecolor='k', alpha=0.3)
plt.xlabel(r'$n\ [\mathrm{cm^{-3}}]$')
plt.ylabel(r'$[\mathrm{e^{-}/H}]$')
plt.legend()
plt.xlim(x1, x2)
plt.ylim(1e-10, 1)
plt.tight_layout()
plt.savefig(rep+'xe_n_'+lab+'.pdf')
plt.close()
"""
