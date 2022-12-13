from popIIIcluster import *
from scipy.interpolate import interp1d
from txt import *
import numpy.random as nrdm
import os

rzamsZ8 = np.array([+4.6071669041284602e-01, -1.2482170192596500e+00, +1.1079694078714399e+00, -2.1778915454158501e-01])
rzamsZ6 = np.array([-3.3237368336527201e-01, +4.6306428086272700e-01, +4.7820892172886598e-02, -9.6166327966181395e-03])
rzamsZ5 = np.array([-4.2008197743730302e-01, +7.1601089423309805e-01, -8.7939534847640000e-02, +1.4674567587019200e-02])
rzamsZ4 = np.array([-4.3620639472716299e-01, +8.4978687662735197e-01, -1.5993111424465600e-01, +2.7669651740859301e-02])
rzamsZ2 = np.array([-3.6870298415699199e-01, +9.4401642726269397e-01, -2.1017534522620901e-01, +3.7277105006281598e-02])

mcoZ8 = np.array([-9.5225083520154796e-01, +9.4357015904211505e-01, +4.7429337580493902e-01, -1.4351236491880501e-01])
mcoZ6 = np.array([-9.2137012965923404e-01, +8.8636637162070298e-01, +5.0592667147908199e-01, -1.4855704392558100e-01])
mcoZ5 = np.array([-9.2895961573096897e-01, +9.2331358192538004e-01, +4.7945677789672903e-01, -1.4307351573737101e-01])
mcoZ4 = np.array([-9.4823722226647700e-01, +9.7236386558568799e-01, +4.5283017654016999e-01, -1.3877489041026900e-01])
mcoZ2 = np.array([-7.3806577478145996e-01, +7.0263944811078005e-01, +5.0208830062187804e-01, -1.2376219079451400e-01])

#"""
mheZ8 = np.array([-7.2712749189720804e-01, +8.3176655071820205e-01, +4.7853206104410101e-01, -1.3980539673247300e-01])
mheZ6 = np.array([-5.2437528299791603e-01, +5.5932229149602297e-01, +5.7797948042505098e-01, -1.4589837221339999e-01])
mheZ5 = np.array([-4.1085136748565099e-01, +4.4292411042459900e-01, +6.2270607618818596e-01, -1.5313454615055200e-01])
mheZ4 = np.array([-3.9425104148142198e-01, +4.9002611835324500e-01, +5.7239586075853399e-01, -1.4093689615634999e-01])
mheZ2 = np.array([-3.8303715910210101e-01, +7.0150741599662503e-01, +2.9607706988601801e-01, -5.9315517041934301e-02])
"""
mheZ8 = np.array([-1.1, 1.5, 0.082, -0.054])
mheZ6 = np.array([-1.1, 1.6, 0.015, -0.041])
mheZ5 = np.array([-1.1, 1.5, 0.036, -0.044])
mheZ4 = np.array([-1.1, 1.4, 0.13, -0.067])
mheZ2 = np.array([-1.2, 1.9, -0.25, 0.022])
"""

temsZ8 = np.array([+1.2305856460495901e+00, +1.0134578934187600e+02, +9.1468769124419097e+01, +1.8034769714629199e+03])
tsgZ8 = np.array([+2.4927377844451301e-01, +2.1430677547749402e+00, +8.5972144678894594e+01, -1.4028396903502400e+02])
remsZ8 = np.array([-8.7660234566099804e-01, +1.9194682681297499e+00, -7.3392245022216696e-01, +1.7274738264264999e-01])
rchebZ8 = np.array([-1.8343478577029000e+01, +4.7783881233948101e+01, -4.0169462871126797e+01, +1.1460630304015901e+01])

temsZ6 = np.array([+1.5056408142507200e+00, +7.4994995850899201e+01, +8.3116022167663198e+02, +4.9040743535153001e+02])
tsgZ6 = np.array([+3.0667170921629699e-01, -2.2573597515177801e+00, +1.3262687600114401e+02, -4.3276720558498297e+01])
remsZ6 = np.array([-4.3397491671777300e-01, +1.3377594599674700e+00, -5.1421774804168696e-01, +1.5412807397524700e-01])
rchebZ6 = np.array([-3.6832391211242701e+00, +1.1084061918623000e+01, -9.7324743496085198e+00, +3.1373519906345702e+00])

temsZ5 = np.array([+1.5608149013984800e+00, +7.3567160614509504e+01, +8.1835557597302898e+02, +2.1085273006198800e+03])
tsgZ5 = np.array([+2.5190328005023799e-01, +2.3937574545125400e+00, +6.8538719197550407e+01, +3.4090149355802299e+02])
remsZ5 = np.array([-9.2456828602432894e-01, +2.5389191763683798e+00, -1.3751410532902100e+00, +3.5045831707709002e-01])
rchebZ5 = np.array([-7.8927435380812403e+00, +2.2851238614423899e+01, -2.0357967863018199e+01, +6.3389279382339296e+00])

temsZ4 = np.array([+1.5517508063147400e+00, +7.8762231265283901e+01, +6.6306742996156697e+02, +3.9780303119567002e+03])
tsgZ4 = np.array([+2.5995643378195399e-01, +1.3758291938603699e+00, +1.1749452241249701e+02, +7.3578237381483504e+01])
remsZ4 = np.array([-1.0109813992939201e+00, +2.8983622792063399e+00, -1.6345733256974599e+00, +4.0423625972161897e-01])
rchebZ4 = np.array([+1.6237359045461699e+00, +5.6527800027967900e-01, -3.1779617931783601e+00, +2.0224444898627301e+00])

temsZ2 = np.array([+1.5839255604016800e+00, +7.9580658743321393e+01, +6.2605049204196496e+02, +4.9099525171608302e+03])
tsgZ2 = np.array([+1.9589322976358600e-01, +7.8483404642259504e+00, +4.5085546669210199e+00, +6.5327035939216103e+02])
remsZ2 = np.array([-1.4359719418587300e+00, +4.1335756478450900e+00, -2.4940809801056401e+00, +5.9090072146249595e-01])
rchebZ2 = np.array([+6.1925984270592203e+00, -9.6076904024682896e+00, +5.6240568009227401e+00, -7.3097573061153098e-01])

lchebZ8 = np.array([+3.8790608169726498e-01, +5.4894852836696097e+00, -1.6450220435261300e+00, +2.0133843339575599e-01])
rgb0Z8 = np.array([+1.9892249955808999e-01, -3.7299109886527798e-01])
rgb1Z8 = np.array([+5.6071811071005395e-01, +3.2722143200640798e-02])

lchebZ6 = np.array([+5.5163281110466300e-01, +5.2726483070966701e+00, -1.5669073962725100e+00 +1.9653692229092701e-01])
rgb0Z6 = np.array([-1.1360958843644700e-01, -1.4734021414496001e-01])
rgb1Z6 = np.array([+6.2134621256327505e-01, -7.9113517758866905e-03])

lchebZ5 = np.array([+8.0634329880297295e-01 +4.7633074342534902e+00 -1.2175693546732000e+00 +1.1848267301948801e-01])
rgb0Z5 = np.array([+5.6659069526797702e-02 -2.9401839565042698e-01])
rgb1Z5 = np.array([+5.9229869792013501e-01 +1.5810282790358201e-02])

lchebZ4 = np.array([+1.0036121148808901e+00, +4.4477153532156501e+00, -1.0517047979711500e+00 +9.0669105159825694e-02])
rgb0Z4 = np.array([+9.0940251890292804e-03, -2.3234236135069800e-01])
rgb1Z4 = np.array([+5.9863657659087000e-01, +6.6527853775347704e-03])

lchebZ2 = np.array([+7.0984159927752299e-01, +5.2269835165592902e+00, -1.6253670199780199e+00 +2.1924040178420401e-01])
rgb0Z2 = np.array([-8.7074249061240602e-02, -1.6922018399772801e-01])
rgb1Z2 = np.array([+6.1896196949691396e-01, -5.7427447948830397e-03])

def Rrgb(M, fit0, fit1, fitl, fitr, Mcut=50, Md=8, Mu=160):
	r0 = Rzams(M, fitr, Md, Mu)
	lcheb = Rzams(M, fitl, Md, Mu, 0)
	rgb0 = Rzams(M, fit0, Md, Mu, 0)
	rgb1 = Rzams(M, fit1, Md, Mu, 0)
	r1 = 10**(rgb0 + lcheb*rgb1)
	r = r0*(M<Mcut) + r1*(M>=Mcut)
	return r

def Rzams(M, fitpara=rzamsZ6, Md=8, Mu=160, log=1, sign=1):
	od = len(fitpara)
	logM = np.log10(M)
	lm = np.array([logM**(sign*i) for i in range(od)])
	logR1 = np.sum(lm*fitpara)
	logMd = np.log10(Md)
	lmd = np.array([logMd**(sign*i) for i in range(od)])
	logRd = np.sum(lmd*fitpara)
	drdmd = np.sum(lmd[1:]*fitpara[1:]*np.arange(1, od)/logMd*sign)
	logR2 = (logM-logMd)*drdmd + logRd
	logMu = np.log10(Mu)
	lmu = np.array([logMu**(sign*i) for i in range(od)])
	logRu = np.sum(lmu*fitpara)
	drdmu = np.sum(lmu[1:]*fitpara[1:]*np.arange(1, od)/logMu*sign)
	logR3 = (logM-logMu)*drdmu + logRu
	logR = logR1 * (logM>logMd)*(logM<logMu) + logR2 * (logM<=logMd)
	logR += logR3 * (logM>=logMu)
	if log>0:
		return 10**logR # Rsun or Msun
	else:
		return logR

def tphase(M, fitpara=rzamsZ6, Md=8, Mu=160, log=0, sign=-1):
	od = len(fitpara)
	lm = np.array([M**(sign*i) for i in range(od)])
	t1 = np.sum(lm*fitpara)
	lmd = np.array([Md**(sign*i) for i in range(od)])
	td = np.sum(lmd*fitpara)
	drdmd = np.sum(lmd[1:]*fitpara[1:]*np.arange(1, od)*sign)/td
	logt2 = (np.log(M)-np.log(Md))*drdmd + np.log(td)
	t2 = np.e**logt2
	lmu = np.array([Mu**(sign*i) for i in range(od)])
	tu = np.sum(lmu*fitpara)
	drdmu = np.sum(lmu[1:]*fitpara[1:]*np.arange(1, od)/Mu*sign)
	t3 = (M-Mu)*drdmu + tu
	t = t1 * (M>Md)*(M<Mu) + t2 * (M<=Md)
	t += t3 * (M>=Mu)
	if log>0:
		return 10**t
	else:
		return t
	
def Mrem(M, mco=mcoZ6, mhe=mheZ6, Md=8, Mu=160):
	MCO = Rzams(M, mco, Md, Mu)
	MFe = (0.161767*MCO + 1.067055) * (MCO<=2.5)
	MFe += (0.314154*MCO + 0.686008) * (MCO>2.5)
	Mr0 = 0
	Mr0 += MFe * (MCO<=5)*(MCO>1.4)
	Mr0 += (MFe + (MCO-5)/2.6 * (M-MFe)) * (MCO>5)*(MCO<7.6)
	Mr0 += M * (MCO>7.6)
	MHe = Rzams(M, mhe, Md, Mu)
	Mr = 0
	Mr += Mr0 * np.logical_or(MHe<=45, MHe>135)
	Mr += 45 * (MHe>45) * (MHe<=65)
	return Mr

#"""	
lM = np.geomspace(1, 500, 100)
lR8 = [Rzams(m, rzamsZ8) for m in lM]
lR6 = [Rzams(m, rzamsZ6) for m in lM]
lR5 = [Rzams(m, rzamsZ5) for m in lM]
lR4 = [Rzams(m, rzamsZ4) for m in lM]
lR2 = [Rzams(m, rzamsZ2) for m in lM]
plt.figure()
plt.loglog(lM, lR8, label=r'$\log(Z/\rm Z_{\odot})=-8$')
plt.loglog(lM, lR6, '--', label=r'$\log(Z/\rm Z_{\odot})=-6$')
plt.loglog(lM, lR5, '-.', label=r'$\log(Z/\rm Z_{\odot})=-5$')
plt.loglog(lM, lR4, ':', label=r'$\log(Z/\rm Z_{\odot})=-4$')
plt.loglog(lM, lR2, ls=(0, (10, 5)), color='k', label=r'$\log(Z/\rm Z_{\odot})=-2$')
plt.xlabel(r'$m_{\star}\ [\rm M_{\odot}]$')
plt.ylabel(r'$R_{\star}\ [\rm R_{\odot}]$')
plt.tight_layout()
plt.legend()
plt.savefig('RZAMS_M.pdf')
plt.close()

lM = np.geomspace(1, 500, 100)
lR8 = [Rzams(m, remsZ8) for m in lM]
lR6 = [Rzams(m, remsZ6) for m in lM]
lR5 = [Rzams(m, remsZ5) for m in lM]
lR4 = [Rzams(m, remsZ4) for m in lM]
lR2 = [Rzams(m, remsZ2) for m in lM]
plt.figure()
plt.loglog(lM, lR8, label=r'$\log(Z/\rm Z_{\odot})=-8$')
plt.loglog(lM, lR6, '--', label=r'$\log(Z/\rm Z_{\odot})=-6$')
plt.loglog(lM, lR5, '-.', label=r'$\log(Z/\rm Z_{\odot})=-5$')
plt.loglog(lM, lR4, ':', label=r'$\log(Z/\rm Z_{\odot})=-4$')
plt.loglog(lM, lR2, ls=(0, (10, 5)), color='k', label=r'$\log(Z/\rm Z_{\odot})=-2$')
plt.xlabel(r'$m_{\star}\ [\rm M_{\odot}]$')
plt.ylabel(r'$R_{\star}\ [\rm R_{\odot}]$')
plt.tight_layout()
plt.legend()
plt.savefig('REMS_M.pdf')
plt.close()

lM = np.geomspace(1, 500, 100)
lR8 = [Rrgb(m, rgb0Z8, rgb1Z8, lchebZ8, rchebZ8, 50) for m in lM]
lR6 = [Rrgb(m, rgb0Z6, rgb1Z6, lchebZ6, rchebZ6, 50) for m in lM]
lR5 = [Rrgb(m, rgb0Z5, rgb1Z5, lchebZ5, rchebZ5, 50) for m in lM]
lR4 = [Rrgb(m, rgb0Z4, rgb1Z4, lchebZ4, rchebZ4, 50) for m in lM]
lR2 = [Rrgb(m, rgb0Z2, rgb1Z2, lchebZ2, rchebZ2, 100) for m in lM]
plt.figure()
plt.loglog(lM, lR8, label=r'$\log(Z/\rm Z_{\odot})=-8$')
plt.loglog(lM, lR6, '--', label=r'$\log(Z/\rm Z_{\odot})=-6$')
plt.loglog(lM, lR5, '-.', label=r'$\log(Z/\rm Z_{\odot})=-5$')
plt.loglog(lM, lR4, ':', label=r'$\log(Z/\rm Z_{\odot})=-4$')
plt.loglog(lM, lR2, ls=(0, (10, 5)), color='k', label=r'$\log(Z/\rm Z_{\odot})=-2$')
plt.xlabel(r'$m_{\star}\ [\rm M_{\odot}]$')
plt.ylabel(r'$R_{\star}\ [\rm R_{\odot}]$')
plt.tight_layout()
plt.legend()
plt.savefig('REHeB_M.pdf')
plt.close()

lM = np.geomspace(1, 500, 100)
lR8 = [Rzams(m, mcoZ8) for m in lM]
lR6 = [Rzams(m, mcoZ6) for m in lM]
lR5 = [Rzams(m, mcoZ5) for m in lM]
lR4 = [Rzams(m, mcoZ4) for m in lM]
lR2 = [Rzams(m, mcoZ2) for m in lM]
plt.figure()
plt.loglog(lM, lR8, label=r'$\log(Z/\rm Z_{\odot})=-8$')
plt.loglog(lM, lR6, '--', label=r'$\log(Z/\rm Z_{\odot})=-6$')
plt.loglog(lM, lR5, '-.', label=r'$\log(Z/\rm Z_{\odot})=-5$')
plt.loglog(lM, lR4, ':', label=r'$\log(Z/\rm Z_{\odot})=-4$')
plt.loglog(lM, lR2, ls=(0, (10, 5)), color='k', label=r'$\log(Z/\rm Z_{\odot})=-2$')
plt.xlabel(r'$m_{\star}\ [\rm M_{\odot}]$')
plt.ylabel(r'$M_{\rm c,CO}\ [\rm M_{\odot}]$')
plt.tight_layout()
plt.legend()
plt.savefig('McCO_M.pdf')
plt.close()

lR8 = [Rzams(m, mheZ8) for m in lM]
lR6 = [Rzams(m, mheZ6) for m in lM]
lR5 = [Rzams(m, mheZ5) for m in lM]
lR4 = [Rzams(m, mheZ4) for m in lM]
lR2 = [Rzams(m, mheZ2) for m in lM]
plt.figure()
plt.loglog(lM, lR8, label=r'$\log(Z/\rm Z_{\odot})=-8$')
plt.loglog(lM, lR6, '--', label=r'$\log(Z/\rm Z_{\odot})=-6$')
plt.loglog(lM, lR5, '-.', label=r'$\log(Z/\rm Z_{\odot})=-5$')
plt.loglog(lM, lR4, ':', label=r'$\log(Z/\rm Z_{\odot})=-4$')
plt.loglog(lM, lR2, ls=(0, (10, 5)), color='k', label=r'$\log(Z/\rm Z_{\odot})=-2$')
plt.xlabel(r'$m_{\star}\ [\rm M_{\odot}]$')
plt.ylabel(r'$M_{\rm c,He}\ [\rm M_{\odot}]$')
plt.tight_layout()
plt.legend()
plt.savefig('McHe_M.pdf')
plt.close()

m1, m2, nm = 5, 200, 500
lM = np.geomspace(m1, m2, nm)

lR8 = [tphase(m, temsZ8, log=0, sign=-1) for m in lM]
lR6 = [tphase(m, temsZ6, log=0, sign=-1) for m in lM]
lR5 = [tphase(m, temsZ5, log=0, sign=-1) for m in lM]
lR4 = [tphase(m, temsZ4, log=0, sign=-1) for m in lM]
lR2 = [tphase(m, temsZ2, log=0, sign=-1) for m in lM]
plt.figure()
plt.loglog(lM, lR8, label=r'$\log(Z/\rm Z_{\odot})=-8$')
plt.loglog(lM, lR6, '--', label=r'$\log(Z/\rm Z_{\odot})=-6$')
plt.loglog(lM, lR5, '-.', label=r'$\log(Z/\rm Z_{\odot})=-5$')
plt.loglog(lM, lR4, ':', label=r'$\log(Z/\rm Z_{\odot})=-4$')
plt.loglog(lM, lR2, ls=(0, (10, 5)), color='k', label=r'$\log(Z/\rm Z_{\odot})=-2$')
plt.xlabel(r'$m_{\star}\ [\rm M_{\odot}]$')
plt.ylabel(r'$t_{\rm EMS}\ [\rm Myr]$')
plt.xlim(m1, m2)
plt.tight_layout()
plt.legend()
plt.savefig('tEMS_M.pdf')
plt.close()

lR8 = [tphase(m, tsgZ8, log=0, sign=-1) for m in lM]
lR6 = [tphase(m, tsgZ6, log=0, sign=-1) for m in lM]
lR5 = [tphase(m, tsgZ5, log=0, sign=-1) for m in lM]
lR4 = [tphase(m, tsgZ4, log=0, sign=-1) for m in lM]
lR2 = [tphase(m, tsgZ2, log=0, sign=-1) for m in lM]
plt.figure()
plt.loglog(lM, lR8, label=r'$\log(Z/\rm Z_{\odot})=-8$')
plt.loglog(lM, lR6, '--', label=r'$\log(Z/\rm Z_{\odot})=-6$')
plt.loglog(lM, lR5, '-.', label=r'$\log(Z/\rm Z_{\odot})=-5$')
plt.loglog(lM, lR4, ':', label=r'$\log(Z/\rm Z_{\odot})=-4$')
plt.loglog(lM, lR2, ls=(0, (10, 5)), color='k', label=r'$\log(Z/\rm Z_{\odot})=-2$')
plt.xlabel(r'$m_{\star}\ [\rm M_{\odot}]$')
plt.ylabel(r'$t_{\rm SG}\ [\rm Myr]$')
plt.xlim(m1, m2)
plt.tight_layout()
plt.legend()
plt.savefig('tSG_M.pdf')
plt.close()

m2 = 300
m1 = 1
lM = np.geomspace(m1, m2, 1000)
lR8 = [Mrem(m, mcoZ8, mheZ8) for m in lM]
lR6 = [Mrem(m, mcoZ6, mheZ6) for m in lM]
lR5 = [Mrem(m, mcoZ5, mheZ5) for m in lM]
lR4 = [Mrem(m, mcoZ4, mheZ4) for m in lM]
lR2 = [Mrem(m, mcoZ2, mheZ2) for m in lM]
plt.figure()
plt.plot(lM, lM, 'k--', lw=3, alpha=0.2)
plt.plot(lM, lR8, label=r'$\log(Z/\rm Z_{\odot})=-8$')
plt.plot(lM, lR6, '--', label=r'$\log(Z/\rm Z_{\odot})=-6$')
plt.plot(lM, lR5, '-.', label=r'$\log(Z/\rm Z_{\odot})=-5$')
plt.plot(lM, lR4, ':', label=r'$\log(Z/\rm Z_{\odot})=-4$')
plt.plot(lM, lR2, ls=(0, (10, 5)), color='k', label=r'$\log(Z/\rm Z_{\odot})=-2$')
plt.xlabel(r'$m_{\star}\ [\rm M_{\odot}]$')
plt.ylabel(r'$M_{\rm rem}\ [\rm M_{\odot}]$')
plt.xlim(m1, m2)
plt.ylim(m1, m2)
#plt.xscale('log')
#plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.savefig('Mrem_M.pdf')
plt.close()

exit()
#"""

def Alope(R1, q1, e=0):
	y = (0.6*q1**(2./3) + np.log(1+q1**(1./3))) / (0.49*q1**(2./3))
	return y * R1 / (1-e)
	
def Lpos(q1, e1=1, e2=-1):
	f = lambda x: (1+q1)*x**5 + (3+2*q1)*x**4 + (3+q1)*x**3 + (1-e1-e2*q1)*x**2 -2*e2*q1*x - e2*q1
	sol = root(f, -0.1)
	return -sol.x[0]
"""
lq1 = np.geomspace(1e-3, 1, 100)
lL1 = [Lpos(q1) for q1 in lq1]
llope = [1/Alope(1, q1) for q1 in lq1]
plt.loglog(lq1, lL1, label=r'$a_{\rm L1}/a$')
plt.loglog(lq1, llope, '--', label=r'$R_{1}/A_{\rm L}$')
plt.xlabel(r'$q_{1}$')
plt.ylabel(r'$a_{\rm L1}/a$ and $R_{1}/A_{\rm L}$')
plt.legend()
plt.tight_layout()
plt.savefig('xL1_q1.pdf')
plt.close()
"""
def pwimf(alp):
	imf = lambda m: m**alp
	return imf

def midbin(l):
	return (l[1:]+l[:-1])*0.5
	
def bins(l):
	return l[1:]-l[:-1]

def masslist(M, N, imf, m1, m2, nb=10000, seed=23333, mmax=5e2, mode=0):
	lm = np.geomspace(m1, m2, nb+1)
	lmm = midbin(lm)
	dm = bins(lm)
	lF = np.hstack([[0], np.cumsum(imf(lmm)*dm)])
	lF = lF/lF[-1]
	invF = interp1d(lF, np.log10(lm))
	nrdm.seed(seed)
	if mode==0:
		r = nrdm.uniform(size=N)
		lM = 10**invF(r)
		lM[lM>mmax] = mmax
	else:
		mtot = 0
		lM = []
		count = 0
		while mtot<M:
			r = nrdm.uniform()
			m = np.min([10**invF(r), M-mtot, mmax])
			if count==0:
				m = min(m, M-m1)
			#if m<m1 and count>0:
			#	lM[count-1] += m
			#	break
			lM.append(m)
			mtot += m
			count += 1
	lM = np.array(sorted(lM, key=lambda x: -x))
	return lM

def gena0(a1, a2):
	return nrdm.uniform()*(a2-a1) + a1
	#return (a2/a1)**(nrdm.uniform()) * a1
	
def gena1(a1, a2):
	b = np.log(a2/a1)
	return np.exp(nrdm.uniform()*b + np.log(a1))

def circe():
	return 0

def popIed():
	return nrdm.uniform()**0.5

wf0 = lambda m: m #**(1./3)

def genkepler(nrdm, m, a, e):
	q = nrdm.uniform()
	ft_E = lambda x: x-e*np.sin(x)-2*np.pi*q
	sol = root(ft_E, np.pi)
	E = sol.x[0]
	r = a*(1-e*np.cos(E))
	ctheta = (np.cos(E)-e)/(1-e*np.cos(E))
	stheta = np.sqrt(1. - ctheta**2)*np.sign(nrdm.uniform()-0.5)
	x = r*ctheta
	y = r*stheta
	p = a*(1-e**2)*AU
	vx = -np.sqrt(GRA*m*Msun/p)*stheta/UV
	vy = np.sqrt(GRA*m*Msun/p)*(e+ctheta)/UV
	return [r, x, y, vx, vy]

def binaryhierarchy(lm, Rc, gena=gena0, gene=circe, seed=666, dRc = 2./3., wf=wf0, dfa=1e-1, mode=1, fang=0.5, Q = 1, fa2=0, fa1=0, dfv=0.1):
	N = len(lm)
	la = np.zeros(N)
	le = np.zeros(N)
	lv = np.zeros(N)
	lind = np.zeros(N, dtype='int')
	ncp = np.zeros(N, dtype='int')
	icp = np.zeros(N, dtype='int')
	lR = np.array([Rzams(m)*Rsun/AU for m in lm])
	lind[0] = -1
	lind[1] = 0
	ncp[0] = 1
	nrdm.seed(seed)
	#la[1] = (nrdm.uniform()*dRc + 1-dRc)*Rc
	la[1] = gena(Rc*(1-dRc), Rc)
	le[1] = gene()
	for i in range(2, N):
		lf = np.cumsum(wf(lm[:i])/np.sum(wf(lm[:i])))
		r = nrdm.uniform()
		sel = r/lf > 1
		if np.sum(sel)>0:
			ind = np.max(np.arange(i)[sel])
		else:
			ind = 0
		lind[i] = ind
		icp[i] = ncp[ind]
		ncp[ind] += 1
		e = gene()
		le[i] = e
		q1 = lm[ind]/lm[i]
		a1 = Alope(lR[ind], q1, e)
		if ind==0:
			q10 = lm[1]/lm[0]
			a2 = Rc
			a20 = (1-Lpos(q10)) * la[1]/(1+e)
		else:
			q10 = lm[ind]/lm[lind[ind]]
			a2 = Lpos(q10) * la[ind]/(1+e)
			a20 = a2
		if fa2>0:
			for k in range(i):
				if lind[k] == ind:
					qt = lm[k]/lm[ind]
					if qt<0.5:
						continue
					a2t = (1-Lpos(qt)) * la[k]/(1+e)
					if a2t/10 < a1:
						continue
					if a2t < a2:
						a2 = a2t
			a20 = min(a2, a20)
		if fa1>0:
			a1 = max(a1, a20/10)
		la[i] = gena(a1, a2)
	for i in range(N-1):
		sel = lind==i
		if np.sum(sel)>0:
			#print(icp[sel])
			licp = icp[sel]
			nrdm.shuffle(licp)
			icp[sel] = licp
			#print(icp[sel])
		
	if mode>0:
		lofs = nrdm.uniform(size=N)
	lpos = np.zeros((N, 3))
	lvel = np.zeros((N, 3))
	for i in range(1, N):
		if mode==0 or ncp[lind[i]]==0:
			m1, m2 = lm[i], lm[lind[i]]
			m = m1+m2
			r, x, y, vx, vy = genkepler(nrdm, m, la[i], le[i])
			v = np.sqrt(vx**2+vy**2)
			z = (nrdm.uniform()-0.5)*dfa*r
			vz = (nrdm.uniform()-0.5)*dfv*v
			R = (r**2 - z**2)**0.5
			vp = (v**2-vz**2)**0.5
			lpos[i] = lpos[lind[i]] + np.array([x*R/r, y*R/r, z])
			dvel = np.array([vx*vp/v, vy*vp/v, vz])
			lvel[i] = lvel[lind[i]] + dvel #*m2/m
			lvel[lind[i]] += 0 #- dvel*m1/m
			lv[i] = v
		else:
			if ncp[lind[i]]==0:
				theta = nrdm.uniform()*2*np.pi
			else:
				theta = (lofs[lind[i]] + nrdm.uniform()*fang/ncp[lind[i]] + icp[i]/ncp[lind[i]])*2*np.pi
			r = la[i]*(1+le[i])
			m1, m2 = lm[i], lm[lind[i]]
			m = m1+m2
			v = (GRA*m*Msun/(la[i]*AU) * (1.0-le[i])/(1.0+le[i]))**0.5/UV
			z = (nrdm.uniform()-0.5)*dfa*r
			vz = (nrdm.uniform()-0.5)*dfv*v
			R = (r**2 - z**2)**0.5
			vp = (v**2-vz**2)**0.5
			x, y = R*np.cos(theta), R*np.sin(theta)
			lpos[i] = lpos[lind[i]] + np.array([x, y, z])
			dvel = np.array([-np.sin(theta)*vp, np.cos(theta)*vp, vz])
			lvel[i] = lvel[lind[i]] + dvel*m2/m
			lvel[lind[i]] += - dvel*m1/m
			lv[i] = v
	#print(lpos)
	#print(UV)
	#print(la)
	#print(lR)
	#print(lvel)
	d = {}
	d['m'] = lm
	d['pos'] = lpos
	d['vel'] = lvel
	d['v'] = lv
	d['R'] = lR
	d['bp'] = [lind, la, le]
	d['Rc'] = Rc
	d['M'] = np.sum(lm)
	d['N'] = N
	if Q>0:
		vp = virialparam(d)
		#print('Q=2K/W=', vp)
		res = (Q/vp)**0.5
		d['vel'] = lvel*res
		d['v'] = lv*res
	return d

def writeinit(d, rep='./', sn=0, fn='init', ext='.dat'):
	l = [d['m'], d['R'], *d['pos'].T, *d['vel'].T, d['v']]
	totxt(rep+fn+'_'+str(sn).zfill(3)+ext, l)

def virialparam(d):
	lm, lpos, lvel0 = d['m'], d['pos'], d['vel']
	velc = np.array([np.sum(v*lm) for v in lvel0.T])/d['M']
	lvel = lvel0 - velc
	N = d['N']
	lv = np.sum(lvel**2, 1)
	K = np.sum(lv*lm)*Msun*0.5*UV**2
	W = 0
	for i in range(N):
		for j in range(i+1, N):
			r = np.linalg.norm(lpos[i] - lpos[j])
			W += GRA*lm[i]*lm[j]*Msun**2./(r*AU)
	return (2*K)/W

def plotdisk(d, fn, rep='./', fac=1.5, norm=1314, smin=2, mode=1, log=1, alpha=0.7,scl=10,vec=0, fcn = 1):
	lx, ly, lz = d['pos'].T
	R = d['Rc']*fac
	M = d['M']
	mlab = r'$M'+r'={:.0f}'.format(M)+r'\ \rm M_{\odot}$'
	if mode>0:
		lr = d['R']*norm #*1e5/R
		lr[lr<smin] = smin
	else:
		lr = smin
	N = d['N']
	lind = d['bp'][0]
	
	print(d['m'])
	#print(lind)
	print('N_star = {}'.format(len(lind)))
	if log>0:
		cs = np.log10(d['m'])
	else:
		cs = d['m']
	plt.figure()
	ax = plt.subplot(111)
	ax.set_aspect('equal','box')
	if fcn>0:
		for i in range(1, N):
			pos = d['pos'][[i, lind[i]]].T
			plt.plot(pos[0], pos[1], 'k--', lw=0.5)
	plt.scatter(lx, ly, s=lr**2, c=cs, cmap=plt.cm.plasma, alpha=alpha)
	cb = plt.colorbar()
	if log>0:
		plt.clim(0, np.log10(5e2))
		cb.set_label(r'$\log(m_{\star}\ [\rm M_{\odot}])$')
	else:
		plt.clim(1, 5e2)
		cb.set_label(r'$m_{\star}\ [\rm M_{\odot}]$')
	if vec>0:
		start = d['pos'][1:].T[:2]
		print(d['v'])
		lv = d['v'][1:]
		end = d['vel'][1:].T[:2]/lv * np.log10(lv*scl+1)
		plt.quiver(*start, *end, color='g')
	plt.xlabel(r'$x\ [\rm AU]$')
	plt.ylabel(r'$y\ [\rm AU]$')
	plt.xlim(-R, R)
	plt.ylim(-R, R)
	#plt.legend()
	plt.text(0.2*R, 0.85*R, mlab)
	plt.tight_layout()
	plt.savefig(rep+fn)
	plt.close()

def genic(rep, st=0, n=10, tf=1e2, ta=1e5, alp=-1.0, mmax=5e2, seed=2333, norm=1e8, mode=1, fang=0.5, Q=1, rN=0, dfa=1e-1, dfv=0.1, fecc=0, rfac=1, mm=1, m10=1, fa2=0, fa1=0, gena=gena0):
	if not os.path.exists(rep):
		os.makedirs(rep)
	N0 = int(N_t(tf)+0.5)
	Mc = M_t(ta)
	Rc = R_t(ta)*rfac
	imf0 = pwimf(alp)
	m20, A = Mmax(tf, ta, alp, m10)
	#m20 = min(m20, mmax)
	nrdm.seed(seed)
	ntot = 0
	for i in range(st, n):
		#print(i)
		if rN>0:
			N = 2 + int((N0-2)*2 * nrdm.uniform()+0.5)
		else:
			N = N0
		seed1 = int(nrdm.uniform()*norm)
		seed2 = int(nrdm.uniform()*norm)
		lm0 = masslist(Mc, N, imf0, m10, m20, seed=seed1, mmax=mmax, mode=mm)
		if fecc>0:
			gene = popIed
		else:
			gene = circe
		d = binaryhierarchy(lm0, Rc, gena=gena, seed=seed2, mode=mode, fang=fang, Q=Q, dfa=dfa, gene=gene, fa2=fa2, fa1=fa1)
		writeinit(d, rep, sn=i)
		ntot += d['N']
	return ntot

if __name__=='__main__':
	rN = 0
	#tf, ta = 1e8, 3e5
	#tf, ta = 4e1, 5e3
	tf, ta = 1e2, 1e5
	#tf, ta = 1e3, 1e6
	#rep = 'tf1e3ta1e6alp1/'
	#rep = 'tf1e8ta3e5alp235/'
	#rep = 'tf4e1ta5e3alp1_close/'
	#rep = 'tf1e2ta1e5alp1_tight_log/'
	rep = 'tf1e2ta1e5alp1_test/'
	#rep = 'tf1e3ta1e6alp1/'
	#alp = -2.35
	#alp = -1.57
	alp, m10 = -1, 1
	#dfa = 2.
	dfa = 1e-1
	dfv = 1e-1
	fecc = 1
	rfac = 1
	#rfac = R_t(1e1)/R_t(ta) 
	#rfac = R_t(5e3)/R_t(ta)
	#rfac = 1./3.
	fa2 = 0 #1
	fa1 = 1 #0
	gena = gena0 # 1
	imf0 = pwimf(alp)
	N0 = int(N_t(tf)+0.5)
	#m10 = 1
	m20, A = Mmax(tf, ta, alp, m10)
	mmax = 5e2
	Q = 0
	Mc = M_t(ta)
	Rc = R_t(ta)*rfac # AU
	rho = Mc*Msun/(4*np.pi*(Rc*AU)**3/3)
	print(N0)
	print('Total (upper) mass: {:.0f} ({:.0f}) Msun'.format(Mc, m20))
	print('Density: {:.2e} cm^-3'.format(rho*0.76/PROTON))
	#m20 = min(m20, 5e2)

	mm = 1
	seed1 = 521
	lm0 = masslist(Mc, N0, imf0, m10, m20, seed=seed1, mmax=mmax, mode=mm)
	seed2 = 1314
	
	Rvir = Rc*0.8/2
	t0 = tff(rho)
	t1 = tdyn(Mc, Rvir)
	t2 = trelax(N0, Mc, Rvir)
	t3 = tdecay(Mc, Rvir)
	print('Free-fall, dynamical, relaxation, decay timescales: {:.2e}, {:.2e}, {:.2e}, {:.2e} yr'.format(t0, t1, t2, t3))
	
	fac = 1
	mode=0
	fang = 0.5
	d = binaryhierarchy(lm0, Rc, gena=gena, seed=seed2, mode=mode, fang=fang, Q=1, fa2=fa2, fa1=fa1)
	#writeinit(d, sn=1024)
	#Q = virialparam(d)
	#print('Virial parameter Q = {:.3f}'.format(Q))
	fn = 'popIIIdisk_tf{:.0e}_ta{:.0e}.pdf'.format(tf, ta)
	plotdisk(d, fn, fac=fac, vec=0, fcn=1)
	st = 0 #1000
	ns = 1000
	seed = 2333
	ntot = genic(rep, st, ns, tf, ta, alp, rN=rN, dfa=dfa, mmax=mmax, Q=Q, fecc=fecc, rfac=rfac, mm=mm, m10=m10, seed=seed, fa2=fa2, fa1=fa1, gena=gena, dfv=dfv)
	print(ntot)
	
	test = 0
	if test==0:
		exit()
	
	Ns = 100000
	nb = 50
	lmt = masslist(Ns, imf0, m10, m20, mmax=mmax)
	ed = np.geomspace(m10, m20, nb)
	plt.figure()
	his, ed, pat = plt.hist(lmt, ed, label=r'$\alpha={:.2f}$, '.format(-alp)+r'$N_{\rm sample}='+'{:d}'.format(Ns)+'$', alpha=0.5)
	plt.xlabel(r'$m_{\star}\ [\rm M_{\odot}]$')
	plt.ylabel(r'Counts')
	plt.xscale('log')
	plt.ylim(0, np.max(his)*1.2)
	plt.legend()
	plt.tight_layout()
	plt.savefig('imfsamtest.pdf')
	plt.close()

	
