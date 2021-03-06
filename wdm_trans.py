from txt import *

OmX = 0.315-0.048
def1={'m':20,'Om':OmX,'h':0.6774,'nu':1.12}

def cosmicT(z = 99, zd = 200.0, zre = 1100):
	Tre = 2.73*(1+zre)
	if z>=zd:
		T = Tre*(1+z)/(1+zre)
	else:
		Td = Tre*(1+zd)/(1+zre)
		T = Td*(1+z)**2/(1+zd)**2
	return T

def alpha(m, Om = def1['Om'], h = def1['h']): # in h^-1 Mpc, m in kev
	a = 0.049*(m)**-1.11*(Om/0.25)**0.11*(h/0.7)**1.22
	return a

def alpha_(m, Om = def1['Om'], h = def1['h']): # in h^-1 Mpc, m in kev
	a = 0.048*(m)**-1.15*(Om/0.4)**0.15*(h/0.65)**1.3
	return a

def mmu2(r, Om = def1['Om'], h = def1['h']):
	m = (r/0.31/(Om/0.3)**0.15/(h/0.65)**1.3)**(-1/1.15)
	return m

def mnu1(a, Om = def1['Om'], h = def1['h']):
	m = (a/0.049/(Om/0.25)**0.11/(h/0.7)**1.22)**(-1/1.11)
	return m

def Rs(m, Om = def1['Om'], h = def1['h']):
	r = 0.31*(Om/0.3)**0.15*(h/0.65)**1.3*(1/m)**1.15
	return r

def transf(k, mode = 0, key = def1['m']):
	if mode==0:
		a = alpha(key)
		nu = def1['nu']
	else:
		a = alpha_(key)
		nu = def1['nu']
	return (1+(a*k)**(2*nu))**(-5/nu)
		
def ngenicf(name = 'ics_L1', mode = 0, key = def1['m'], label='wdm', base='inputspec', ext='.txt'):
	d0 = np.array(retxt(base+'_'+name+ext,4,1,0))
	d = [[] for x in range(4)]
	d[0], d[2] = d0[0], d0[2]
	head = restr(base+'_'+name+ext)[0]
	d[1], d[3] = d0[1]*transf(d0[0],mode,key), d0[3]*transf(d0[2],mode,key)
	totxt(base+'_'+name+'_'+label+ext,d,head,1)
	return [d0, d, mode, key]

def plotspec1(d):
	mode = d[2]
	key = d[3]
	fig = plt.figure()
	plt.plot(d[0][2],d[0][3],label='cdm')
	plt.plot(d[1][2],d[1][3],label='wdm')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$k\ [h\ \mathrm{Mpc^{-1}}]$')
	plt.ylabel(r'$\Delta^{2}(k)\propto k^{3}P(k)\ [\mathrm{a.u.}]$')
	if mode==0:
		plt.title(r'$m_{\mathrm{WDM}}c^{2}='+str(key)+r'\ \mathrm{kev}$')
	else:
		plt.title(r'$\alpha=$'+str(key))
	plt.legend()
	plt.tight_layout()
	if mode==0:
		plt.savefig('powspec'+'_mWDM'+str(key)+'kev.pdf')
	else:
		plt.savefig('powspec'+'_alpha'+str(key)+'.pdf')
	#plt.show()

"""
def plotspec2(d):
	fig = plt.figure()
	plt.plot(d[0][0],d[0][1],label='cdm')
	plt.plot(d[1][0],d[1][1],label='wdm')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$k\ [h\mathrm{Mpc^{-1}}]$')
	plt.ylabel(r'$P(k)\ [\mathrm{a.u.}]$')
	plt.legend()
	plt.tight_layout()
"""
