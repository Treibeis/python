from sclrelt import *

lZth = [1e-6, 1e-4, 1e-2, 1e-1]
lnth=[1, 10, 100]

def snapdata(lsn, drep, ext=1, bs=4, lZth = lZth, Zsun=0.02, metal=1, nump=0, lnth=lnth):
	ns = len(lsn)
	nZth = len(lZth)
	nnth = len(lnth)
	lz = np.zeros(ns)
	fionm = np.zeros(ns)
	fionv = np.zeros(ns)
	fmp = np.zeros((nnth, ns))
	met1 = np.zeros(ns)
	met2 = np.zeros(ns)
	mett = np.zeros(ns)
	met1v = np.zeros(ns)
	met2v = np.zeros(ns)
	mettv = np.zeros(ns)
	vff1 = np.zeros((nZth, ns))
	vff2 = np.zeros((nZth, ns))
	vfft = np.zeros((nZth, ns))
	i = 0
	for sn in lsn:
		fname = drep+'snapshot_'+str(sn).zfill(3)+'.hdf5'
		ds = yt.load(fname)
		lz[i] = ds['Redshift']
		le = bs*0.5*(1-ext)*1e3*np.ones(3)
		re = bs*0.5*(1+ext)*1e3*np.ones(3)
		ad = ds.box(le, re)
		mgas = np.array(ad[('PartType0', 'Masses')].to('g'))
		ngas = len(mgas)
		print('z = {}'.format(lz[i]))
		print('Total number of gas particles: ', ngas)
		if nump>0:
			sel = np.random.choice(ngas, min(int(nump),ngas),replace=False)
		else:
			sel = mgas>0
		mgas = mgas[sel]
		dgas = np.array(ad[('PartType0', 'Density')].to('g/cm**3'))[sel]
		vgas = mgas/dgas
		Mtot = np.sum(mgas)
		Vtot = np.sum(vgas)
		fion = np.array(ad[('PartType0','Primordial HII')])[sel]
		
		fionm[i] = np.sum(mgas*fion)/Mtot
		fionv[i] = np.sum(vgas*fion)/Vtot
		print('Ionized fraction: ', fionm[i], fionv[i])
		
		if metal>0:
			Zt = np.array(ad[('PartType0', 'Metallicity_00')]/Zsun)[sel]
			Z1 = np.array(ad[('PartType0', 'Metallicity_01')]/Zsun)[sel] # Pop II
			Z2 = np.array(ad[('PartType0', 'Metallicity_02')]/Zsun)[sel] # Pop III

			for j in range(nnth):
				nth = lnth[j]
				seld = dgas*0.76/PROTON>nth
				ndens = np.sum(seld)
				if ndens>0:
					mgasd = mgas[seld]
					fmp[j, i] = np.sum(mgasd[Zt[seld]<=lZth[0]])/np.sum(mgasd)
		
			met1[i] = np.sum(mgas*Z1)/Mtot
			met2[i] = np.sum(mgas*Z2)/Mtot
			mett[i] = np.sum(mgas*Zt)/Mtot
			
			met1v[i] = np.sum(mgas*Z1*Zsun)/Vtot
			met2v[i] = np.sum(mgas*Z2*Zsun)/Vtot
			mettv[i] = np.sum(mgas*Zt*Zsun)/Vtot
			for j in range(nZth):
				Zth = lZth[j]
				sel1 = Z1>Zth
				sel2 = Z2>Zth
				selt = Zt>Zth
				vff1[j, i] = np.sum(vgas[sel1])/Vtot
				vff2[j, i] = np.sum(vgas[sel2])/Vtot
				vfft[j, i] = np.sum(vgas[selt])/Vtot
		print('Metal-poor fraction: ', fmp[:, i])
		print('Mean metallicity: ', met1[i], met2[i], mett[i])
		print('Metal volume filling fraction: ')#, vff1[0][i], vff2[0][i], vfft[0][i])
		print(vff1[:,i])
		print(vff2[:,i])
		print(vfft[:,i])
		i += 1
		del mgas, dgas, vgas, fion, ad, ds
	d = {}
	d['lz'] = lz
	d['fion'] = [fionm, fionv]
	d['Zm'] = [met1, met2, mett]
	d['Zmv'] = [met1v, met2v, mettv]
	d['vf'] = [vff1, vff2, vfft]
	d['mp'] = fmp
	return d
	
def logft(t, dt, t0, L):
	return L/(1+np.exp(-(t-t0)/dt))
	
zre, dz = 7.56542697, 1.60677831
	
def xefit(z, xer=2e-4, zre=zre, dz=dz):
	y = (1+z)**1.5
	yre = (1+zre)**1.5
	dy = 1.5*(1+zre)**0.5 * dz
	f = 0.5*(1-xer)*(1+np.tanh((yre-y)/dy)) + xer
	return f
	
def xefit0(z, xe6, alp):
	f = (1.0-(1-xe6)*((1+z)/7)**3) * (z<=6)
	f += xe6 * np.exp(alp*(6.0-z)) * (z>6)
	return f
	
lsn = [1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 22, 23]
	
if __name__=='__main__':	
	#drep = '/media/friede/Seagate Backup Plus Drive/HighzBH/FDbox_Lseed/'
	drep = '/media/friede/Seagate Backup Plus Drive/HighzBH/FDzoom_Hseed/'
	#rep = './'
	#rep = 'rock_box/snapdata/'
	rep = 'rock_zoom/'
	mode = 1
	lsn_ = [0] + lsn
	#lsn_ = np.linspace(0, 23, 24, dtype='int')
	ext = 0.3
	metal = 1
	
	#d = snapdata(lsn_, drep, ext, metal=metal)
	#exit()
	
	if not os.path.exists(rep):
		os.makedirs(rep)
	
	if mode==0:
		d = snapdata(lsn_, drep, ext, metal=metal)
		lz = d['lz']
		fion = d['fion']
		Zm = d['Zm']
		Zmv = d['Zmv']
		vf1, vf2, vft = d['vf']
		fmp = d['mp']
		#"""
		totxt(rep+'zbase.txt', [lz])
		totxt(rep+'fion.txt', fion)
		if metal>0:
			totxt(rep+'fmp.txt', fmp)
			totxt(rep+'Zmean.txt', Zm)
			totxt(rep+'Zmean_volume.txt', Zmv)
			totxt(rep+'vf1.txt', vf1)
			totxt(rep+'vf2.txt', vf2)
			totxt(rep+'vft.txt', vft)
		#"""
	else:
		lz = np.array(retxt(rep+'zbase.txt',1,0,0))[0]
		#fion = np.array(retxt(rep+'fion.txt',2,0,0))
		fion = np.array(retxt('fion.txt',2,0,0))
		if metal>0:
			fmp = np.array(retxt(rep+'fmp.txt',3,0,0))
			Zm = np.array(retxt(rep+'Zmean.txt',3,0,0))
			Zmv = np.array(retxt(rep+'Zmean_volume.txt',3,0,0))
			vf1 = np.array(retxt(rep+'vf1.txt',4,0,0))
			vf2 = np.array(retxt(rep+'vf2.txt',4,0,0))
			vft = np.array(retxt(rep+'vft.txt',4,0,0))
			lall = [lz, vf1[1], vf2[1], vft[1], Zm[0], Zm[1], Zm[2]]
			head = ['Redshift', 'F(Pop2)', 'F(Pop3)', 'F(all)', 'Zmean(Pop2)', 'Zmean(Pop3)', 'Zmean(all)']
			totxt(rep+'metal_z.txt', lall, head, 1, 0)

	#reps = rep
	reps = './'
	sfr2 = np.array(retxt(reps+'popII_sfr.txt',4))
	sfr3 = np.array(retxt(reps+'popIII_sfr.txt',4))
	x1, x2 = 4, 20
	plt.figure()
	for i in range(3):
		f = fmp[i]
		nth = lnth[i]
		plt.plot(lz[f>0], f[f>0], ls=lls[i], label=r'$n_{\rm H}>'+r'{:.0f}'.format(nth)+r'\ \rm cm^{-3}$')
	plt.plot(1/sfr2[0]-1, sfr3[2]/(sfr2[2]+sfr3[2]), ls=':', label=r'$\dot{\rho}_{\star,\rm PopIII}/\dot{\rho}_{\star}$')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$f_{\rm mp} (Z<10^{-6}\ \rm Z_{\odot})$')
	plt.xlim(x1, x2)
	#plt.ylim(y1, y2)
	plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'fmp_z.pdf')
	plt.close()

	fit = curve_fit(xefit, lz, fion[1], [2e-4, 8, 4], bounds=([2e-4, 6, 1], [3e-4, 10, 10]))
	print(fit[0])
	xer, zre, dz = fit[0]
	x1, x2 = np.min(lz), 14#np.max(lz)
	y1, y2 = 0, 1
	plt.figure()
	plt.plot(lz, fion[0], label='Mass-weighted')
	plt.plot(lz, fion[1], '--', label='Volume-weighted')
	lz_ = np.linspace(x1, x2, 1000)
	#obs11 = xefit(lz_, xer, 8.33, 3.38)
	#obs12 = xefit(lz_, xer, 6.69, 3.38)
	obs11 = xefit(lz_, xer, 8.2, 1e-3)
	obs12 = xefit(lz_, xer, 6.68, 1e-3)
	obs21 = xefit(lz_, xer, 8.2, 0.94)
	obs22 = xefit(lz_, xer, 6.68, 0.94)
	dobs = np.array([obs11, obs12, obs21, obs22])
	#dobs = np.array([obs21, obs22])
	obs1 = np.max(dobs, axis=0)
	obs2 = np.min(dobs, axis=0)
	#dz = 1.0
	#print(xefit(0, xer=xer, zre=zre, dz=dz), xefit(6, xer=xer, zre=zre, dz=dz))
	plt.plot(lz_, xefit(lz_, xer, zre, dz), 'k-.', label='Fit by tanh')
	plt.plot(lz_, xefit(lz_, xer, zre, 0.94), 'k:', label=r'$\hat{f}_{\rm ion}$')
	plt.fill_between(lz_, obs1, obs2, fc='gray', alpha=0.5, label=r'CMB+FRBs')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$f_{\rm ion}$')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	#plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'fion_z.pdf')
	plt.close()
		
	lt = np.array([TZ(z)/YR/1e9 for z in lz])
	
	fit = curve_fit(logft, lt, fion[1], [0.5, 1.0, 0.8])
	print(fit[0])
	dt, t0, L = fit[0]
	zion = 6
	zos = 10
	tion = TZ(zion)/1e9/YR
	tos = TZ(zos)/YR/1e9
	dt_ = dt*(tion-tos)/(t0-tos)/2
	print('End of reionization: {:.3f} ({:.3f}) Gyr'.format(tion, 2*t0-tos))
	print('gamma = {:.6f} ({:.6f}) Gyr^-1'.format(dt_, dt))
	zion_ = ZT(np.log10(2*t0-tos))
	
	x1, x2 = np.min(lt), np.max(lt)
	y1, y2 = 1e-4, 1
	plt.figure()
	plt.plot(lt, fion[0], label='Mass-weighted')
	plt.plot(lt, fion[1], '--', label='Volume-weighted')
	lt_ = np.linspace(x1, x2, 100)
	plt.plot(lt_, logft(lt_, dt, t0, L), 'k-.', label='Logistic fit')
	#plt.plot([x1, x2], [0.8]*2, 'k-', lw=0.5)
	plt.plot([2*t0-tos]*2, [y1, y2], 'k:', label=r'$t_{\rm ion}$ ($z_{\rm ion}\simeq '+'{:.1f}'.format(zion_)+'$)')
	plt.annotate(r'$t_{\rm os}$', (tos, 0.01), (tos-0.01, 0.15), arrowprops=dict(width=2,headwidth=5., color='k'))
	#plt.plot([tos], [0.03], ls='none', marker=r'$\downarrow$')
	#plt.text(tos, 0.03, r'$t_{\rm os}$')
	plt.xlabel(r'$t\ [\rm Gyr]$')
	plt.ylabel(r'$f_{\rm ion}$')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	#plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'fion_t.pdf')
	plt.close()
	
	if metal==0:
		exit()
	
	cf = np.array(retxt('collapse_rat_z.txt',4))
	cf_z = interp1d(cf[0], cf[1])
	
	y1, y2 = 3e-7, 3
	x1, x2 = np.min(lz), 20
	sel = lz<=20
	plt.figure()
	plt.plot(lz, Zm[2], 'k-', label='Total')
	plt.plot(lz, Zm[1], '--', label='Pop III')
	plt.plot(lz, Zm[0], '-.', label='Pop II')
	plt.plot(lz[sel], Zm[2][sel]/cf_z(lz[sel]), 'k:', label=r'$\langle Z\rangle_{\rm col}(M_{\rm halo}>M_{\rm th}^{\rm mol})$')#label='Collapsed')
	plt.plot([x1, x2], [1e-4]*2, color='k', ls=(0, (10, 5)), label=r'$Z_{\rm th}=10^{-4}\ \rm Z_{\odot}$')
	plt.fill_between([x1, x2], [10**-3.5]*2, [1e-6]*2, fc='gray', alpha=0.5, label=r'$Z_{\rm crit}\sim 10^{-6}-10^{-3.5}\ \rm Z_{\odot}$')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\langle Z\rangle\ [\rm Z_{\odot}]$')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.yscale('log')
	plt.legend(ncol=2)
	plt.tight_layout()
	plt.savefig(rep+'Zmean_z.pdf')
	plt.close()
	
	#y1, y2 = 3e-8, 3e-3
	x1, x2 = np.min(lz), 20
	sel = lz<=20
	plt.figure()
	plt.plot(lz, Zmv[2], 'k-', label='Total')
	plt.plot(lz, Zmv[1], '--', label='Pop III')
	plt.plot(lz, Zmv[0], '-.', label='Pop II')
	plt.plot(lz[sel], Zmv[1][sel]/cf_z(lz[sel])*18*np.pi**2, 'k:', label=r'$\langle \rho_{\rm met}\rangle_{\rm col}(M_{\rm halo}>M_{\rm th}^{\rm mol})$')#label='Collapsed')
	#plt.plot([x1, x2], [1e-4]*2, 'k:', label=r'$Z_{\rm th}=10^{-4}\ \rm Z_{\odot}$')
	lzb = np.linspace(x1, x2, 100)
	lrho = np.array([rhom(1/(1+z))*0.048/0.315 for z in lzb])
	plt.fill_between(lzb, lrho*1e-6*Zsun, lrho*10**-3.5*Zsun, fc='gray', alpha=0.5, label=r'$Z_{\rm crit}\sim 10^{-6}-10^{-3.5}\ \rm Z_{\odot}$')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\langle \rho_{\rm met}\rangle\ [\rm g\ cm^{-3}]$')
	plt.xlim(x1, x2)
	#plt.ylim(y1, y2)
	plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'Zmean_vol_z.pdf')
	plt.close()
	
	pisnfac = (40/(2+5))**(3*0.38)
	vfr1 = np.array(retxt('vfref1.txt',2))
	vfr2 = np.array(retxt('vfref2.txt',2))
	vfr3 = np.array(retxt('vfref3.txt',2))
	f1 = interp1d(*vfr2)
	f2 = interp1d(*vfr3)
	lzj = np.linspace(6, 20, 100)
	
	y1, y2 = 1e-5, 70
	x2 = 20
	plt.figure()
	plt.plot(lz, vft[1], 'k-', lw=3, label='Total ($Z>10^{-4}\ \mathrm{Z_{\odot}}$)')
	plt.plot(lz, vf2[1], '--', lw=3, label='Pop III ($Z>10^{-4}\ \mathrm{Z_{\odot}}$)')
	plt.plot(lz, vf1[1], '-.', lw=3, label='Pop II ($Z>10^{-4}\ \mathrm{Z_{\odot}}$)')
	plt.plot(lz, vf1[1]+vf2[1]*pisnfac, ls='-', marker=r'$\downarrow$', color='k', markersize=10, label=r'All in PISNe')
	#plt.plot(lz, vft[1], 'k--', label='Total ($Z>10^{-4}\ \mathrm{Z_{\odot}}$)')
	plt.plot(lz, vft[2], 'k--', label='Total ($Z>10^{-2}\ \mathrm{Z_{\odot}}$)')
	plt.plot(lz, vft[3], 'k-.', label='Total ($Z>10^{-1}\ \mathrm{Z_{\odot}}$)')
	plt.plot(cf[0], cf[1]/(18*np.pi**2), 'k:', label=r'$\mathcal{F}_{\rm col}(M_{\rm halo}>M_{\rm th}^{\rm mol})$', lw=3)
	plt.fill_between(lzj, f1(lzj), f2(lzj), fc='g', label=r'JCS13 ($Z>0$)', alpha=0.5)
	plt.plot(vfr1[0], vfr1[1], ls=(0, (10, 5)), color='r', label=r'PA14 ($Z>10^{-4}\ \mathrm{Z_{\odot}}$)')
	#plt.plot([7, 7], [2.65e-2+2.33e-3, 3.81e-2+5.55e-3], 'X', color='r', label='WJ11 ($Z>10^{-4\ (6)}\ \mathrm{Z_{\odot}}$)')
	plt.plot([7], [2.65e-2+2.33e-3], 'X', color='r', label='WJ11 ($Z>10^{-4}\ \mathrm{Z_{\odot}}$)')
	plt.plot([7.6], [6.22e-2], '*', color='orange', label='XH16 ($Z>10^{-4}\ \mathrm{Z_{\odot}}$)')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\mathcal{F}$')
	plt.xlim(x1, x2)
	plt.ylim(y1, y2)
	plt.yscale('log')
	plt.legend(ncol=2)
	plt.tight_layout()
	plt.savefig(rep+'vf_z.pdf')
	plt.close()
	
	totxt(rep+'metal_ratio.txt', [lz, Zm[1]/Zm[2]] + [vf2[i]/vft[i] for i in range(4)])
