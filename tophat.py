from cosmology import *
from bdms import *
from radcool import *
import chemi as chemi1
import cheminet as chemi2
from txt import *
import os
import multiprocessing as mp
import time
import hmf
import hmf.wdm
from numba import njit

proton = PROTON/GeV_to_mass
electron = ELECTRON/GeV_to_mass

def debye(eps):
	return 68-2*np.log(eps/1e-6)

def sigmilli(eps, mx, mb, norm=1e20):
	Mx, Mb = mx*GeV_to_mass, mb*GeV_to_mass
	ut = Mx*Mb/(Mx+Mb)
	out = 2*np.pi*(1./137)**2*eps**2*debye(eps)/ut**2
	return out/norm * (HBAR*SPEEDOFLIGHT)**2
	
def eps_m_f(mx, f, emax=1e-4):
	a = 1.7e-4*(mx/0.3)*(1e-2/f)**(3./4)
	b = a*(a>=emax) + emax*(a<emax)
	out = a * (mx<=0.04*(f/1e-2)) + b * (mx>0.04*(f/1e-2))
	#a = 6e-7*(mx/1e-3)*(1e-2/f)**(3./4)
	return a*(a<=1e-4)*(a>=1e-6) + 1e-4*(a>1e-4)+ 1e-6*(a<1e-6)
	
def sigmc(eps, mx, f, norm=1e20, xe=1e-4):
	#Tmin = T_b(17)/(1+mx/proton/f)
	#dT = 0.01*Tmin
	#p = Q_IDMB_mc(1, 0, 4, 4+dT, f, PROTON, mx*GeV_to_mass)
	#p0 = Q_IDMB(1, 0, Tmin, Tmin*1.01, PROTON, mx*GeV_to_mass)
	#e = Q_IDMB_mc(1, 0, 4, 4+dT, f, ELECTRON, mx*GeV_to_mass)
	#e0 = Q_IDMB(1, 0, Tmin, Tmin*1.01, PROTON, mx*GeV_to_mass)
	T = T_b(17)/(1+6.*f/mx)
	T0 = T_b(17)/(1+6./mx)
	a = sigmilli(eps, mx, proton) #*abs(p/p0)
	b = sigmilli(eps, mx, electron)*xe #*abs(e/e0)
	return (a+b)*f * (T0/T)**(3/2)
	
zi_tag = 2

if zi_tag==0:
	z0_default = 300
	x0_default = [1., 5e-4, 2.5e-19, 2e-11, 3e-16] + \
			 [5e-4, 1.0, 1.4e-20, 0.0] + \
			 [1.0, 5e-4, 8.4e-11] + \
			 [1.0, 1e-4, 0, 0, 1e-14]
elif zi_tag==1:
	z0_default = 800
	x0_default = [1., 2e-3, 1.6e-20, 1.3e-12, 4e-18] + \
			 [2e-3, 1.0, 1.4e-18, 0.0] + \
			 [1.0, 2e-3, 3.4e-12] + \
			 [1.0, 2.2e-9, 0, 0, 0]
else:
	z0_default = 1000
	x0_default = [.9, 1e-1, 1.5e-19, 1.2e-13, 7.6e-18] +\
			 [1e-1, 1.0, 2.7e-16, 0.0]+ \
			 [0.9, 1e-1, 3.4e-13] + \
			 [1.0, 1e-10, 0, 0, 0]


def initial(z0 = z0_default, v0 =30, mode = 0, Mdm = 0.3, sigma = 8e-20, x0 = x0_default, z00 = 1100, Om = 0.315, Ob = 0.048, h = 0.6774, T0 = 2.726, vmin = 0.0):
	#0: XH, 1: XH+, 2: XH-, 3: XH2, 4: XH2+,
	#5: Xe, 6: XHe, 7: XHe+, 8: XHe++, 
	#9: XD, 10: XD+, 11: XHD,
	#12: Li, 13: Li+, 14: Li-, 15: LiH+, 16: LiH
	d = {}
	if mode!=0:
		d0 = thermalH(z00, z0, v0, Mdm, sigma, Om, Ob, T0=T0, h=h)
		d['Tb'] = d0['Tb'][-1]
		d['Tdm'] = d0['Tdm'][-1]
		uth = (d['Tb']*BOL/PROTON+d0['Tdm'][-1]*BOL/(Mdm*GeV_to_mass))**0.5
		d['vbdm'] = max(d0['v'][-1], uth*vmin)
	else:
		d['Tb'] = T_b(z0, T0=T0)
		d['Tdm'] = T_dm(z0, Mdm, T0=T0)
		d['vbdm'] = vbdm_z(z0, v0)
	d['X']  = x0
	return d

init = initial()

def M_T(T, z, delta, Om = 0.315):
	return (T/(delta*Om/(200*0.315))**(1/3)*10/(1+z)/190356.40337133306)**(3/2)*1e10

delta0 = 200.

def coolt(Tb_old, Tdm_old, v_old, nb, nold, rhob_old, rhodm_old, z, mode, Mdm, sigma, gamma, Om, Ob = 0.048, h = 0.6774, X = 0.76, J_21 = 0, T0 = 2.726, vmin = 0.0):
	xh = 4*X/(1+3*X)
	dTs_dt = [0]*3
	if mode!=0:
		uth = (Tb_old*BOL/PROTON+Tdm_old*BOL/(Mdm*GeV_to_mass))**0.5
		v = max(v_old, uth*vmin)
		dTs_dt = bdmscool(Tdm_old, Tb_old, v, rhob_old, rhodm_old, Mdm, sigma, gamma, X)
	dTb_dt = cool(Tb_old, nb, nold, J_21, z, gamma, X, T0) + dTs_dt[1]
	if abs(dTb_dt) <= Tb_old/TZ(0):
		return TZ(0)
	else:
		return -Tb_old/dTb_dt

@njit	
def GammaC0(over, xe):
	#logG = np.log10(over) * 1.115724778945266 -18.31281413734767
	return 0.0 #10**logG * xe

def evolve(Mh = 1e6, zvir = 20, z0 = z0_default, v0 = 30, mode = 0, fac = 1.0, Mdm = 0.3, sigma = 8e-20, num = int(1e3), epsT = 1e-3, epsH = 1e-2, dmax = 18*np.pi**2, gamma = 5/3, X = 0.76, D = 2.38e-5, Li = 4.04e-10, T0 = 2.726, Om = 0.315, Ob = 0.048, h = 0.6774, dtmin = YR, J_21=0.0, Tmin = 1., vmin = 0.0, nmax = int(1e6), init =init, hat=1, fnth=1):
	start = time.time()
	#print(Mdm, sigma, init['Tb'], init['Tdm'], init['vbdm'])
	xh = 4.0*X/(1.0+3.0*X)
	xhe, xd, xli = 1-xh, D, Li
	refa = np.array([xh]*6+[xhe]*3+[xd]*3+[xli]*5)
	t0 = TZ(z0)
	t1 = TZ(zvir)
	tpost = min(fac/H(1/(1+zvir), Om, h), TZ(0)-t1)
	tmax = t1 + tpost
	dt0 = (tmax-t0)/num
	dt0_ = (tmax-t0)/nmax
	#print('Time step: {} yr'.format(dt0/YR))
	if hat>0:
		rhodm_z = lambda x: rho_z(x, zvir, dmax, Om, h)*(Om-Ob)/Om
		rhob_z = lambda x: rho_z(x, zvir, dmax, Om, h)*Ob/Om
	else:
		rhodm_z = lambda x: rhom(1/(1+x), Om, h)*(Om-Ob)/Om
		rhob_z = lambda x: rhom(1/(1+x), Om, h)*Ob/Om
	Ns = len(init['X'])
	lz = [z0]
	lt = [t0]
	lTb = [max(init['Tb'], Tmin)]
	lTdm = [max(init['Tdm'], Tmin/1e10)]
	lv = [init['vbdm']]
	lrhob = [rhob_z(z0)]
	lrhodm = [rhodm_z(z0)]
	lX = [[x] for x in init['X']]
	t_cum, count, total = t0, 0, 0
	tag0 = 0
	tag1 = 0
	TV = Tvir(Mh, zvir, delta0)
	VV = Vcir(Mh, zvir, delta0)
	tag2 = 0
	if lTb[0]>TV:
		tag2 = 1
	Tb_V = TV
	pV = []
	pV_pri = []
	para = [Mdm, sigma, gamma, Om, Ob, h, X, J_21, T0, vmin]
	tcool = 0.0
	tffV = 1/H(1/(1+zvir), Om, h) #tff(zvir, dmax)
	#yy = np.zeros(Ns, dtype='float')
	tagt = 0
	while t_cum < tmax or z>zvir:
		if count==0:
			z = z0
			dt_T = dtmin
			yy = np.array([x[count] for x in lX])
			mgas = mmw(yy[5], yy[7], X)*PROTON
			nb = lrhob[0]/mgas
			nold = yy * refa
			Tb_old = lTb[count]
			Tdm_old = lTdm[count]
			v_old = lv[count]
			rhob_old, rhodm_old = lrhob[count], lrhodm[count]
			dlnrho_dt_old = Dlnrho(t0, t0+dtmin/2.0, zvir, dmax, hat=hat)
			dTs_dt_old = [0]*3
			dTs_dt = [0]*3
			if mode!=0:
				dTs_dt_old = bdmscool(Tdm_old, Tb_old, v_old, rhob_old, rhodm_old, Mdm, sigma, gamma, X)
			dTb_dt_old = cool(max(Tb_old, 10*Tmin), nb, nold*nb, J_21, z0, gamma, X, T0) \
						+ dTs_dt_old[1] + dlnrho_dt_old*(gamma-1)*Tb_old \
						+ GammaC0(rhob_old/rhom(1/(1+z0))*Om/Ob, yy[5])*max(T0*(1+z0)-Tb_old, 0)
						#+ GammaC(z0, Om, Ob, h = h, X = X, T0 = T0)*(T0*(1+z0)-Tb_old)
			dTdm_dt_old = dTs_dt_old[0] + dlnrho_dt_old*(gamma-1)*Tdm_old
			dv_dt_old = dTs_dt_old[2] + dlnrho_dt_old*(gamma-1)*v_old/2
		else:
			if (t_cum-t0)/(t1-t0)<0.01:
				dt_T = dt0_
			else:
				dt_T = dt0
			if abs(dTb_dt_old*dt_T)>epsT*Tb_old and Tb_old>10*Tmin:
				dt_T = epsT*Tb_old/abs(dTb_dt_old)
				#if dt_T < dt0_ and Tb_old<=Tmin*100:
				#	dt_T = dt0_
		if dt_T + t_cum>t1 and tagt==0:
			dt_T = t1 - t_cum
			tagt = 1
		if dt_T + t_cum>tmax:
			dt_T = tmax - t_cum

		if count==0:
			Cr0, Ds0 = np.zeros(Ns,dtype='float'), np.zeros(Ns,dtype='float')
			abund0 = chemi1.chemistry1(Tb_old, nold*nb, dt_T, epsH, J_21, Ns, xh*nb, xhe*nb, xd*nb, xli*nb, Cr0, Ds0, z = z, T0 = T0, fnth=fnth)
			Cr0, Ds0 = abund0[5], abund0[6]
		else:
			Cr0, Ds0 = abund[5], abund[6]

		nold = yy * refa
		abund = chemi1.chemistry1(Tb_old, nold*nb, dt_T, epsH, J_21, Ns, xh*nb, xhe*nb, xd*nb, xli*nb, Cr0, Ds0, z = z, T0 = T0, fnth=fnth)
		#abund = chemi2.chemistry1(Tb_old, nold*nb, dt_T, epsH, J_21, Ns, xh*nb, xhe*nb, xd*nb, xli*nb, z = z, T0 = T0)
		nold = abund[0]/nb
		for x in range(Ns):
			if refa[x]!=0:
				yy[x] = nold[x]/refa[x]
			else:
				yy[x] = 0.0
		mgas = mmw(yy[5], yy[7], X)*PROTON
		#if count<10:
		#	print(nold, Tb_old)
		t_cum += abund[1]
		z = z_t(t_cum)
		dlnrho_dt = Dlnrho(t_cum, t_cum + abund[1]/2.0, zvir, dmax, hat=hat)
		uth = (Tb_old*BOL/PROTON+Tdm_old*BOL/(Mdm*GeV_to_mass))**0.5
		if mode!=0: #and (Tb_old>Tdm_old or v_old>vmin*uth):
			dTs_dt = bdmscool(Tdm_old, Tb_old, v_old, rhob_old, rhodm_old, Mdm, sigma, gamma, X)
		dTb_dt = cool(max(Tb_old,10*Tmin), nb, nold*nb, J_21, z, gamma, X, T0) + dTs_dt[1] + GammaC0(rhob_old/rhom(1/(1+z))*Om/Ob, yy[5])*max(T0*(1+z)-Tb_old, 0) #+ GammaC(z, Om, Ob, h = h, X = X, T0 = T0)*(T0*(1+z)-Tb_old)
		if tag0==0:
			dTb_dt += dlnrho_dt*(gamma-1)*Tb_old
		dTdm_dt = dTs_dt[0] + dlnrho_dt*(gamma-1)*Tdm_old
		dv_dt = dTs_dt[2] + dlnrho_dt*(gamma-1)*v_old/2
		Tb_old = max(Tb_old + (dTb_dt + dTb_dt_old)*abund[1]/2.0, Tmin)
		Tdm_old = max(Tdm_old + (dTdm_dt + dTdm_dt_old)*abund[1]/2.0, 0.0)
		v_old = max(v_old + (dv_dt + dv_dt_old)*abund[1]/2.0, vmin*uth)
		dTb_dt_old = dTb_dt
		dTdm_dt_old = dTdm_dt
		dv_dt_old = dv_dt
		if tag0==0:
			rhob_old = rhob_z(z)
		rhodm_old = rhodm_z(z)
		nb = rhob_old/mgas
		#print(z, nb, mgas)
		#total += abund[4]
		total += abund[2]
		count += 1
		if tag2==1:
			if Tb_old<TV:
				tag2 = 0
		#if max(Tb_old, Tdm_old) > TV and tag0==0 and tag2==0:
		#	Tb_old = TV
		#	tag0 = 1
		if t_cum>=t1 and tag1==0 and tag0==0 and hat>0:
			pV_pri = [nold[3]/refa[3], nold[11]/refa[11], nold[5]/refa[5], Tb_old, v_old]
			Tb_V = Tb_old
			Tb_old = TV #max(TV, Tb_old)
			Tdm_old = TV #max(TV, Tdm_old)
			pV = [Tb_old, Tdm_old, v_old, nb, nold*nb, rhob_old, rhodm_old, z]
			tcool = coolt(*pV, mode, *para)
			#v_old = max(VV, v_old)
			tag1 = 1
			#print('x_H2 = {}'.format(nold[3]/refa[3]))
		if (count%10==0)or(t_cum>=tmax):
			lt.append(t_cum)#[count] = t_cum
			lTb.append(Tb_old)#[count] = Told
			lTdm.append(Tdm_old)#[count] = nb
			lv.append(v_old)
			lrhob.append(rhob_old)
			lrhodm.append(rhodm_old)
			lz.append(z)
			for x in range(Ns):
				if refa[x]!=0:
					lX[x].append(nold[x]/refa[x])#[count] = newX[x]
				else:
					lX[x].append(0.0)
	d = {}
	d['t'] = np.array(lt)/YR/1e6
	d['z'] = np.array(lz)
	d['Tb'] = np.array(lTb)
	d['Tdm'] = np.array(lTdm)
	d['v'] = np.array(lv)
	d['rho'] = np.array(lrhob) + np.array(lrhodm)
	d['nb'] = np.array(lrhob)/mgas
	d['X'] = np.array(lX)
	d['rat'] = Tb_old/TV
	d['rat0'] = tpost/(t1 + tpost)
	d['s'] = int(tpost/tmax > Tb_old/TV)
	d['Tvir'] = TV
	d['TbV'] = Tb_V
	d['rat1'] = Tb_V/TV
	d['rat2'] = tcool/tffV
	d['m'] = M_T(Tb_V/d['rat0'], zvir, dmax)
	d['pV'] = pV
	d['pV_pri'] = pV_pri
	d['para'] = para
	end = time.time()
	#print(t_cum-t1)
	#print('Time taken: {} s'.format(end-start))
	print(count, total)
	return d

Mup = lambda z: 2.5e7*((1+z)/10)**-1.5
Mdown = lambda z: 1.54e5*((1+z)/31)**-2.074 #1e6*((1+z)/10)**-2

def Vcool(z, v0, vc = 3.714e5, alpha = 4.015):
	vbd = vbdm_z(z, v0)
	return ((alpha*vbd)**2 + vc**2)**0.5

def stored(d, Mh = 1e6, zvir = 20, v0 = 30, mode = 0, Mdm = 0.3, sigma = 8e-20, rep = 'data/'):
	if not os.path.exists(rep):
		os.makedirs(rep)
	out0 = [d['t'], d['z'], d['Tb'], d['Tdm'], d['v'], d['rho'], d['nb']]
	out1 = [[d['rat'], d['rat0'], d['rat1'], d['rat2'], d['s'], d['Tvir'], d['TbV'], d['m']]]
	base = 'M'+str(int(Mh/1e6 * 100)/100)+'_z'+str(zvir)+'_v'+str(int(v0*100)/100)
	if mode!=0:
		base = base + '_Mdm'+str(Mdm)+'_sigma'+str(sigma)
	totxt(rep+'dataD_'+base+'.txt',out0,0,0,0)
	totxt(rep+'dataX_'+base+'.txt',d['X'],0,0,0)
	totxt(rep+'dataP_'+base+'.txt',out1,0,0,0)
	return d

def readd(Mh = 1e6, zvir = 20, v0 = 30, mode = 0, Mdm = 0.3, sigma = 8e-20, dmax = 18*np.pi**2, rep = 'data/'):
	base = 'M'+str(int(Mh/1e6 * 100)/100)+'_z'+str(zvir)+'_v'+str(int(v0*100)/100)
	if mode!=0:
		base = base + '_Mdm'+str(Mdm)+'_sigma'+str(sigma)
	rd0 = np.array(retxt(rep+'dataD_'+base+'.txt',7,0,0))
	rd1 = np.array(retxt(rep+'dataP_'+base+'.txt',1,0,0)[0])
	d = {}
	d['X'] = np.array(retxt(rep+'dataX_'+base+'.txt',17,0,0))
	d['t'], d['z'], d['Tb'], d['Tdm'], d['v'], d['rho'], d['nb'] = \
		rd0[0], rd0[1], rd0[2], rd0[3], rd0[4], rd0[5], rd0[6]
	d['rat'], d['rat0'], d['rat1'], d['rat2'], d['s'], d['Tvir'], d['TbV'] = \
		rd1[0], rd1[1], rd1[2], rd1[3], rd1[4], rd1[5], rd1[6]
	d['m'] = M_T(d['TbV']/d['rat0'], zvir, dmax)
	return d

def Mth_z(z1, z2, nzb = 10, m1 = 1e2, m2 = 1e10, nmb = 100, mode = 0, z0 = z0_default, v0 = 30, Mdm = 0.3, sigma = 8e-20, rat = 1.0, dmax = 18*np.pi**2, Om = 0.315, h = 0.6774, fac = 1e-3, vmin = 0.0, alpha = 3., sk = False, init = init):
	m0 = (m1*m2)**0.5
	lz = np.linspace(z1, z2, nzb)
	#lz = np.logspace(np.log10(z1), np.log10(z2), nzb)
	out = []
	lxh2 = []
	lxhd = []
	lxe = []
	lTb = []
	lvr = []
	for z in lz:
		mmax = Mup(z)*10
		mmin = M_T(200, z, delta0, Om)
		lm = np.logspace(np.log10(mmin), np.log10(mmax), nmb)
		d = evolve(m0, z, z0, v0, mode, Mdm = Mdm, sigma = sigma, dmax = dmax, Om = Om, h = h, fac = fac, init = init)
		tffV = tff(z, dmax)
		#tffV = 1/H(1/(1+z), Om, h)
		#lT = [Tvir(m, z, delta0) for m in lm]
		red = (dmax/delta0)**0.5
		lT = [Tvir(m/red, z, dmax) for m in lm]
		if mode!=0 and sk:
			pV = d['pV'][3:]
			lvv = [Vcir(m, z, delta0) for m in lm]
			lv = np.zeros(nmb)
			for i in range(nmb):
				uth = (lT[i]*BOL/PROTON+lT[i]*BOL/(Mdm*GeV_to_mass))**0.5
				dvdt = bdmscool(lT[i], lT[i], lvv[i], *pV[-3:-1], Mdm, sigma, d['para'][2], d['para'][6])[2]
				vf = max(lvv[i] + dvdt * tffV, uth*vmin)
				lv[i] = 0.5 * ((vf*lvv[i])**0.5 + d['pV'][2])
			lt0 = [coolt(T, T, v, *pV, mode, *d['para'])/tffV for T, v in zip(lT, lv)]
		else:
			pV = d['pV'][2:]
			lt0 = [coolt(T, T, *pV, mode, *d['para'])/tffV for T in lT]
		lt00 = np.array(lt0)
		ltt0 = lt00[lt00>0]
		if ltt0 is []:
			print(Mdm, sigma, 'Heating!')
			return []
		else:
			imax = lt0.index(np.max(lt00))
			if imax<nmb-1:
				lt00 = np.array(lt0[imax:])
				ltt0 = lt00[lt00>0]
			imin = lt0.index(np.min(lt00))
			if imin==0:
				print(Mdm, sigma, 'lower bound')
				mth = mmin
			else:
				lt0 = np.array(lt0[imax:imin+1])
				lm = lm[imax:imin+1]
				lt = lt0[lt0>0]
				lm = lm[lt0>0]
				if np.min(np.log10(lt))>=np.log10(rat):
					mth = np.max(lm)
					print(Mdm, sigma, 'Upper bound')
				elif np.max(np.log10(lt))<=np.log10(rat):
					mth = np.min(lm)
				else:
					rat_m = interp1d(np.log10(lt), np.log10(lm))
					mth = 10**rat_m(np.log10(rat))
		#if mode!=0:
		mth0 = mth
		mth = mth * (1+alpha*d['pV'][2]**2/Vcir(mth, z, delta0)**2/(dmax/delta0)**(2/3))**(3/2)
		print(Mdm, sigma, mth/1e6, mth0/1e6, z)#mth_stm(mth0, z, v0, alpha)/1e6)
		out.append(mth)
		lxh2.append(d['pV_pri'][0])
		lxhd.append(d['pV_pri'][1])
		lxe.append(d['pV_pri'][2])
		lTb.append(d['pV_pri'][3])
		lvr.append(d['pV_pri'][4])
	return [np.array(out), lz, lxh2, lxhd, lxe, lTb, lvr]

def mth_stm(mth, z, v0, alpha = 3., dmax = delta0):
	return mth * (1 + alpha*vbdm_z(z, v0)**2/Vcir(mth, z, 1)**2)

def parasp(v0 = 30., m1 = -4, m2 = 2, s1 = -1, s2 = 4, z = 20, dmax = 200, nbin = 10, fac = 1e-3, rat = 1.0, ncore=4, nmb = 100, alpha = 3., sk = False):
	lm = np.logspace(m1, m2, nbin)
	ls = np.logspace(s1, s2, nbin)
	X, Y = np.meshgrid(lm, ls, indexing = 'ij')
	#mmax = Mup(z)*10
	lMh = np.zeros(X.shape)
	lXH2 = np.zeros(X.shape)
	lXHD = np.zeros(X.shape)
	lXe = np.zeros(X.shape)
	lTb = np.zeros(X.shape)
	lvr = np.zeros(X.shape)
	np_core = int(nbin/ncore)
	lpr = [[i*np_core, (i+1)*np_core] for i in range(ncore-1)] + [[(ncore-1)*np_core, nbin]]
	print(lpr)
	manager = mp.Manager()
	def sess(pr0, pr1, j):
		out = []
		for i in range(pr0, pr1):
			init = initial(Mdm = lm[i], sigma = ls[j]*1e-20, v0 = v0, mode = 1)
			d = Mth_z(z,z,1, Mdm = lm[i], sigma = ls[j]*1e-20, v0 = v0, dmax = dmax, rat = rat, fac = fac, nmb = nmb, mode = 1, alpha = alpha, sk = sk, init = init)
			out.append([x[0] for x in d])
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
		lMh[:,i] = np.hstack([x[1][0] for x in out])
		lXH2[:,i] = np.hstack([x[1][2] for x in out])
		lXHD[:,i] = np.hstack([x[1][3] for x in out])
		lXe[:,i] = np.hstack([x[1][4] for x in out])
		lTb[:,i] = np.hstack([x[1][5] for x in out])
		lvr[:,i] = np.hstack([x[1][6] for x in out])
		#for j in range(nbin):
		#	sol = T21_pred(v0, lm[i], ls[j]*1e-20, xa0)
		#	lT[i,j] = -sol[0]
		#	lTb[i,j] = sol[1]
	return X, Y*1e-20, lMh, lXH2, lXHD, lXe, lTb, lvr

def vdis(v, sigma = 30.):
	return v**2 * np.exp(-v**2*1.5/sigma**2)

def Nhalo(z, lm0, lv0, mode = 0, Mdm = 0.3, h = 0.6774, sigma = 30., vmax = 5.):
	sel = lv0 < vmax*sigma
	lm = lm0[sel]
	lv = lv0[sel]
	lw = vdis(lv, sigma)
	Mmin = min(np.log10(np.min(lm*h))-1, np.log10(Mup(z))-1)
	if mode == 0:
		hmf_ = hmf.MassFunction()
		hmf_.update(n=0.966, sigma_8=0.829,cosmo_params={'Om0':0.315,'H0':67.74},Mmin=Mmin,Mmax=np.log10(Mup(z))+2,z=z)
	else:
		hmf_ = hmf.wdm.MassFunctionWDM(wdm_mass=Mdm*1e6)
		hmf_.update(n=0.966, sigma_8=0.829,cosmo_params={'Om0':0.315,'H0':67.74},Mmin=Mmin,Mmax=np.log10(Mup(z))+2,z=z)
	intm = np.log10(hmf_.m/h)
	intn = np.log10(hmf_.ngtm)
	nm = interp1d(intm, intn)
	lnpopIII = (10**nm(np.log10(lm))- 10**nm(np.log10(Mup(z))))
	lnpopIII = lnpopIII * (lnpopIII>0)
	out = lnpopIII * lw
	return np.trapz(out, lv)/np.trapz(lw, lv)
		

if __name__=="__main__":
	tag = 1
	v0 = 24 #60
	nbin = 32 #64
	ncore = 8
	dmax = delta0 * 100
	rat = 1.
	fac = 1e-3
	#alpha0 = 3.
	alpha = 0.7
	sk = False
	#sk = True
	if zi_tag==0:
		rep = '100-sigma_test0/'
	else:
		rep = '100-sigma_test/'
	zvir = 20
	if not os.path.exists(rep):
		os.makedirs(rep)
	init0 = initial(v0 = v0, mode = 0)
	init1 = initial(v0 = v0, mode = 1)

	"""
	if tag==0:
		d = Mth_z(zvir, 30, 2, v0 = v0, dmax = dmax, Mdm = 0.3, sigma = 8e-20, mode = 0, rat = rat, alpha = alpha, init = init0)
		d_ = Mth_z(zvir, 30, 2, v0 = v0, dmax = dmax, Mdm = 0.3, sigma = 8e-20, mode = 1, rat = rat, alpha = alpha, init = init1)
		print('Mth at z = 20: {} 10^6 Msun (CDM)'.format(d[0][0]/1e6))
		print('Mth at z = 20: {} 10^6 Msun (BDMS)'.format(d_[0][0]/1e6))
		totxt(rep+'ref_'+str(v0)+'.txt', np.array(d).T, 0,0,0)
		totxt(rep+'default_'+str(v0)+'.txt', np.array(d_).T,0,0,0)
		X, Y, Mh, XH2, XHD, Xe, Tb, Vr = parasp(v0, m1 = -4, m2 = 2, s1 = -1, s2 = 4, nbin = nbin, ncore = ncore, dmax = dmax, fac = fac, rat = rat, alpha = alpha, sk = sk, z = zvir)
		totxt(rep+'X_'+str(v0)+'.txt',X,0,0,0)
		totxt(rep+'Y_'+str(v0)+'.txt',Y,0,0,0)
		totxt(rep+'Mh_'+str(v0)+'.txt',Mh,0,0,0)
		totxt(rep+'XH2_'+str(v0)+'.txt',XH2,0,0,0)
		totxt(rep+'XHD_'+str(v0)+'.txt',XHD,0,0,0)
		totxt(rep+'Xe_'+str(v0)+'.txt',Xe,0,0,0)
		totxt(rep+'Tb_'+str(v0)+'.txt',Tb,0,0,0)
		totxt(rep+'Vr_'+str(v0)+'.txt',Vr,0,0,0)
	else:
		X = np.array(retxt(rep+'X_'+str(v0)+'.txt',nbin,0,0))
		Y = np.array(retxt(rep+'Y_'+str(v0)+'.txt',nbin,0,0))
		Mh = np.array(retxt(rep+'Mh_'+str(v0)+'.txt',nbin,0,0))
		XH2 = np.array(retxt(rep+'XH2_'+str(v0)+'.txt',nbin,0,0))
		XHD = np.array(retxt(rep+'XHD_'+str(v0)+'.txt',nbin,0,0))
		Xe = np.array(retxt(rep+'Xe_'+str(v0)+'.txt',nbin,0,0))
		Tb = np.array(retxt(rep+'Tb_'+str(v0)+'.txt',nbin,0,0))
		Vr = np.array(retxt(rep+'Vr_'+str(v0)+'.txt',nbin,0,0))

	Vr0 = 1e-5
	Vr = Vr*(Vr>Vr0) + Vr0*(Vr<=Vr0)

	refMh, z, xH2r, xHDr, xer, Tbr, vrr = retxt(rep+'ref_'+str(v0)+'.txt',1,0,0)[0]
	#refMh = mth_stm(refMh, 17, v0, alpha = alpha0)
	print('Reference mass thresold: {} 10^6 Msun'.format(refMh/1e6))
	print('Reference H2 abundance: {} * 10^-4'.format(xH2r*1e4))
	print('Reference HD abundance: {} * 10^-3'.format(xHDr*1e3))
	print('Reference e abundance: {} *10^-5'.format(xer*1e5))
	print('Reference Tb: {} K'.format(Tbr))
	print('Reference V_bdm: {} km s^-1'.format(vrr/1e5))
	Mbd = Mup(zvir)*2
	#Mh = Mh*(Mh<Mbd) + Mbd*(Mh>=Mbd)
	plt.figure()
	ctf = plt.contourf(X, Y, np.log10(Mh), np.linspace(5.5, 8, 2*nbin), cmap=plt.cm.Blues)
	for c in ctf.collections:
		c.set_edgecolor('face')
	cb = plt.colorbar()
	cb.set_label(r'$\log(M_{\mathrm{th}}\ [M_{\odot}])$',size=12)
	plt.contour(X, Y, np.log10(Mh), [np.log10(refMh)+2e2], colors='k')
	print(np.min(Mh[Mh!=np.nan]))
	plt.contour(X, Y, np.log10(Mh), [np.log10(Mup(zvir))], colors='k', linestyles='--')
	plt.contour(X, Y, np.log10(Mh), [0.99+np.log10(Mup(zvir))], colors='k', linestyles='-.')
	plt.plot([0.3], [8e-20], '*', color='purple')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$m_{\chi}c^{2}\ [\mathrm{GeV}]$')
	plt.ylabel(r'$\sigma_{1}\ [\mathrm{cm^{2}}]$')
	plt.tight_layout()
	plt.savefig(rep+'MthMap_v0_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	ctf = plt.contourf(X, Y, np.log10(XH2), 2*nbin, cmap=plt.cm.Blues)
	for c in ctf.collections:
		c.set_edgecolor('face')
	cb = plt.colorbar()
	cb.set_label(r'$\log(x_{\mathrm{H_{2}}})$',size=12)
	plt.contour(X, Y, np.log10(XH2), [np.log10(xH2r)], colors='k')
	plt.plot([0.3], [8e-20], '*', color='purple')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$m_{\chi}c^{2}\ [\mathrm{GeV}]$')
	plt.ylabel(r'$\sigma_{1}\ [\mathrm{cm^{2}}]$')
	plt.tight_layout()
	plt.savefig(rep+'XH2Map_v0_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	ctf = plt.contourf(X, Y, np.log10(XHD), 2*nbin, cmap=plt.cm.Blues)
	for c in ctf.collections:
		c.set_edgecolor('face')
	cb = plt.colorbar()
	cb.set_label(r'$\log(x_{\mathrm{HD}})$',size=12)
	plt.contour(X, Y, np.log10(XHD), [np.log10(xHDr)], colors='k')
	plt.plot([0.3], [8e-20], '*', color='purple')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$m_{\chi}c^{2}\ [\mathrm{GeV}]$')
	plt.ylabel(r'$\sigma_{1}\ [\mathrm{cm^{2}}]$')
	plt.tight_layout()
	plt.savefig(rep+'XHDMap_v0_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	ctf = plt.contourf(X, Y, np.log10(Xe), 2*nbin, cmap=plt.cm.Blues)
	for c in ctf.collections:
		c.set_edgecolor('face')
	cb = plt.colorbar()
	cb.set_label(r'$\log(x_{\mathrm{e}})$',size=12)
	plt.contour(X, Y, np.log10(Xe), [np.log10(xer)], colors='k')
	plt.plot([0.3], [8e-20], '*', color='purple')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$m_{\chi}c^{2}\ [\mathrm{GeV}]$')
	plt.ylabel(r'$\sigma_{1}\ [\mathrm{cm^{2}}]$')
	plt.tight_layout()
	plt.savefig(rep+'XeMap_v0_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	ctf = plt.contourf(X, Y, np.log10(Tb), 2*nbin, cmap=plt.cm.Blues)
	for c in ctf.collections:
		c.set_edgecolor('face')
	cb = plt.colorbar()
	cb.set_label(r'$\log(T_{b}\ [\mathrm{K}])$',size=12)
	plt.contour(X, Y, np.log10(Tb), [np.log10(Tbr)], colors='k')
	plt.plot([0.3], [8e-20], '*', color='purple')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$m_{\chi}c^{2}\ [\mathrm{GeV}]$')
	plt.ylabel(r'$\sigma_{1}\ [\mathrm{cm^{2}}]$')
	plt.tight_layout()
	plt.savefig(rep+'TbMap_v0_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	ctf = plt.contourf(X, Y, Vr/1e5, 2*nbin, cmap=plt.cm.Blues)
	for c in ctf.collections:
		c.set_edgecolor('face')
	cb = plt.colorbar()
	cb.set_label(r'$v_{b\chi,V}\ [\mathrm{km\ s^{-1}}]$',size=12)
	plt.contour(X, Y, Vr/1e5, [vrr/1e5], colors='k')
	plt.plot([0.3], [8e-20], '*', color='purple')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$m_{\chi}c^{2}\ [\mathrm{GeV}]$')
	plt.ylabel(r'$\sigma_{1}\ [\mathrm{cm^{2}}]$')
	plt.tight_layout()
	plt.savefig(rep+'VrMap_v0_'+str(v0)+'.pdf')
	plt.close()
	"""

	"""

	#lm, lz, lxh2, lxhd = Mth_z(10, 100, 46, mode = 0, rat = rat, dmax = dmax, fac = fac)
	#totxt(rep+'Mthz_CDM.txt',[lz, lm, lxh2, lxhd],0,0,0)

	#tag = 0
	#v0 = 0.1
	#rat = 10.
	if tag==0:
		d_ = Mth_z(10, 60, 51, mode = 1, v0 = v0, rat = rat, dmax = dmax, fac = fac, alpha = alpha, sk = sk, init = init1)
		d = Mth_z(10, 60, 51, mode = 0, v0 = v0, rat = rat, dmax = dmax, fac = fac, alpha = alpha, sk = sk, init = init0)
		totxt(rep+'Mthz_CDM_'+str(v0)+'.txt',d,0,0,0)
		totxt(rep+'Mthz_BDMS_'+str(v0)+'.txt',d_,0,0,0)
	lm_, lz_, lxh2_, lxhd_, lxe_, lTb_, lvr_ = np.array(retxt(rep+'Mthz_BDMS_'+str(v0)+'.txt',7,0,0))
	lm, lz, lxh2, lxhd, lxe, lTb, lvr = np.array(retxt(rep+'Mthz_CDM_'+str(v0)+'.txt',7,0,0))
	#lm = mth_stm(lm, lz, v0, alpha = alpha0)
	plt.figure()
	plt.plot(lz, lm, label='CDM')
	plt.plot(lz_, lm_, '--', label='BDMS')
	#plt.plot(lz, Mdown(lz), 'k-.', label='Trenti & Stiavelli (2009)')
	plt.fill_between([15,20],[1e4,1e4],[3e8,3e8], facecolor='gray', label='EDGES')
	plt.legend()
	plt.xlabel(r'$z_{\mathrm{vir}}$')
	plt.ylabel(r'$M_{\mathrm{th}}\ [M_{\odot}]$')
	#plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()
	plt.xlim(15, 100)
	plt.ylim(1e4, 1e8)
	plt.savefig(rep+'Mth_z_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	plt.plot(lz, lxh2, label='CDM')
	plt.plot(lz_, lxh2_, '--', label='BDMS')
	plt.legend()
	plt.xlabel(r'$z_{\mathrm{vir}}$')
	plt.ylabel(r'$x_{\mathrm{H_{2}}}$')
	#plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep+'XH2_z_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	plt.plot(lz, lxhd, label='CDM')
	plt.plot(lz_, lxhd_, '--', label='BDMS')
	plt.legend()
	plt.xlabel(r'$z_{\mathrm{vir}}$')
	plt.ylabel(r'$x_{\mathrm{HD}}$')
	#plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep+'XHD_z_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	plt.plot(lz, lxe, label='CDM')
	plt.plot(lz_, lxe_, '--', label='BDMS')
	plt.legend()
	plt.xlabel(r'$z_{\mathrm{vir}}$')
	plt.ylabel(r'$x_{\mathrm{e}}$')
	#plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep+'Xe_z_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	plt.plot(lz, lTb, label='CDM')
	plt.plot(lz_, lTb_, '--', label='BDMS')
	plt.legend()
	plt.xlabel(r'$z_{\mathrm{vir}}$')
	plt.ylabel(r'$T_{b}\ [\mathrm{K}]$')
	#plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep+'Tb_z_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	plt.plot(lz, lvr/1e5, label='CDM')
	plt.plot(lz_, lvr_/1e5, '--', label='BDMS')
	plt.legend()
	plt.xlabel(r'$z_{\mathrm{vir}}$')
	plt.ylabel(r'$v_{b\chi,V}\ [\mathrm{km\ s^{-1}}]$')
	#plt.xscale('log')
	#plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep+'Vr_z_'+str(v0)+'.pdf')
	plt.close()
	#plt.show()
	"""
	
	"""
	rep = 'Nhrat/'
	if not os.path.exists(rep):
		os.makedirs(rep)
	lls = ['-', '--', '-.', ':']*2
	tag = 1
	#rat = 10.
	nz = 51
	z1, z2 = 10, 60
	Mdm =  0.3
	sigma = 8e-20
	#Mdm = 1e-3 
	#sigma = 1e-18 
	#Mdm = 10.
	#sigma = 1e-20

	#d0 = Mth_z(z1, z2, nz, mode = 0, rat = rat, dmax = dmax, fac = fac)
	#totxt(rep+'ref_z.txt', d0, 0, 0)

	#lv = np.logspace(0, 3, 31)
	lv = np.linspace(0, 150, 16)
	#lv = np.logspace(0, np.log10(150), 20)
	vd, vu = np.min(lv), np.max(lv)
	if tag==0:
		d0 = Mth_z(z1, z2, nz, mode = 0, rat = rat, dmax = dmax, fac = fac, v0 = 0)
		totxt(rep+'ref_z.txt', d0, 0, 0)
		out = []
		out_ = []
		for v in lv:
			initv0 = initial(v0 = v, mode = 0)
			initv1 = initial(v0 = v, mode = 1, Mdm = Mdm, sigma = sigma)
			d = Mth_z(z1, z2, nz, mode = 1, v0 = v, rat = rat, dmax = dmax, fac = fac, alpha = alpha, sk = sk, init = initv1, Mdm = Mdm, sigma = sigma)
			d_ = Mth_z(z1, z2, nz, mode = 0, rat = rat, dmax = dmax, fac = fac, v0 = v, alpha = alpha, sk = sk, init = initv0, Mdm = Mdm, sigma = sigma)
			out.append(d)
			out_.append(d_)
		lz = d[1]

		totxt(rep+'zbase.txt',[lz],0,0,0)
		totxt(rep+'vbase.txt',[lv],0,0,0)

		lm = [[x[0][i] for x in out] for i in range(len(d[0]))]
		lxh2 = [[x[2][i] for x in out] for i in range(len(d[0]))]
		lxhd = [[x[3][i] for x in out] for i in range(len(d[0]))]
		lxe = [[x[4][i] for x in out] for i in range(len(d[0]))]
		lTb = [[x[5][i] for x in out] for i in range(len(d[0]))]
		lvr = [[x[6][i] for x in out] for i in range(len(d[0]))]

		totxt(rep+'Mth_v.txt',lm,0,0,0)
		totxt(rep+'xh2_v.txt',lxh2,0,0,0)
		totxt(rep+'xhd_v.txt',lxhd,0,0,0)
		totxt(rep+'xe_v.txt',lxe,0,0,0)
		totxt(rep+'Tb_v.txt',lTb,0,0,0)
		totxt(rep+'vr_v.txt',lvr,0,0,0)

		lm_ = [[x[0][i] for x in out_] for i in range(len(d[0]))]
		lxh2_ = [[x[2][i] for x in out_] for i in range(len(d[0]))]
		lxhd_ = [[x[3][i] for x in out_] for i in range(len(d[0]))]
		lxe_ = [[x[4][i] for x in out_] for i in range(len(d[0]))]
		lTb_ = [[x[5][i] for x in out_] for i in range(len(d[0]))]
		lvr_ = [[x[6][i] for x in out_] for i in range(len(d[0]))]

		totxt(rep+'Mth_v0.txt',lm_,0,0,0)
		totxt(rep+'xh2_v0.txt',lxh2_,0,0,0)
		totxt(rep+'xhd_v0.txt',lxhd_,0,0,0)
		totxt(rep+'xe_v0.txt',lxe_,0,0,0)
		totxt(rep+'Tb_v0.txt',lTb_,0,0,0)
		totxt(rep+'vr_v0.txt',lvr_,0,0,0)

	#d0 = retxt(rep+'ref_z.txt',7,0,0)
	#mr, zr, xh2r, xhdr, xer, Tbr, vrr = d0
	lv = np.array(retxt(rep+'vbase.txt',1,0,0)[0])
	lz = retxt(rep+'zbase.txt',1,0,0)[0]
	nz = len(lz)
	lm = np.array(retxt(rep+'Mth_v.txt',nz,0,0))
	lxh2 = np.array(retxt(rep+'xh2_v.txt',nz,0,0))
	lxhd = np.array(retxt(rep+'xhd_v.txt',nz,0,0))
	lxe = np.array(retxt(rep+'xe_v.txt',nz,0,0))
	lTb = np.array(retxt(rep+'Tb_v.txt',nz,0,0))
	lvr = np.array(retxt(rep+'vr_v.txt',nz,0,0))
	lm_ = np.array(retxt(rep+'Mth_v0.txt',nz,0,0))
	lxh2_ = np.array(retxt(rep+'xh2_v0.txt',nz,0,0))
	lxhd_ = np.array(retxt(rep+'xhd_v0.txt',nz,0,0))
	lxe_ = np.array(retxt(rep+'xe_v0.txt',nz,0,0))
	lTb_ = np.array(retxt(rep+'Tb_v0.txt',nz,0,0))
	lvr_ = np.array(retxt(rep+'vr_v0.txt',nz,0,0))
	plt.figure()
	a = [plt.plot(lv, lm[i], label=r'$z_{vir}='+str(int(lz[i]*100)/100)+'$, BDMS', ls = lls[i]) for i in range(nz)]
	a = [plt.plot(lv, lm_[i], label=r'$z_{vir}='+str(int(lz[i]*100)/100)+'$, CDM',color='k',ls=lls[i],lw=0.5) for i in range(nz)]
	plt.legend()
	plt.xlabel(r'$v_{b\chi,0}\ [\mathrm{km\ s^{-1}}]$')
	plt.ylabel(r'$M_{\mathrm{th}}\ [M_{\odot}]$')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(vd, vu)
	plt.tight_layout()
	plt.savefig(rep+'Mth_v.pdf')
	plt.close()

	plt.figure()
	a = [plt.plot(lv, lxh2[i], label=r'$z_{vir}='+str(int(lz[i]*100)/100)+'$, BDMS', ls = lls[i]) for i in range(nz)]
	a = [plt.plot(lv, lxh2_[i], label=r'$z_{vir}='+str(int(lz[i]*100)/100)+'$, CDM',color='k',ls=lls[i],lw=0.5) for i in range(nz)]
	plt.legend()
	plt.xlabel(r'$v_{b\chi,0}\ [\mathrm{km\ s^{-1}}]$')
	plt.ylabel(r'$x_{\mathrm{H_{2}}}$')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(vd, vu)
	plt.tight_layout()
	plt.savefig(rep+'XH2_v.pdf')
	plt.close()

	plt.figure()
	a = [plt.plot(lv, lxhd[i], label=r'$z_{vir}='+str(int(lz[i]*100)/100)+'$, BDMS', ls = lls[i]) for i in range(nz)]
	a = [plt.plot(lv, lxhd_[i], label=r'$z_{vir}='+str(int(lz[i]*100)/100)+'$, CDM',color='k',ls=lls[i],lw=0.5) for i in range(nz)]
	plt.legend()
	plt.xlabel(r'$v_{b\chi,0}\ [\mathrm{km\ s^{-1}}]$')
	plt.ylabel(r'$x_{\mathrm{HD}}$')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(vd, vu)
	plt.tight_layout()
	plt.savefig(rep+'XHD_v.pdf')
	plt.close()

	plt.figure()
	a = [plt.plot(lv, lxe[i], label=r'$z_{vir}='+str(int(lz[i]*100)/100)+'$, BDMS', ls = lls[i]) for i in range(nz)]
	a = [plt.plot(lv, lxe_[i], label=r'$z_{vir}='+str(int(lz[i]*100)/100)+'$, CDM',color='k',ls=lls[i],lw=0.5) for i in range(nz)]
	plt.legend()
	plt.xlabel(r'$v_{b\chi,0}\ [\mathrm{km\ s^{-1}}]$')
	plt.ylabel(r'$x_{\mathrm{e}}$')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(vd, vu)
	plt.tight_layout()
	plt.savefig(rep+'Xe_v.pdf')
	plt.close()

	plt.figure()
	a = [plt.plot(lv, lTb[i], label=r'$z_{vir}='+str(int(lz[i]*100)/100)+'$, BDMS', ls = lls[i]) for i in range(nz)]
	a = [plt.plot(lv, lTb_[i], label=r'$z_{vir}='+str(int(lz[i]*100)/100)+'$, CDM',color='k',ls=lls[i],lw=0.5) for i in range(nz)]
	plt.legend()
	plt.xlabel(r'$v_{b\chi,0}\ [\mathrm{km\ s^{-1}}]$')
	plt.ylabel(r'$T_{b}\ [\mathrm{K}]$')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(vd, vu)
	plt.tight_layout()
	plt.savefig(rep+'Tb_v.pdf')
	plt.close()

	plt.figure()
	a = [plt.plot(lv, lvr[i]/dmax**(1/3)/1e5, label=r'$z_{vir}='+str(int(lz[i]*100)/100)+'$, BDMS', ls = lls[i]) for i in range(nz)]
	#a = [plt.plot(lv, [vbdm_z(zr[i], v)/1e5  for v in lv], label=r'$z_{vir}='+str(int(lz[i]*100)/100)+'$, CDM',color='k',ls=lls[i],lw=0.5) for i in range(nz)]
	a = [plt.plot(lv, lvr_[i]/dmax**(1/3)/1e5, label=r'$z_{vir}='+str(int(lz[i]*100)/100)+'$, CDM',color='k',ls=lls[i],lw=0.5) for i in range(nz)]
	plt.legend()
	plt.xlabel(r'$v_{b\chi,0}\ [\mathrm{km\ s^{-1}}]$')
	plt.ylabel(r'$v_{b\chi,V}\ [\mathrm{km\ s^{-1}}]$')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(vd, vu)
	plt.ylim(1e-2, 1e2)
	plt.tight_layout()
	plt.savefig(rep+'Vr_v.pdf')
	plt.close()
	"""

	#"""
	tag = 0
	m = 1e6
	zvir = 20
	v0 = 24
	Mdm =  0.3 #0.001
	sigma = 8e-20 #1e-17
	if zi_tag==0:
		rep0 = 'example_test0/'
		drep = 'data0/'
	else:
		rep0 = 'example_test/'
		drep = 'data_test/'
	
	if not os.path.exists(rep0):
		os.makedirs(rep0)
	#dmax = delta0 #18 * np.pi**2 * 1
	init0 = initial(v0 = v0, mode = 0, Mdm = Mdm, sigma = sigma)
	init1 = initial(v0 = v0, mode = 1, Mdm = Mdm, sigma = sigma)
	if tag==0:
		#print('!!!')
		d0 = stored(evolve(m, zvir, mode = 0, dmax = dmax, v0 = v0, init = init0, Mdm = Mdm, sigma = sigma), m, zvir, mode = 0, v0 = v0, rep = drep, Mdm = Mdm, sigma = sigma)
		d1 = stored(evolve(m, zvir, mode = 1, dmax = dmax, v0 = v0, init = init1, Mdm = Mdm, sigma = sigma), m, zvir, mode = 1, v0 = v0, rep = drep, Mdm = Mdm, sigma = sigma)
	else:
		d0 = readd(m, zvir, v0, mode = 0, rep = drep, Mdm = Mdm, sigma = sigma)
		d1 = readd(m, zvir, v0, mode = 1, rep = drep, Mdm = Mdm, sigma = sigma)
	
	mgas = mmw()*PROTON
	nIGM = [rhom(1/(1+z))*0.048/(0.315*mgas) for z in d0['z']]
	plt.figure()
	plt.plot(d0['t'], d0['nb'], label='Top-hat model')
	#plt.plot(d1['t'], d1['nb'], '--', label='BDMS')
	plt.plot(d0['t'], nIGM, '-.', label='IGM')
	plt.xlabel(r'$t\ [\mathrm{Myr}]$')
	plt.ylabel(r'$n\ [\mathrm{cm^{-3}}]$')
	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep0+'Example_n_t_m6_'+str(m/1e6)+'_z_'+str(zvir)+'_v0_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	plt.plot(d0['t'], d0['Tb'], label=r'$T_{b}$, CDM')
	plt.plot(d1['t'], d1['Tb'], '--', label=r'$T_{b}$, BDMS')
	#plt.plot(d0['t'], d0['Tdm'], '-.', label=r'$T_{\chi}$, CDM')
	plt.plot(d1['t'], d1['Tdm'], '-.', label=r'$T_{\chi}$, BDMS')
	plt.plot(d1['t'], (d1['z']+1)*2.726, ':', label=r'$T_{cmb}$')
	plt.plot(d1['t'], T_b(d1['z']), 'k-', label=r'$T_{b}(\mathrm{IGM})$, CDM', lw=0.5)
	plt.xlabel(r'$t\ [\mathrm{Myr}]$')
	plt.ylabel(r'$T\ [\mathrm{K}]$')
	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	#plt.xlim(np.min(d0['t']), np.max(d0['t']))
	#plt.ylim(1, 3e3)
	plt.tight_layout()
	plt.savefig(rep0+'Example_T_t_m6_'+str(m/1e6)+'_z_'+str(zvir)+'_v0_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	plt.plot(d0['t'], d0['X'][3], label='CDM')
	plt.plot(d1['t'], d1['X'][3], '--', label='BDMS')
	plt.xlabel(r'$t\ [\mathrm{Myr}]$')
	plt.ylabel(r'$x_{\mathrm{H_{2}}}$')
	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(1e-8, np.max(d0['X'][3])*1.5)
	plt.tight_layout()
	plt.savefig(rep0+'Example_xH2_t_m6_'+str(m/1e6)+'_z_'+str(zvir)+'_v0_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	plt.plot(d0['t'], d0['X'][5], label='CDM')
	plt.plot(d1['t'], d1['X'][5], '--', label='BDMS')
	plt.xlabel(r'$t\ [\mathrm{Myr}]$')
	plt.ylabel(r'$x_{\mathrm{e}}$')
	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	#plt.ylim(1e-5, 1e-3)
	plt.tight_layout()
	plt.savefig(rep0+'Example_xe_t_m6_'+str(m/1e6)+'_z_'+str(zvir)+'_v0_'+str(v0)+'.pdf')
	plt.close()

	plt.figure()
	#plt.plot(d0['t'], d0['X'][0], label='0')
	plt.plot(d1['z'], d1['X'][3], label='3')
	plt.plot(d1['z'], d1['X'][5], label='5')
	#plt.plot(d0['t'], d0['X'][9], label='9')
	plt.plot(d1['z'], d1['X'][10], label='10')
	plt.plot(d1['z'], d1['X'][11], label='11')
	plt.plot(d1['z'], d1['X'][2], label='2')
	plt.plot(d1['z'], d1['X'][4], label='4')
	#plt.plot(d1['t'], d1['X'][5], '--', label='BDMS')
	#plt.xlabel(r'$t\ [\mathrm{Myr}]$')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$x$')
	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	#plt.ylim(1e-5, 1e-3)
	plt.tight_layout()
	plt.savefig(rep0+'Example_X_z_m6_'+str(m/1e6)+'_z_'+str(zvir)+'_v0_'+str(v0)+'.pdf')
	plt.close()
	#plt.show()

	vIGM = [vbdm_z(z, v0)/1e5 for z in d0['z']]
	plt.figure()
	plt.plot(d0['t'], d0['v']/1e5, label='CDM')
	plt.plot(d1['t'], d1['v']/1e5, '--', label='BDMS')
	plt.plot(d0['t'], vIGM, '-.', label='IGM')
	plt.xlabel(r'$t\ [\mathrm{Myr}]$')
	plt.ylabel(r'$v_{b\chi}\ [\mathrm{km\ s^{-1}}]$')
	plt.legend()
	plt.ylim(np.min(vIGM)/10, np.max(vIGM)*10)
	plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig(rep0+'Example_vbDM_t_m6_'+str(m/1e6)+'_z_'+str(zvir)+'_v0_'+str(v0)+'.pdf')
	plt.close()
	#"""

	"""
	tag = 0
	zvir = 50
	lm = 10**np.linspace(4,8,10)
	if tag==0:
		ld0 = [stored(evolve(m, zvir), m, zvir) for m in lm]
		#a = [stored(d, m, zvir) for d, m in zip(ld0, lm)]
	else:
		ld0 = [readd(m, zvir) for m in lm]
	lrat = np.array([d['rat'] for d in ld0])
	lrat0 = np.array([d['rat0'] for d in ld0])
	lrat1 = np.array([d['rat1'] for d in ld0])
	lrat2 = np.array([d['rat2'] for d in ld0])
	
	plt.figure()
	plt.plot(lm, lrat/lrat0, label=r'$(T_{f}/T_{V})/(\Delta t/t_{f})$')
	plt.plot(lm, lrat1/lrat0, '--', label=r'$(T(z_{V})/T_{V})/(\Delta t/t_{f})$')
	plt.plot(lm, lrat2, '-.', label=r'$t_{\mathrm{cool}}/t_{\mathrm{ff}}$')
	plt.plot(lm, np.ones(len(lm)), 'k:')
	plt.legend()
	plt.xlim(np.min(lm), np.max(lm))
	plt.ylim(1e-2, 1e2)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$M\ [M_{\odot}]$')
	plt.ylabel('Ratio')
	plt.tight_layout()
	plt.savefig('Ratio_z'+str(zvir)+'.pdf')
	#plt.show()

	mth = [d['m'] for d in ld0]
	print(mth)
	"""


