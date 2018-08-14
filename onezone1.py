from coolingf import *
from reactionT import *
#from reaction import *
from scipy.integrate import odeint
import time
import sys

OmegaM = 0.3089
hubble = 0.6774
SPEEDOFLIGHT = 2.99792458e+10
PROTON = 1.6726e-24
pc = 3.08568e+18
#YR = 3600*24*365
G = 6.674e-8
kB = 1.3806e-16
H00 = 100*10**5/pc/10**6
H0 = H00*hubble 

REDSHIFT = 5.0
NMAX = 1e12

lT00 = np.hstack([10**np.linspace(1,3.5,5), 10**np.linspace(3.55,3.75,5), 10**np.linspace(3.8,4.2,20), 10**np.linspace(4.25,4.95,5), 10**np.linspace(5,6,5)])

def Xe(T, Ifac = 20.0, T0 = 2e4):
	return 1.0/(1.0+np.exp(-(np.log10(T)-np.log10(2e4))*Ifac))

def n0(z = REDSHIFT):
	return 0.3*((1+z)/21.0)**3

def shock(v, X):
	mH = PROTON*4.0/(1.0+3.0*X)
	return mH*(v*100000.0)**2/3.0/kB

def NH2(T,nh):
	return 4.8e19*(T/nh)**0.5

def Hubble(z):
	return H0*(OmegaM*(1+z)**3+1-OmegaM)**0.5

def tff(rho):
	return (3.0*np.pi/32.0/G/rho)**0.5

def main1(Tini = 2.0e2, nini = n0(), Xh2=0.0, Ion=1.0e-4, Li = 4.6e-10, D = 4e-5, z = REDSHIFT, mode = 1, J_21 = 0.0, num = 1e4, dtini =1.e-4, epsT = 1.e-4, epsH = 1.e-3, fac1 = 10.0, fac2 = 10.0, gamma = 5.0/3.0, X = 0.76):
	start = time.time()
	mH = PROTON*4.0/(1.0+3.0*X)
	num = int(num)
	dtmin = dtini*3.14e7
	y0 = initial(Tini, nini, Xh2, Ion, X)
	print('Mode '+str(mode)+', [D/H], [Li/H] = '+str(D)+', '+str(Li))
	if mode==1:
		tmax = min(fac1/Hubble(z), fac2*2*tff(y0[1]*mH))
	else:
		tmax = fac1/Hubble(z)
	dt0 = tmax/num
	out = evolve(D, Li, y0, dt0, dtmin, num, J_21, z, mode, epsT, epsH, gamma, X)
	end = time.time()
	print(end-start)
	return out

def main2(vini = 1.0e2, nini = 1.0e2, Xh2=1.e-16, Ion=1.0, Li = 4.6e-10, D = 4e-5, z = REDSHIFT, mode = 0, J_21 = 0.0, num = 1e4, dtini =1.e-4, epsT = 1.e-4, epsH = 1.e-3, fac1 = 10.0, fac2 = 0.9, gamma = 5.0/3.0, X = 0.76):
	dtmin = dtini*3.14e7
	Tin = shock(vini, X)
	start = time.time()
	mH = PROTON*4.0/(1.0+3.0*X)
	num = int(num)
	y0 = initial(Tin, nini, Xh2, Ion, X, [0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0])
	print('Mode '+str(mode)+', [D/H], [Li/H] = '+str(D)+', '+str(Li))
	if mode==1:
		tmax = min(fac1/Hubble(z), fac2*2*tff(y0[1]*mH))
	else:
		tmax = fac1/Hubble(z)
	dt0 = tmax/num
	out = evolve(D, Li, y0, dt0, dtmin, num, J_21, z, mode, epsT, epsH, gamma, X)
	end = time.time()
	print(end-start)
	return out

def equilibrium1(lT, num = 1e4, frac = 0.1, lim = 1e-3, maxtime = 500.0, dtini = 1e-4, eps = 1.e-3, fac = 0.05, nb = 1e2, Xh2 = 1e-16, Ifac=20.0, D = 4e-4, Li = 4.6e-10, J_21 = 0.0, gamma = 5.0/3.0, X = 0.76):
	y0 = initial(lT[0], nb, Xh2, 1.0, X)
	#print(y0)
	dtmin = dtini*3.14e7
	Ns = len(y0)-2
	mH = PROTON*4.0/(1.0+3.0*X)
	num = int(num)
	xnh, xnhe, xnd, xnli = nb*(y0[2]+y0[3]+y0[4]+2.0*(y0[5]+y0[6]))*xh(X), nb*(y0[8]+y0[9]+y0[10])*(1-xh(X)), nb*(y0[11]+y0[12]+y0[13])*xh(X)*D, nb*sum([y0[x] for x in range(12,17)])*Li*xh(X)
	refa = np.array([xnh]*6+[xnhe]*3+[xnd]*3+[xnli]*5)
	out = [[] for x in range(Ns)]
	std = [[] for x in range(Ns)]
	#data = []
	for i in range(len(lT)):
		tmax = fac*2*tff(y0[1]*mH)*(lT[len(lT)-1]/lT[i])**0.5
		dt0 = tmax/num
		start = time.time()
		lX = [[0.0] for x in range(Ns)]
		Ion = Xe(lT[i], Ifac)
		y0 = initial(lT[i], nb, Xh2, Ion, X)
		for x in range(Ns):
			if refa[x]!=0:
				lX[x][0] = y0[x+2]
			else:
				lX[x][0] = 0.0
		t_cum, count = 0.0, 0
		while t_cum<tmax:
			if count==0:
				dt = dtmin
				Cr0, Ds0 = np.zeros(Ns,dtype='float'), np.zeros(Ns,dtype='float')
				nold = np.array([lX[x][count]*refa[x] for x in range(Ns)])
			else:
				dt = dt0
				Cr0, Ds0 = abund[5], abund[6]
			abund = chemistry1(lT[i], nold, dt, eps, J_21, Ns, xnh, xnhe, xnd, xnli, Cr0, Ds0)
			t_cum += abund[1]
			nold = abund[0]
			#newX = nold/nb
			for x in range(Ns):
				if refa[x]!=0:
					lX[x].append(nold[x]/refa[x])#[count] = newX[x]
				else:
					lX[x].append(0.0)
			#ref = []
			#for m in range(Ns):
			#	if lX[m][count]>1e-30:
			#		ref.append( abs((newX[m]-lX[m][count])/lX[m][count]) )
			#final = max(ref)
			#if final<=lim:
			#	for j in range(Ns):
			#		out[j].append(newX[x])
			#		break
			count += 1
			if t_cum>=tmax:
				lerr = []
				if (lT[i]>5e3)and(lT[i]<1.5e4):
					control = range(Ns)
				elif lT[i]<2.7e4:
					control = [0,1,3,5,6,7,8,12,13,14,15,16]
				else:
					control = [1,3,5,6,7,8,16]
				for j in control:#range(Ns):
					if np.average(lX[j][-int(frac*num):])>1e-10:
						dev = np.average(lX[j][-int(frac*num):-int(frac*num/2.0)])-np.average(lX[j][-int(frac*num/2.0):])
						lerr.append(abs(dev/np.average(lX[j][-int(frac*num):])))
						print(j,lerr[len(lerr)-1])
				error = max(lerr)
				print(error)
				if error>lim:
					tmax = min(tmax*2.0,2*tff(y0[1]*mH)*maxtime)
					dt = dt*2.0
		if tmax==2*tff(y0[1]*mH)*maxtime:
			print('Fail to reach equilibrium for T =',lT[i])
		for j in range(Ns):
			out[j].append(np.average(lX[j][-int(frac*num):]))
			std[j].append(np.std(lX[j][-int(frac*num):]))
		#data.append(lX)
		#print(final)
		end = time.time()
		print('Time for '+str(lT[i])+':',end-start,'(computational)',tmax/(2*tff(y0[1]*mH)),'[2 tff] (physical)')
	totxt('Tbase.txt',[lT],0,0,0)
	totxt('equilibrium.txt',out,0,0,0)
	return {'T':np.array(lT), 'X':np.array(out), 'n':nb, 'std': np.array(std)}#, 'd':data}
					

def evolve(D, Li, y0, dt0, dtmin, num, J_21, z, mode, epsT, epsH, gamma, X):
	total = 0
	tmax = dt0*num
	print('tmax, dt_ref = '+str(tmax)+', '+str(dt0))
	Ns = len(y0)-2
	yy = np.zeros(Ns, dtype='float')
	lt = [0.0] #np.zeros(num+1,dtype='float')
	lT = [0.0] #np.zeros(num+1,dtype='float')
	ln = [0.0] #np.zeros(num+1,dtype='float')
	lX = [[0.0] for x in range(Ns)]#np.zeros(num+1,dtype='float') for x in range(Ns)]
	ltform, ltdest = [0.0], [0.0] #np.zeros(num+1,dtype='float'), np.zeros(num+1,dtype='float')
	ltform[0], ltdest[0] = dt0, dt0
	Tcmb = 2.73*(1+z)
	mH = PROTON*4.0/(1.0+3.0*X)
	lT[0], ln[0] = y0[0], y0[1]
	Pir = lT[0]*ln[0]
	nb = y0[1]
	t_cum, count = 0.0, 0
	while t_cum<tmax:
		nbold = nb
		xnh, xnhe, xnd, xnli = nb*(y0[2]+y0[3]+y0[4]+2.0*(y0[5]+y0[6]))*xh(X), nb*(y0[8]+y0[9]+y0[10])*(1-xh(X)), nb*(y0[11]+y0[12]+y0[13])*xh(X)*D,	nb*sum([y0[x] for x in range(12,17)])*Li*xh(X)
		refa = np.array([xnh]*6+[xnhe]*3+[xnd]*3+[xnli]*5)
		if count==0:
			dt_T = dtmin
			for x in range(Ns):
				if refa[x]!=0:
					lX[x][0] = y0[x+2]
					yy[x] = y0[x+2]
				else:
					lX[x][0], yy[x] = 0.0, 0.0
			nold = np.array([lX[x][count]*refa[x] for x in range(Ns)])
			Told = lT[count]
			dT_dt_old_ = cool(Told, nb, 0.0, nold, J_21, z, mode, gamma, X)
			dT_dt_old = dT_dt_old_[0]
			dnb_dt = dT_dt_old_[1]
		#if abs(dT_dt_old*dt_T)>epsT*Told:
		else:
			nold = yy*refa
		#	if Told>1e5:
		#		dt_T = dtmin
		#	else:
			dt_T = min(dt0,1.e5*3.14e7)#dt0
			if abs(dT_dt_old*dt_T)>epsT*Told:
				dt_T = epsT*Told/abs(dT_dt_old)
		if dt_T + t_cum>tmax:
			dt_T = tmax - t_cum
			t_cum = tmax
		if count==0:
			Cr0, Ds0 = np.zeros(Ns,dtype='float'), np.zeros(Ns,dtype='float')
			abund0 = chemistry1(Told, nold, dt_T, epsH, J_21, Ns, xnh, xnhe, xnd, xnli, Cr0, Ds0)
			Cr0, Ds0 = abund0[5], abund0[6]
		else:
			Cr0, Ds0 = abund[5], abund[6]
		abund = chemistry1(Told, nold, dt_T, epsH, J_21, Ns, xnh, xnhe, xnd, xnli, Cr0, Ds0)
		nold = abund[0]
		for x in range(Ns):
			if refa[x]!=0:
				yy[x] = nold[x]/refa[x]
			else:
				yy[x] = 0.0
		t_cum += abund[1]
		dT_dt = cool(Told, nb, dnb_dt, nold, J_21, z, mode, gamma, X)
		Told = Told + (dT_dt[0] + dT_dt_old)*abund[1]/2.0
		dT_dt_old = dT_dt[0]
		if Told<=Tcmb:
			Told = Tcmb
			dt_T = min(dt0,1.e6*3.14e7)
		nb = nb + (dT_dt[1] + dnb_dt)*abund[1]/2.0
		dnb_dt = dT_dt[1]
		if mode==0:
			nb = Pir/Told
		if mode<0:
			dnb_dt = (nb-nbold)/dt_T
		#dt_T = min(dt_T*2.0, dt0 - t_cumT)
		total += abund[4]
		count += 1
		if (count%100==0)or(t_cum>=tmax):
			lt.append(t_cum)#[count] = t_cum
			lT.append(Told)#[count] = Told
			ln.append(nb)#[count] = nb
			for x in range(Ns):
				if refa[x]!=0:
					lX[x].append(nold[x]/refa[x])#[count] = newX[x]
				else:
					lX[x].append(0.0)
			ltform.append(abund[2])#[count] = abund[2]
			ltdest.append(abund[3])#[count] = abund[3]
		#if Told==Tcmb:
		#	break
		#print(t_cum)
		if mode==1:
			if nb>=NMAX:
				lt.append(t_cum)#[count] = t_cum
				lT.append(Told)#[count] = Told
				ln.append(nb)#[count] = nb
				for x in range(Ns):
					if refa[x]!=0:
						lX[x].append(nold[x]/refa[x])#[count] = newX[x]
					else:
						lX[x].append(0.0)
				ltform.append(abund[2])#[count] = abund[2]
				ltdest.append(abund[3])#[count] = abund[3]
				break
	return {'t':np.array(lt)*Hubble(z), 'T': np.array(lT), 'n': np.array(ln), 'X': np.array(lX), 'tform': np.array(ltform), 'tdest': np.array(ltdest), 'num': total}

def cool(T, ntot, ndot, n, J_21, z, mode, gamma, X):
	Tcmb = 2.73*(1+z)
	mH = PROTON*4.0/(1.0+3.0*X)
	out = np.zeros(2,dtype='float')
	nh = n[0]+n[1]+n[2]+2.0*(n[3]+n[4])
	ny = n
	Gam = 0.0 #J_21*(5.1e-23*n[0] + 1.2e-22*n[6] + 2.5e-24*n[7])
	L = np.zeros(11,dtype='float')
	if T>5e3:
		gff=1.1+0.34*np.exp(-(5.5-np.log10(T))**2/3.0)
		L[0] = 1.42e-27*gff*np.sqrt(T)*(ny[1]+ny[7]+4.0*ny[8])*ny[5]
	else:
		L[0] = 1.e-42
	T5 = T/1.e5
	XH1 = np.sqrt(T)/(1.0+np.sqrt(T5))
	if T>5e3:
		L[1]=1.27e-21*XH1*np.exp(-157809.1/T)*ny[0]*ny[5]
	else:
		L[1]=1.e-42
	if T>8e3:
		L[2]=9.38e-22*XH1*np.exp(-285335.4/T)*ny[6]*ny[5]
	else:
		L[2]=1.e-42
	if T>1e4:
		L[3]=4.95e-22*XH1*np.exp(-631515.0/T)*ny[7]*ny[5]
	else:
		L[3]=1.e-42
	T3=T/1.e3
	T6=T/1.e6
	XH4 = 1.0/(1.e0+T6**0.7)
	if T>5e3:
		L[4]=8.7e-27*XH4*np.sqrt(T)*T3**(-0.2)*ny[1]*ny[5]
	else:
		L[4]=1.e-42
	if T>5e3:
		L[5]=1.55e-26*T**(0.3647)*ny[7]*ny[5]
	else:
		L[5]=1.e-42
	if T>5e3:
		L[6]=3.48e-26*XH4*np.sqrt(T)*T3**(-0.2)*ny[8]*ny[5]
	else:
		L[6]=1.e-42
	XH5=np.exp(-470000.e0/T)*(1.e0+0.3e0*np.exp(-94000.e0/T))
	if T>5e3:
		L[7]=1.2e-13*XH5*T**(-1.5)*ny[7]*ny[5]
	else:
		L[7]=1.e-42
	XH2=1.0/(1.0+np.sqrt(T5))
	if T>5e3:
		L[8]=7.5e-19*XH2*np.exp(-118348.0/T)*ny[0]*ny[5]
	else:
		L[8]=1.e-42
	XH3=XH2*T**(-0.397)
	if T>8e3:
		L[9]=5.54e-17*XH3*np.exp(-473638.0/T)*ny[7]*ny[5]
	else:
		L[9]=1.e-42
	L[10]=5.4e-36*(1.e0+z)**4*ny[5]*(T-Tcmb)
	#L[0] = LambdaBre(T, n[1], n[7], n[8], n[5]) 
	#L[1] = LambdaIC(T, z, n[5]) - LambdaIC(Tcmb, z, n[5]) 
	#L[2] = LambdaHI(T, n[0], n[5]) + LambdaHII(T, n[1], n[5]) 
	#L[3] = LambdaHeI(T, n[6], n[5]) + LambdaHeII(T, n[7], n[5]) + LambdaHeIII(T, n[8], n[5]) 
	LH2, LHD, LLiH = 0.0, 0.0, 0.0
	if T<=2e4:
		nhd = n[11]# 0.01*xnd 
		LH2 = LambdaH2(T, n[3], n[0]) - LambdaH2(Tcmb, n[3], n[0]) 
		LHD = LambdaHD(T, nhd, n[0], nh) - LambdaHD(Tcmb, nhd, n[0], nh) 
		LLiH = LambdaLiH(T, n[16], n[0]) - LambdaLiH(Tcmb, n[16], n[0])
	Lam = sum(L)+LH2+LHD+LLiH-Gam
	expansion = 0.0#- 3*kB*T*Hubble(z)
	if mode==0:
		out[0] = (-Lam/ntot +expansion)*(gamma-1.0)/kB/gamma
		out[1] = -out[0]*ntot/T
	elif mode<0:
		dT_c = -Lam/(1.5*kB*ntot)
		dT_ad = (gamma-1.0)*T*ndot/ntot
		out[0] = dT_c + dT_ad
		out[1] = -out[0]*ntot/T
	else:
		out[1] = ntot/tff(ntot*mH)
		out[0] = (-Lam/ntot + expansion + kB*T*out[1]/ntot)*(gamma-1.0)/kB
	return out

# For illustration
	
def equilibrium_i(lT, nb = 1e4, X = 0.76, D = 4e-5, Li = 4.6e-10, J_21 = 0.0, Ns = 17):
	xnh = nb*xh(X)
	xnd = xnh*D
	xnhe = nb-xh(X)*nb
	xnli = xnh*Li
	refa = np.array([xnh]*6+[xnhe]*3+[xnd]*3+[xnli]*5)
	#out = [[] for x in range(Ns)]
	out = []
	for s in range(len(lT)):
		ny = np.array([xnh, 0.0, 0.0, 0.0, 0.0, 0.0, xnhe, 0.0, 0.0, xnd, 0.0, 0.0, xnli, 0.0, 0.0, 0.0, 0.0])
		k = rates(J_21, lT[s], ny)
		ny[1] = xnh*k[0]/(k[0]+k[3])
		ny[0] = max(0,xnh - ny[1])
		if lT[s]>10**4:
			ny[8] = xnhe*k[1]*k[2]/(k[1]*k[2]+k[5]*k[1]+k[4]*k[5])
			ny[7] = k[5]*ny[8]/k[2]
			ny[6] = max(0,xnhe -ny[7]-ny[8])
			ny[7] = xnhe - ny[8]-ny[6]
		ny[5] = ny[1]+ny[7]+2.0*ny[8]
		#XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
		XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]#+ k[24] +k[11]*ny[4];
		if XDENOM1>1e-52:
			a = k[8]*ny[0]*ny[5]/XDENOM1
			b = k[13]*ny[5]/XDENOM1 
			#ny[2]=XNUM1/XDENOM1;
		else:
			a = 1#print('Weird production of H-.')
		#XNUM2=(k[6]*ny[0] + k[16]*ny[3]+k[20]*ny[2])*ny[1]#+ k[26]*ny[3];
		XDENOM2=k[7]*ny[0]+k[10]*ny[5]+k[11]*ny[2]#+k[25];
		if XDENOM2>1e-52:
			c = k[6]*ny[0]*ny[1]/XDENOM2
			d = k[16]*ny[1]/XDENOM2
			#ny[4]=XNUM2/XDENOM2;
		else:
			a = 1#print('Weird production of H2+.')
		XDENOM3=(k[13]+k[17])*ny[5]+k[14]*ny[0]+k[16]*ny[1]
		if XDENOM3>1e-52:
			f = k[7]*ny[0]/XDENOM3
			e = k[9]*ny[0]/XDENOM3
			base = 1.0 - b*e-d*f
			if base>1e-52:
				ny[3] = (a*e+c*f)/base
				ny[2] = a+b*ny[3]
				ny[4] = c+d*ny[3]
			else:
				a = 1#print('Fail to solve H2 directly.')
		else:
			a = 1#print('Weird production of H2.')
					
		for x in range(100):
			XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
			XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]+ k[24] +k[11]*ny[4];
			if XDENOM1>1e-52:
				ny[2]=XNUM1/XDENOM1
			else:
				a = 1#print('Weird production of H-.')
			XNUM2=(k[6]*ny[0] + k[16]*ny[3]+k[20]*ny[2])*ny[1]+ k[26]*ny[3];
			XDENOM2=k[7]*ny[0]+k[10]*ny[5]+k[11]*ny[2]#+k[25];
			if XDENOM2>1e-52:
				ny[4]=XNUM2/XDENOM2;
			else:
				a = 1#print('Weird production of H2+.')
			base0 = (k[13]*ny[5]+k[14]*ny[0]+k[15]*ny[3]+k[16]*ny[1]+k[17]*ny[5])
			if base0>1e-52:
				ny[3] = (k[7]*ny[4]*ny[0]+k[9]*ny[2]*ny[0]+k[11]*ny[4]*ny[2])/base0

		base1 = (ny[3]*k[32]*ny[1]*k[30]+ny[1]**2*k[33]*k[30]+ny[1]*k[33]*(ny[0]*k[31]+ny[5]*k[3]))
		base2 = ny[3]*k[32]
		if base1>1e-52:
			ny[11] = xnd*ny[3]*k[32]*ny[1]*k[30]/base1
		if base2>1e-52:
			ny[10] = ny[1]*k[33]*ny[11]/base2
		ny[9] = xnd - ny[10]-ny[11]
		b = k[36]*ny[5]/(k[35]*ny[1]+k[38]*ny[0])		
		a1, a2 = (k[37]*ny[0]+k[46]*ny[2])/k[39]/ny[0], k[38]/k[39]
		a = a1+a2*b
		c = k[43]*ny[0]/(k[34]*ny[5]+k[40]*ny[0])
		h = (k[44]+k[45])*ny[1]/(k[34]*ny[5]+k[40]*ny[0])
		DesLi = k[41]*ny[1]+k[36]*ny[5]+k[37]*ny[0]+(k[44]+k[45])*ny[1]+k[46]*ny[2]
		d, e, f, g = k[34]*ny[5]/DesLi, k[42]*ny[5]/DesLi, k[35]*ny[1]/DesLi, k[39]*ny[0]/DesLi
		i = k[40]*ny[0]/(k[43]*ny[0]+k[42]*ny[5])
		j = k[41]*ny[1]/(k[43]*ny[0]+k[42]*ny[5])
		if ny[5]<1e-52 or ny[1]<1e-52 or 1-i*c<=0:
			ny[12] = xnli/(a1+1.0)
			ny[16] = a1*ny[12]
		else:
			x = (j+i*h)/(1-i*c)
			y = (h+c*j)/(1-i*c)
			l = max(0.0,(1-d*y-e*x-a1*g)/(f+a2*g))
			ny[12] = xnli/(1+y+x+a1+(1+a2)*l)
			ny[13] = y*ny[12]
			ny[14] = l*ny[12]
			ny[15] = x*ny[12]
			ny[16] = a1*ny[12]+a2*ny[14]	
		out.append(ny)
	return np.array(out)

def coolfunc(T, n, J_21 = 0.0, z = 0.0, gamma = 5.0/3.0, X = 0.76):
	Tcmb = 2.73*(1+z)
	mH = PROTON*4.0/(1.0+3.0*X)
	out = np.zeros(2,dtype='float')
	nh = n[0]+n[1]+n[2]+2.0*(n[3]+n[4])
	ny = n
	Gam = 0.0 #J_21*(5.1e-23*n[0] + 1.2e-22*n[6] + 2.5e-24*n[7])
	L = np.zeros(11,dtype='float')
	if T>5e3:
		gff=1.1+0.34*np.exp(-(5.5-np.log10(T))**2/3.0)
		L[0] = 1.42e-27*gff*np.sqrt(T)*(ny[1]+ny[7]+4.0*ny[8])*ny[5]
	else:
		L[0] = 1.e-42
	T5 = T/1.e5
	XH1 = np.sqrt(T)/(1.0+np.sqrt(T5))
	if T>5e3:
		L[1]=1.27e-21*XH1*np.exp(-157809.1/T)*ny[0]*ny[5]
	else:
		L[1]=1.e-42
	if T>8e3:
		L[2]=9.38e-22*XH1*np.exp(-285335.4/T)*ny[6]*ny[5]
	else:
		L[2]=1.e-42
	if T>1e4:
		L[3]=4.95e-22*XH1*np.exp(-631515.0/T)*ny[7]*ny[5]
	else:
		L[3]=1.e-42
	T3=T/1.e3
	T6=T/1.e6
	XH4 = 1.0/(1.e0+T6**0.7)
	if T>5e3:
		L[4]=8.7e-27*XH4*np.sqrt(T)*T3**(-0.2)*ny[1]*ny[5]
	else:
		L[4]=1.e-42
	if T>5e3:
		L[5]=1.55e-26*T**(0.3647)*ny[7]*ny[5]
	else:
		L[5]=1.e-42
	if T>5e3:
		L[6]=3.48e-26*XH4*np.sqrt(T)*T3**(-0.2)*ny[8]*ny[5]
	else:
		L[6]=1.e-42
	XH5=np.exp(-470000.e0/T)*(1.e0+0.3e0*np.exp(-94000.e0/T))
	if T>5e3:
		L[7]=1.2e-13*XH5*T**(-1.5)*ny[7]*ny[5]
	else:
		L[7]=1.e-42
	XH2=1.0/(1.0+np.sqrt(T5))
	if T>5e3:
		L[8]=7.5e-19*XH2*np.exp(-118348.0/T)*ny[0]*ny[5]
	else:
		L[8]=1.e-42
	XH3=XH2*T**(-0.397)
	if T>8e3:
		L[9]=5.54e-17*XH3*np.exp(-473638.0/T)*ny[7]*ny[5]
	else:
		L[9]=1.e-42
	L[10]=5.4e-36*(1.e0+z)**4*ny[5]*(T-Tcmb)
	#L[0] = LambdaBre(T, n[1], n[7], n[8], n[5]) 
	#L[1] = LambdaIC(T, z, n[5]) - LambdaIC(Tcmb, z, n[5]) 
	#L[2] = LambdaHI(T, n[0], n[5]) + LambdaHII(T, n[1], n[5]) 
	#L[3] = LambdaHeI(T, n[6], n[5]) + LambdaHeII(T, n[7], n[5]) + LambdaHeIII(T, n[8], n[5]) 
	LH2, LHD, LLiH = 0.0, 0.0, 0.0
	if T<=2e4:
		ny[3] = ny[0]/1e3
		nhd = ny[9]/10#n[11]# 0.01*xnd
		ny[16] = ny[12] 
		LH2 = LambdaH2(T, ny[3], ny[0]) #- LambdaH2(Tcmb, n[3], n[0]) 
		LHD = LambdaHD(T, nhd, ny[0], nh) #- LambdaHD(Tcmb, nhd, n[0], nh) 
		LLiH = LambdaLiH(T, ny[16], ny[0]) #- LambdaLiH(Tcmb, n[16], n[0])
	Lam = sum(L)+LH2+LHD+LLiH-Gam
	return 	[Lam, LH2, LHD, LLiH, sum(L)]+[x for x in L]

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
	plt.show()

		
