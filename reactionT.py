from txt import *
from scipy.optimize import root
kB = 1.3806e-16


lab=['-','--','-.']
sty=np.hstack([[m for x in range(4)] for m in lab])

def xh(X = 0.76):
	return 4.0*X/(1.0+3.0*X)

def initial(Tini = 2.0e5, nini = 1.0e2, Xh2=1e-16, Ion=1.0, X = 0.76, D = [1.0, 0.0, 0.0], Li = [1.0, 0.0, 0.0, 0.0, 0.0]):
	#0: T, 1: n, 2: XH, 3: XH+, 4: XH-, 5: XH2, 6: XH2+
	#7: Xe, 8: XHe, 9: XHe+, 10: XHe++, 11: XD, 12: XD+, 13: XHD
	#14: Li, 15: Li+, 16: Li-, 17: LiH+, 18: LiH
	y = np.zeros(19, dtype='float')
	y[0] = Tini
	y[1] = nini
	y[3] = Ion#*xh(X)# H+
	y[5] = Xh2#*xh(X) # H2
	y[2] = 1.0-y[3]-y[4]-2.0*(y[5]+y[6]) # H
	y[7] = Ion#*xh(X) # e-
	y[8] = 1.0#(1-xh(X)) # He
	y[12] = Ion#*xh(X)
	y[11] = 1.0-y[12]#*xh(X)
	y[13] = D[2]#*xh(X)
	y[15] = Ion
	y[14] = 1.0-y[15]
	for x in range(5):
		y[14+x] = Li[x]
	for i in range(19):
		y[i] = max(y[i], 0.0)
	return y

def get_shield(NH2, T, xnh):
	b5 = 9.12*(T/1e4)**0.5
	#a = 1.1
	A1 = 0.8711*np.log10(T)-1.928;
	A2 = -0.9639*np.log10(T)+3.892;
	a = A1*np.exp(-0.2856*np.log10(xnh)) + A2;
	x = NH2/5e14
	f = 0.965/(1+x/b5)**a + 0.035/(1+x)**0.5 * np.exp(-8.5e-4*(1+x)**0.5)
	return f

def HDshield(NH2, NHI):
	x = NH2/2.34e19
	y1 = 1./(1+x)**0.238 * np.exp(-5.2e-3*x)
	x = NHI/2.85e23
	y2 = 1./(1+x)**1.62 * np.exp(-0.149*x)
	return y1*y2

def fshield(T, nh, xh2):
	LJ = 4.8e19*(T/nh)**0.5
	NH2 = nh*xh2*LJ
	return get_shield(NH2, T, nh)

def J21_bal(T, nh, xh2, xe, J21, Ns = 17, D = 4e-5):
	#xs = fshield(T, nh, xh2)
	n = np.zeros(Ns)
	n[0] = nh
	n[3] = nh*xh2
	n[5] = nh*xe
	xnd = D*nh
	n[9] = xnd
	n[10] = xnd*xe
	n[11] = xnd*xh2
	#k = rates(1, T, n)
	d = chemistry1(T, n, 1.0, 1e-4, J21, Ns, nh, 0, xnd, 0, [], [], 0, trace=1e-6)
	ny, dt_cum, tform, tdest, total, Cr, Ds = d
	rath = Ds[3]*nh*xh2/Cr[3]
	ratd = Ds[11]*xnd*xh2/Cr[11]
	#print(ny/nh)
	#print(tform, tform*1.2e-11*J21*fshield(T, nh, xh2))
	return rath, ratd

def rates(J_21, T, n):
	k = np.zeros(34+10+2+1,dtype='float')
	tt = T**0.5
	t5fac = 1.0/(1.0+(T/1.e5)**0.5)
	logT = np.log10(T)
	nh = n[0]+n[1]+n[2]+2.0*(n[3]+n[4])
	LJ = 4.8e19*(T/nh)**0.5
	NH2 = n[3]*LJ
	# Recombination: Hep from Black (1981, MNRAS 197, 553) Hp and Hepp from Cen (1992, ApJS 78, 341)
	if T>5e3 or 1: 
		k5a = 1.5e-10*T**-0.6353 
	else:
		k5a = 0.0
	if T>10**3.6: # 
		k5b = 1.9e-3*(T**-1.5)*np.e**(-4.7e5/T)*(1.0+0.3*np.e**(-9.4e4/T))
	else:
		k5b = 0.0
	k[4] = k5a + k5b # He+ + e- = He dielectronic recombination of HeII (Black 1992)
	k[3] = 8.4e-11/tt*(T/1.e3)**-0.2/(1.0+(T/1.e6)**0.7) # H+ + e- = H + hnu
	k[5] = 3.36e-10/tt*((T/1.e3)**-0.2)/(1.0+(T/1.e6)**0.7) # He++ + e- = He+ + hnu (Cen 1992)
	# Collisional ionization: from Black, with Cen fix at high-T):
	if T>10**2.8:# H + e- = H+ + 2e-
		k[0] = 5.85e-11*tt*np.e**(-157809.1/T)*t5fac 
	else:
		k[0] = 0.0
	if T>1e3: # He + e- = He+ + 2e- (Cen 1992)
		k[1] = 2.38e-11*tt*np.e**(-285335.4/T)*t5fac
	else:
		k[1] = 0.0
	if T>10**3.4: # He+ + e- = He++ + 2e- (Cen 1992)
		k[2] = 5.68e-12*tt*np.e**(-631515.0/T)*t5fac
	else:
		k[2] = 0.0
	# Additional
	if T<=4000.0: # H + H+ = H2+ + hnu ???
		k[6] = 1.38e-23*(T**1.845)
	else:
		if T<1e5:
			k[6] = -6.157e-17 + 3.255e-20*T - 4.152e-25*T**2
		else:
			k[6] = 0.0
	if k[6]<=0.0:
		k[6] = 0.0
	k[7] = 6.4e-10 # H2+ + H = H2 + H+
	# Abel et al. 1997
	if T<6000 or 1: # H + e- = H- + hnu
		k[8] = 1.429e-18*T**0.762*T**(0.1523*logT)*T**(-3.274e-2*logT**2) 
	else:
		k[8] = 3.802e-17*T**(0.1998*logT)*10**(4.04e-5*logT**6-5.45e-3*logT**4)
	k[9] = 1.3e-9 # H + H- = H2 + e-
	k[10] = 1.68e-8*(T/300.0)**-0.29 # e- + H2+ = 2H ???
	k[11] = 5.0e-6/tt # H2+ + H- = H2 + H Too efficient???
	k[12] = 7.0e-7/tt # H- + H+ = 2H  
	if T>10**2.2: # e- + H2 = H- + H   ?
		k[13] = (2.7e-8/(T*tt))*np.e**(-43000.0/T)*t5fac #!!!
	else:
		k[13] = 0.0
	if T<=10**2.2: # H2 + H = 3H, H2 + H2 = H2 + H + H   ???
		k[14], k[15] = 0.0, 0.0 #!!!
	else:
		x = logT-4.0
		ncr_1 = 10**(4.0-0.416*x-0.327*x**2)
		ncr_2 = 10**(4.13-0.968*x+0.119*x**2)
		kh_1 = 3.52e-9*np.e**(-43900.0/T)
		kh_2 = 5.48e-9*np.e**(-53000.0/T)
		if T>=7390:
			kl_1 = 6.11e-14*np.e**(-29300.0/T)
		else:
			if T>10**2.6:
				kl_1 = 2.67e-15*np.e**(-(6750.0/T)**2)
			else:
				kl_1 = 0.0
		if T>=7291:
			kl_2 = 5.22e-14*np.e**(-32200.0/T)
		else:
			if T>10**2.8:
				kl_2 = 3.17e-15*np.e**(-(4060.0/T)-(7500.0/T)**2)
			else:
				kl_2 = 0.0
		k[14] = kh_1*(kl_1/kh_1)**(1.0/(1.0+nh/ncr_1))
		k[15] = kh_2*(kl_2/kh_2)**(1.0/(1.0+nh/ncr_2))
	# Shapiro and Kang
	if T>10**2 : # H2 + H+ = H2+ + H
		k[16] = 2.4e-9*np.e**(-21200.0/T) # !!!
	else:
		k[16] = 0.0
	if T>10**2.6: # H2 + e- = 2H + e-
		k[17] = 4.38e-10*np.e**(-102000.0/T)*T**0.35 # !!!
	else:
		k[17] = 0.0
	if T>10**1.6: # H- + e- = H + 2e-
		k[18] = 4.e-12*T*np.e**(-8750.0/T)
	else:
		k[18] = 0.0
	if T>10**1.6: # H- + H = 2H + e-
		k[19] = 5.3e-20*T**2.17*np.e**(-8750.0/T)
	else:
		k[19] = 0.0
	if T>1e4: # H- + H+ = H2+ + e-   ?
		k[20] = 1.e-8*T**-0.4
	else:
		if T>10**1.8:
			k[20] = 4.e-4*T**-1.4*np.e**(-15100.0/T)
		else:
			k[20] = 0.0
	# Radiative processes
	#if NH2<=1:
	#	xshield = 1.0
	#else:
	#	xshield = NH2**-0.75
	xshield = get_shield(NH2, T, nh)
	k[21] = 0#5.16e-12*J_21 # H + hnu = H+ + e-
	k[22] = 0#7.9e-12*J_21  # He -> He+
	k[23] = 0#1.9e-13*J_21	# He+ -> He++
	k[24] = 1.1e-10*J_21 * 1.71 * xshield # H- +hnu = H + e- ?
	k[25] = 4.8e-12*J_21 * xshield # H2+ + hnu = H + H+
	k[26] = 1.2e-11*J_21 * xshield # H2 + hnu = H2+ + e-
	k[27] = 1.38e-12 * 0.97 * J_21 * xshield
	
	# Three-body reations from Palla et al.,1983,ApJ 271,632
	k[28] = 5.5e-29/T # H + H + H = H2 + H
	k[29] = 0.125*k[28] # H + H + H2 = H2 + H2
# HD from Galli 1998
	k[30]=3.7e-10*T**0.28*np.e**(-43.0/T) # D + H+ = D+ + H
	k[31]=3.7e-10*T**0.28 # D+ + H = D + H+
	k[32]=2.1e-9 # D+ + H2 = H+ + HD
	k[33]=1.e-9*np.e**(-464.0/T) # HD + H+ = H2 + D+

	k[34]=1.036e-11/((T/107.7)**0.5*(1+(T/107.7)**0.5)**0.612*(1+(T/1.177e7)**0.5)**1.388)
	# Li+ + e- = Li + hnu
	k[35]=6.3e-6*T**-0.5-7.6e-9+2.6e-10*T**0.5+2.7e-14*T # Li- + H+ = Li + H
	k[36]=6.1e-17*T**0.58*np.exp(-T/17200) # Li + e- = Li- + hnu
	k[37]= 1.0/(5.6e19*T**-0.15+7.2e15*T**1.21) # Li(2S) + H = LiH + hnu
	k[38]=4.0e-10 # Li- + H = LiH + e-
	k[39]=2.0e-11 # LiH + H = Li + H2
	k[40]=10**(-22.4+0.999*np.log10(T)-0.351*np.log10(T)**2) # Li+ + H = LiH+ + hnu
	k[41]=4.8e-14*T**-0.49 # Li + H+ = LiH+ + hnu
	k[42]=3.8e-7*T**-0.47 # LiH+ + e- = Li + H
	k[43]=3.0e-10 # LiH+ + H = Li+ + H2
	k[44]=0.0#2.5e-40*T**7.9*np.exp(-T/1210)
	k[45]=0.0#1.7e-13*T**-0.051*np.exp(-T/282000)
	k[46]=0.0#4.e-10

	S2 = 1.0/(1.0+3.0*np.exp(-1.85*1.6*10**-12/T/kB))
	k[37]=k[37]*S2 + (1.9e-14*T**-0.34+2.0e-16*T**0.18*np.exp(-T/5100.0))*(1.0-S2)
	if k[35]<=0:
		k[35] = 0.0
	return k

lT = 10**np.linspace(1,6,200)
lT0 = np.hstack([10**np.linspace(1,3.7,10), 10**np.linspace(3.8,4.8,20), 10**np.linspace(5,6,10)])

def equilibrium2(lT, nb = 1e2, X = 0.76, D = 4e-5, Li = 4.6e-10, J_21 = 0.0, Ns = 17):
	xnh = nb*xh(X)
	xnd = xnh*D
	xnhe = nb-xh(X)*nb
	xnli = xnh*Li
	refa = np.array([xnh]*6+[xnhe]*3+[xnd]*3+[xnli]*5)
	out = [[] for x in range(Ns)]
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
		XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]+ k[24] +k[11]*ny[4];
		if XDENOM1>1e-52:
			a = k[8]*ny[0]*ny[5]/XDENOM1
			b = k[13]*ny[5]/XDENOM1 
			#ny[2]=XNUM1/XDENOM1;
		else:
			print('Weird production of H-.')
		#XNUM2=(k[6]*ny[0] + k[16]*ny[3]+k[20]*ny[2])*ny[1]#+ k[26]*ny[3];
		XDENOM2=k[7]*ny[0]+k[10]*ny[5]+k[11]*ny[2]+k[25];
		if XDENOM2>1e-52:
			c = k[6]*ny[0]*ny[1]/XDENOM2
			d = k[16]*ny[1]/XDENOM2
			#ny[4]=XNUM2/XDENOM2;
		else:
			print('Weird production of H2+.')
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
				print('Fail to solve H2 directly.')
		else:
			print('Weird production of H2.')
					
		for x in range(100):
			XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
			XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]+ k[24] +k[11]*ny[4];
			if XDENOM1>1e-52:
				ny[2]=XNUM1/XDENOM1
			else:
				print('Weird production of H-.')
			XNUM2=(k[6]*ny[0] + k[16]*ny[3]+k[20]*ny[2])*ny[1]+ k[26]*ny[3];
			XDENOM2=k[7]*ny[0]+k[10]*ny[5]+k[11]*ny[2]+k[25];
			if XDENOM2>1e-52:
				ny[4]=XNUM2/XDENOM2;
			else:
				print('Weird production of H2+.')
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
		#	l = 1-(f*b+g*a+d*h)
		#	if l>0:
		#		ny[12] = xnli/(a+b+1+h+(1+c)*l/(d*c+e))
		#		ny[14] = b*ny[12]
		#		ny[16] = a*ny[12]
		#		ny[15] = l*ny[12]/(d*c+e)
		#		ny[13] = c*ny[15]+h*ny[12]
		#	else:
		#		print('Running out of LiH+ at T =',lT[s])
		#		ny[15] = 0.0
		#		m = 1-d*h-a*g
		#		if m>0:
		#			ny[12] = xnli/(1+b+a+m/f)
		#			ny[13] = b*ny[12]
		#			ny[16] = a*ny[12]
		#			ny[14] = m*ny[12]/f
		#		else:
		#			print('Running out of Li- at T =',lT[s])
		#			ny[14] = 0.0
		#			q = 1-d*h
		#			if q>0:
		#				ny[12] = xnli/(1+b+q/g)
		#				ny[13] = b*ny[12]
		#				ny[16] = q*ny[12]/g
		#			else:
		#				print('Running out of LiH at T =',lT[s])
		#				ny[16] = 0
		#				ny[13] = xnli/(1+d)
		#				ny[12] = d*ny[13]
			x = (j+i*h)/(1-i*c)
			y = (h+c*j)/(1-i*c)
			l = max(0.0,(1-d*y-e*x-a1*g)/(f+a2*g))
			ny[12] = xnli/(1+y+x+a1+(1+a2)*l)
			ny[13] = y*ny[12]
			ny[14] = l*ny[12]
			ny[15] = x*ny[12]
			ny[16] = a1*ny[12]+a2*ny[14]	
				
		for j in range(Ns):
			if refa[j]!=0:
				out[j].append(ny[j]/refa[j])
			else:
				out[j].append(0.0)
	return  {'T':np.array(lT), 'X':np.array(out), 'n':nb}
	
def chemistry1(T, nin, dt0, epsH, J_21, Ns, xnh, xnhe, xnd, xnli, Cr0, Ds0, H2_flag, trace=1e-6):
	total = 0
	out = np.zeros(Ns,dtype='float')
	dt_cum = 0.0
	dt = dt0
	Cr, Ds = np.zeros(Ns,dtype='float'), np.zeros(Ns,dtype='float')
	ny = nin
	nh = ny[0]+ny[1]+ny[2]+2.0*(ny[3]+ny[4])
	LJ = 4.8e19*(T/nh)**0.5
	NH2 = ny[3]*LJ
	NHD = ny[11]*LJ
	NHI = ny[0]*LJ
	while dt_cum<dt0:
		k = rates(J_21, T, ny)
		Cr[5]=k[21]*ny[0]+k[22]*ny[6]+k[23]*ny[7]+(k[0]*ny[0]+k[1]*ny[6]+k[2]*ny[7])*ny[5]
		Ds[5]=k[3]*ny[1]+k[4]*ny[7]+k[5]*ny[8]
		if (dt*abs(Cr[5]-Ds[5]*ny[5])>epsH*abs(ny[5])) and (ny[5]>0):
			dt = dt/2.0 #epsH*ny[5]/abs(Cr[5]-Ds[5]*ny[5])#
			continue
		if (ny[5]<=1e-4*xnh and T<2e4 and H2_flag>=1):
			for i in [3,11,16]:
				if (dt*abs(Cr0[i]-Ds0[i]*ny[i])>epsH*abs(ny[i])) and (ny[i]/xnh>trace):
					dt = epsH*abs(ny[i]/(Cr0[i]-Ds0[i]*ny[i])) #dt/2.0
		
		#if dt>1.e5*3.14e7:
		#	dt = 1.e5*3.14e7
		
		if dt + dt_cum>dt0:
			dt = dt0 - dt_cum
			dt_cum = dt0
		else:
			dt_cum += dt
		Cr[5]=k[21]*ny[0]+k[22]*ny[6]+k[23]*ny[7]+(k[0]*ny[0]+k[1]*ny[6]+k[2]*ny[7])*ny[5] #+k[24]*ny[2]+k[26]*ny[3]
		Ds[5]=k[3]*ny[1]+k[4]*ny[7]+k[5]*ny[8]
		ny[5]=(ny[5]+Cr[5]*dt)/(1.e0+Ds[5]*dt)

		Cr[0]=k[3]*ny[1]*ny[5];
		Ds[0]=k[0]*ny[5]+k[21]+2*k[28]*ny[0]**2+2*k[29]*ny[0]*ny[3];
		ny[0]=(ny[0]+Cr[0]*dt)/(1.e0+Ds[0]*dt);

		Cr[1]=k[0]*ny[5]*ny[0] +k[21]*ny[0]
		Ds[1]=k[3]*ny[5];
		ny[1]=(ny[1]+Cr[1]*dt)/(1.e0+Ds[1]*dt);

		Cr[6]=k[4]*ny[7]*ny[5];
		Ds[6]=k[1]*ny[5]+k[22];
		ny[6]=(ny[6]+Cr[6]*dt)/(1.e0+Ds[6]*dt);

		Cr[7]=(k[1]*ny[5]+k[22])*ny[6]+k[5]*ny[5]*ny[8];
		Ds[7]=(k[2]+k[4])*ny[5]+k[23];
		ny[7]=(ny[7]+Cr[7]*dt)/(1.e0+Ds[7]*dt);

		Cr[8]=k[2]*ny[7]*ny[5]+k[23]*ny[7];
		Ds[8]=k[5]*ny[5];
		ny[8]=(ny[8]+Cr[8]*dt)/(1.e0+Ds[8]*dt);
#/**** calculate equilibrium abundance for H- *********************/
		XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
		XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]+ k[24] +k[11]*ny[4];
		#print(XNUM1, XDENOM1)
		if XDENOM1>1e-52:
			ny[2]=XNUM1/XDENOM1;
		else:
			print('Weird production of H-.')
#/**** calculate equilibrium abundance for H2+ ********************/
		XNUM2=(k[6]*ny[0] + k[16]*ny[3]+k[20]*ny[2])*ny[1] +k[26]*ny[3];
		XDENOM2=k[7]*ny[0]+k[10]*ny[5]+k[11]*ny[2]+k[25];
		if XDENOM2>1e-52:
			ny[4]=XNUM2/XDENOM2;
		else:
			print('Weird production of H2+.')

		Ds[3]=k[13]*ny[5]+k[14]*ny[0]+k[15]*ny[3]+k[16]*ny[1]+k[17]*ny[5]+k[26]+k[27];
		Cr[3]=k[7]*ny[4]*ny[0]+k[9]*ny[2]*ny[0]+k[11]*ny[4]*ny[2]+k[28]*ny[0]**3+k[29]*ny[0]**2*ny[3];
		ny[3]=(ny[3]+Cr[3]*dt)/(1.e0+Ds[3]*dt);
		tform, tdest = 0.0, 0.0
		if Cr[3]>1e-52:
			tform=ny[3]/Cr[3];
		if Ds[3]>1e-52:
			tdest=1.e0/Ds[3];

		ny[0]=xnh-2.e0*ny[3]-ny[1]-ny[2]-2.e0*ny[4];
		if (ny[0] < 0.e0):
			ny[0]=0.e0;
		ny[6]=xnhe-ny[7]-ny[8];
		if (ny[6] < 0.e0):
			ny[6]=0.e0;
		ny[7]=xnhe-ny[6]-ny[8]
		if (ny[7]<0.0):
			ny[7]=0.0
		ny[8] = xnhe-ny[6]-ny[7]
		if ny[8]<0.0:
			ny[8]=0.0
		ny[1]=ny[5]+ny[2]-ny[4]-ny[7]-2.e0*ny[8];
		if (ny[1] < 0.e0):
			ny[1]=0.e0;
		ny[0]=xnh-2.e0*ny[3]-ny[1]-ny[2]-2.e0*ny[4];
		if (ny[0] < 0.e0):
			ny[0]=0.e0;

		XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
		XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]+ k[24] +k[11]*ny[4];
		ny[2]=XNUM1/XDENOM1;

		for i in range(9):
			if ny[i]<1e-30:
				ny[i] = 0.0

		kdeHD = 1.38e-12 * 0.97 * J_21 * get_shield(NHD, T, nh) * HDshield(NH2, NHI)

		Cr[10]=k[30]*ny[9]*ny[1]+k[33]*ny[11]*ny[1]
		Ds[10]=k[3]*ny[5]+k[31]*ny[0]+k[32]*ny[3]
		ny[10]=min((ny[10]+Cr[10]*dt)/(1.0+Ds[10]*dt),xnd)
		Cr[11]=k[32]*ny[10]*ny[3]
		Ds[11]=k[33]*ny[1] + kdeHD
		ny[11]=min((ny[11]+Cr[11]*dt)/(1.0+Ds[11]*dt),xnd)
		ny[9]=xnd-ny[11]-ny[10]

		Cr[12] = k[34]*ny[13]*ny[5]+k[42]*ny[15]*ny[5]+k[35]*ny[14]*ny[1]+k[39]*ny[16]*ny[0]
		Ds[12] = k[41]*ny[1]+k[36]*ny[5]+k[37]*ny[0]+(k[44]+k[45])*ny[1]+k[46]*ny[2]
		ny[12] = min((ny[12]+Cr[12]*dt)/(1.0+Ds[12]*dt),xnli)

		Cr[13] = k[43]*ny[0]*ny[15]+(k[44]+k[45])*ny[12]*ny[1]
		Ds[13] = k[34]*ny[5]+k[40]*ny[1]
		ny[13] = min((ny[13]+Cr[13]*dt)/(1.0+Ds[13]*dt),xnli)

		Cr[15] = k[40]*ny[0]*ny[13]+k[41]*ny[1]*ny[12]
		Ds[15] = k[43]*ny[0]+k[42]*ny[5]
		ny[15] = min((ny[15]+Cr[15]*dt)/(1.0+Ds[15]*dt),xnli)

		Cr[16] = k[37]*ny[12]*ny[0]+k[38]*ny[14]*ny[0]+k[46]*ny[2]*ny[12]
		Ds[16] = k[39]*ny[0]
		ny[16] = min((ny[16]+Cr[16]*dt)/(1.0+Ds[16]*dt),xnli)

		Cr[14] = k[36]*ny[12]*ny[5]
		Ds[14] = k[35]*ny[1]+k[38]*ny[0]
		ny[14] = min((ny[14]+Cr[14]*dt)/(1.0+Ds[14]*dt),xnli)

		#ny[16] = xnli
		ny[12] = min(max(0,xnli-ny[15]-ny[13]-ny[14]-ny[16]),xnli)
		ny[13] = min(max(0,xnli-ny[12]-ny[14]-ny[15]-ny[16]),xnli)
		ny[14] = min(max(0,xnli-ny[12]-ny[13]-ny[15]-ny[16]),xnli)
		ny[15] = min(max(0,xnli-ny[12]-ny[13]-ny[14]-ny[16]),xnli)
		ny[16] = min(max(0,xnli-ny[12]-ny[14]-ny[15]-ny[13]),xnli)

		for i in range(17):
			if ny[i]<1e-30:
				ny[i] = 0.0
		#dt = min(dt*2.0, dt0 - dt_cum)
		#print(dt)
		dt = dt0
		total += 1
	#tform, tdest = 0.0, 0.0
	return [ny, dt_cum, tform, tdest, total, Cr, Ds]

title1 = [r'$\mathrm{H^{0}}$', r'$\mathrm{H^{+}}$', r'$\mathrm{e^{-}}$', r'$\mathrm{He}$', r'$\mathrm{He^{+}}$', r'$\mathrm{He^{++}}$', r'$\mathrm{H_{2}^{+}}$', r'$\mathrm{H_{2}}$', r'$\mathrm{HD}$']
index1 = [0,1,5,6,7,8,4,3,11]

title2 = [r'$\mathrm{H_{2}}$', r'$\mathrm{H^{0}}$', r'$\mathrm{H^{+}}$', r'$\mathrm{e^{-}}$', r'$\mathrm{Li}$', r'$\mathrm{Li^{+}}$', r'$\mathrm{Li^{-}}$', r'$\mathrm{LiH^{+}}$', r'$\mathrm{LiH}$']
index2 = [3,0,1,5,12,13,14,15,16]

title3 = [r'$\mathrm{H^{0}}$', r'$\mathrm{H^{+}}$', r'$\mathrm{e^{-}}$', r'$\mathrm{D}$', r'$\mathrm{D^{+}}$', r'$\mathrm{HD}$', r'$\mathrm{H^{-}}$', r'$\mathrm{H_{2}^{+}}$', r'$\mathrm{H_{2}}$']
index3 = [0,1,5,9,10,11,2,4,3]

title4 = [r'$\mathrm{H^{0}}$', r'$\mathrm{H^{+}}$', r'$\mathrm{e^{-}}$', r'$\mathrm{H_{2}}$', r'$\mathrm{HD}$', r'$\mathrm{LiH}$', r'$\mathrm{Li}$', r'$\mathrm{Li^{+}}$', r'$\mathrm{LiH^{+}}$']
index4 = [0,1,5,3,11,16,12,13,15]

title0 = [r'$\mathrm{H^{0}}$', r'$\mathrm{H^{+}}$', r'$\mathrm{e^{-}}$', r'$\mathrm{H_{2}^{+}}$', r'$\mathrm{H_{2}}$', r'$\mathrm{HD}$', r'$\mathrm{Li^{-}}$', r'$\mathrm{LiH^{+}}$', r'$\mathrm{LiH}$']
index0 = [0,1,5,4,3,11,14,15,16]

def plotA(d, name = 'EA_test', index = index2, tit = title2, size = (16,16)):
	subnum = [331+x for x in range(9)]
	plt.figure(figsize=size)
	for i in range(9):
		plt.subplot(subnum[i])
		plt.plot(d['T'],d['X'][index[i]])
		plt.xscale('log')
		plt.xlabel(r'$T\ [\mathrm{K}]$')
		plt.ylabel(r'rAbundance')
		plt.title(tit[i])
	plt.tight_layout()
	plt.savefig(name+'.pdf')
	plt.show()

def plotAN(d1, d0, name = 'EAN', index = index0, tit = title0, size = (16,16)):
	subnum = [331+x for x in range(9)]
	plt.figure(figsize=size)
	for i in range(9):
		plt.subplot(subnum[i])
		plt.plot(d1['T'],d1['X'][index[i]],color='k')
		plt.plot(d0['T'],d0['X'][index[i]],'^',color='k')
		plt.xscale('log')
		if i<3:
			plt.ylim(0.0,1.2)
		plt.xlabel(r'$T\ [\mathrm{K}]$')
		plt.ylabel(r'rAbundance')
		plt.title(tit[i])
	plt.tight_layout()
	plt.savefig(name+'.pdf')
	plt.show()


def chemistry2(T, nin, dt0, epsH, J_21, Ns, xnh, xnhe, xnd, Cr0, Ds0):
	total = 0
	out = np.zeros(Ns,dtype='float')
	dt_cum = 0.0
	dt = dt0
	Cr, Ds = np.zeros(Ns,dtype='float'), np.zeros(Ns,dtype='float')
	Crr, Dss = np.zeros(Ns,dtype='float'), np.zeros(Ns,dtype='float')
	ny = nin
	nyy = nin
	while dt_cum<dt0:
		k = rates(J_21, T, ny)
		Cr[5]=k[21]*ny[0]+k[22]*ny[6]+k[23]*ny[7]+(k[0]*ny[0]+k[1]*ny[6]+k[2]*ny[7])*ny[5]
		Ds[5]=k[3]*ny[1]+k[4]*ny[7]+k[5]*ny[8]
		if (dt*abs(Cr[5]-Ds[5]*ny[5])>epsH*abs(ny[5])) and (ny[5]>0):
			dt = dt/2.0 #epsH*ny[5]/abs(Cr[5]-Ds[5]*ny[5])#
			continue
		for i in [3,11]:
			if (dt*abs(Cr0[i]-Ds0[i]*ny[i])>epsH*abs(ny[i])) and (ny[i]/xnh>1e-10):
				dt = epsH*abs(ny[i]/(Cr0[i]-Ds0[i]*ny[i])) #dt/2.0
		if dt + dt_cum>dt0:
			dt = dt0 - dt_cum
			dt_cum = dt0
		else:
			dt_cum += dt
		Cr[5]=k[21]*ny[0]+k[22]*ny[6]+k[23]*ny[7]+(k[0]*ny[0]+k[1]*ny[6]+k[2]*ny[7])*ny[5] +k[24]*ny[2]+k[26]*ny[3]
		Ds[5]=k[3]*ny[1]+k[4]*ny[7]+k[5]*ny[8]
		ny[5]=(ny[5]+Cr[5]*dt)/(1.e0+Ds[5]*dt)

		Cr[0]=k[3]*ny[1]*ny[5];
		Ds[0]=k[0]*ny[5]+k[21];
		ny[0]=(ny[0]+Cr[0]*dt)/(1.e0+Ds[0]*dt);

		Cr[1]=k[0]*ny[5]*ny[0]+k[21]*ny[0]
		Ds[1]=k[3]*ny[5];
		ny[1]=(ny[1]+Cr[1]*dt)/(1.e0+Ds[1]*dt);

		Cr[6]=k[4]*ny[7]*ny[5];
		Ds[6]=k[1]*ny[5]+k[22];
		ny[6]=(ny[6]+Cr[6]*dt)/(1.e0+Ds[6]*dt);

		Cr[7]=(k[1]*ny[5]+k[22])*ny[6]+k[5]*ny[5]*ny[8];
		Ds[7]=(k[2]+k[4])*ny[5]+k[23];
		ny[7]=(ny[7]+Cr[7]*dt)/(1.e0+Ds[7]*dt);

		Cr[8]=k[2]*ny[7]*ny[5]+k[23]*ny[7];
		Ds[8]=k[5]*ny[5];
		ny[8]=(ny[8]+Cr[8]*dt)/(1.e0+Ds[8]*dt);
#/**** calculate equilibrium abundance for H- *********************/
		XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
		XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]+ k[24] +k[11]*ny[4];
		#print(XNUM1, XDENOM1)
		if XDENOM1>1e-52:
			ny[2]=XNUM1/XDENOM1;
		else:
			print('Weird production of H-.')
#/**** calculate equilibrium abundance for H2+ ********************/
		XNUM2=(k[6]*ny[0] + k[16]*ny[3]+k[20]*ny[2])*ny[1]+ k[26]*ny[3];
		XDENOM2=k[7]*ny[0]+k[10]*ny[5]+k[11]*ny[2]+k[25];
		if XDENOM2>1e-52:
			ny[4]=XNUM2/XDENOM2;
		else:
			print('Weird production of H2+.')

		Ds[3]=k[13]*ny[5]+k[14]*ny[0]+k[15]*ny[3]+k[16]*ny[1]+k[17]*ny[5]+k[26]+k[27];
		Cr[3]=k[7]*ny[4]*ny[0]+k[9]*ny[2]*ny[0]+k[11]*ny[4]*ny[2];
		ny[3]=(ny[3]+Cr[3]*dt)/(1.e0+Ds[3]*dt);
		tform, tdest = 0.0, 0.0
		#if Cr[3]>1e-52:
		#	tform=ny[3]/Cr[3];
		#if Ds[3]>1e-52:
		#	tdest=1.e0/Ds[3];

		ny[0]=xnh-2.e0*ny[3]-ny[1]-ny[2]-2.e0*ny[4];
		if (ny[0] < 0.e0):
			ny[0]=0.e0;
		ny[6]=xnhe-ny[7]-ny[8];
		if (ny[6] < 0.e0):
			ny[6]=0.e0;
		ny[1]=ny[5]+ny[2]-ny[4]-ny[7]-2.e0*ny[8];
		if (ny[1] < 0.e0):
			ny[1]=0.e0;
		ny[0]=xnh-2.e0*ny[3]-ny[1]-ny[2]-2.e0*ny[4];
		if (ny[0] < 0.e0):
			ny[0]=0.e0;

		XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
		XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]+ k[24] +k[11]*ny[4];
		ny[2]=XNUM1/XDENOM1;

		Cr[10]=k[30]*ny[9]*ny[1]+k[33]*ny[11]*ny[1]
		Ds[10]=k[3]*ny[5]+k[31]*ny[0]+k[32]*ny[3]
		ny[10]=(ny[10]+Cr[10]*dt)/(1.0+Ds[10]*dt)
		Cr[11]=k[32]*ny[10]*ny[3]
		Ds[11]=k[33]*ny[1]
		ny[11]=(ny[11]+Cr[11]*dt)/(1.0+Ds[11]*dt)
		ny[9]=xnd-ny[11]-ny[10]
			
		for i in range(12):
			if ny[i]<1e-30:
				ny[i] = 0.0
			#dt = min(dt*2.0, dt0 - dt_cum)
			#print(dt)
		ny0 = ny

		Crr[5]=k[21]*ny[0]+k[22]*ny[6]+k[23]*ny[7]+(k[0]*ny[0]+k[1]*ny[6]+k[2]*ny[7])*ny[5] +k[24]*ny[2]+k[26]*ny[3]
		Dss[5]=k[3]*ny[1]+k[4]*ny[7]+k[5]*ny[8]
		ny[5]=(ny[5]+(Cr[5]-Ds[5]*nyy[5]+Crr[5])*dt/2.0)/(1.e0+Dss[5]*dt/2.0)

		Crr[0]=k[3]*ny[1]*ny[5];
		Dss[0]=k[0]*ny[5]+k[21];
		ny[0]=(ny[0]+(Cr[0]-Ds[0]*nyy[0]+Crr[0])*dt/2.0)/(1.e0+Dss[0]*dt/2.0)

		Crr[1]=k[0]*ny[5]*ny[0]+k[21]*ny[0]
		Dss[1]=k[3]*ny[5];
		ny[1]=(ny[1]+(Cr[1]-Ds[1]*nyy[1]+Crr[1])*dt/2.0)/(1.e0+Dss[1]*dt/2.0)

		Crr[6]=k[4]*ny[7]*ny[5];
		Dss[6]=k[1]*ny[5]+k[22];
		ny[6]=(ny[6]+(Cr[6]-Ds[6]*nyy[6]+Crr[6])*dt/2.0)/(1.e0+Dss[6]*dt/2.0)

		Crr[7]=(k[1]*ny[5]+k[22])*ny[6]+k[5]*ny[5]*ny[8];
		Dss[7]=(k[2]+k[4])*ny[5]+k[23];
		ny[7]=(ny[7]+(Cr[7]-Ds[7]*nyy[7]+Crr[7])*dt/2.0)/(1.e0+Dss[7]*dt/2.0)

		Crr[8]=k[2]*ny[7]*ny[5]+k[23]*ny[7];
		Dss[8]=k[5]*ny[5];
		ny[8]=(ny[8]+(Cr[8]-Ds[8]*nyy[8]+Crr[8])*dt/2.0)/(1.e0+Dss[8]*dt/2.0)
#/**** calculate equilibrium abundance for H- *********************/
		XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
		XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]+ k[24] +k[11]*ny[4];
		#print(XNUM1, XDENOM1)
		if XDENOM1>1e-52:
			ny[2]=(ny[2]+XNUM1/XDENOM1)/2.0;
		else:
			print('Weird production of H-.')
#/**** calculate equilibrium abundance for H2+ ********************/
		XNUM2=(k[6]*ny[0] + k[16]*ny[3]+k[20]*ny[2])*ny[1]+ k[26]*ny[3];
		XDENOM2=k[7]*ny[0]+k[10]*ny[5]+k[11]*ny[2]+k[25];
		if XDENOM2>1e-52:
			ny[4]=(ny[4]+XNUM2/XDENOM2)/2.0;
		else:
			print('Weird production of H2+.')

		Dss[3]=k[13]*ny[5]+k[14]*ny[0]+k[15]*ny[3]+k[16]*ny[1]+k[17]*ny[5]+k[26]+k[27];
		Crr[3]=k[7]*ny[4]*ny[0]+k[9]*ny[2]*ny[0]+k[11]*ny[4]*ny[2];
		ny[3]=(ny[3]+(Cr[3]-Ds[3]*nyy[3]+Crr[3])*dt/2.0)/(1.e0+Dss[3]*dt/2.0)
		tform, tdest = 0.0, 0.0
		if (Cr[3]+Crr[3])/2.0>1e-52:
			tform=2.0*ny[3]/(Cr[3]+Crr[3]);
		if (Ds[3]+Dss[3])/2.0>1e-52:
			tdest=2.e0/(Ds[3]+Dss[3]);

		ny[0]=xnh-2.e0*ny[3]-ny[1]-ny[2]-2.e0*ny[4];
		if (ny[0] < 0.e0):
			ny[0]=0.e0;
		ny[6]=xnhe-ny[7]-ny[8];
		if (ny[6] < 0.e0):
			ny[6]=0.e0;
		ny[1]=ny[5]+ny[2]-ny[4]-ny[7]-2.e0*ny[8];
		if (ny[1] < 0.e0):
			ny[1]=0.e0;
		ny[0]=xnh-2.e0*ny[3]-ny[1]-ny[2]-2.e0*ny[4];
		if (ny[0] < 0.e0):
			ny[0]=0.e0;

		XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
		XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]+ k[24] +k[11]*ny[4];
		ny[2]=(XNUM1/XDENOM1+ny[2])/2.0;

		Crr[10]=k[30]*ny[9]*ny[1]+k[33]*ny[11]*ny[1]
		Dss[10]=k[3]*ny[5]+k[31]*ny[0]+k[32]*ny[3]
		ny[10]=(ny[10]+(Cr[10]-Ds[10]*nyy[10]+Crr[10])*dt/2.0)/(1.e0+Dss[10]*dt/2.0)
		Crr[11]=k[32]*ny[10]*ny[3]
		Dss[11]=k[33]*ny[1]
		ny[11]=(ny[11]+(Cr[11]-Ds[11]*nyy[11]+Crr[11])*dt/2.0)/(1.e0+Dss[11]*dt/2.0)
		ny[9]=xnd-ny[11]-ny[10]
			
		Cr = (Cr+Crr)/2.0
		for x in range(Ns):
			if ny[x]!=0:
				Ds[x] = (Ds[x]*nyy[x]+Dss[x]*ny0[x])/ny[x]
			else:
				Ds[x] = 1.0/dt
		for i in range(12):
			if ny[i]<1e-30:
				ny[i] = 0.0
		dt = dt0
		total += 1
	#tform, tdest = 0.0, 0.0
	return [ny, dt_cum, tform, tdest, total, Cr, Ds]

#def J21_bal(T, nh, xh2, xe, J21, Ns = 17, D = 4e-5):
if __name__=='__main__':
	x1, x2 = 1e-2, 1e2
	lJ = np.geomspace(x1, x2, 1000)
	lparam = [[2.5e2, 1e2, 5e-4, 1e-5]]#, [1e3, 1e1, 1e-8, 6e-5]]#, [1e3, 1e2, 1e-8, 6e-6]]
	drate = [[J21_bal(*param, j) for j in lJ] for param in lparam]
	plt.figure()
	for lrate in drate:
		lrate = np.array(lrate).T
		plt.loglog(lJ, lrate[0])
		plt.loglog(lJ, lrate[1], '--')
	plt.plot([x1, x2], [1]*2, 'k-', lw=0.5)
	plt.xlim(x1, x2)
	plt.xlabel(r'$J_{\rm LW,21}$')
	plt.ylabel(r'$k_{\rm des}/k_{\rm form}$')
	plt.tight_layout()
	plt.savefig('rate_ratio_J21.pdf')
	plt.close()
	#plt.show()



