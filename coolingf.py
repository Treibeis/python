from txt import *
from scipy.interpolate import *
kB = 1.3806e-16

RH=[-8.0,-7.522878745280337,-7.0,-6.522878745280337,-6.0,-5.522878745280337,-5.0,-4.522878745280337,-4.0,-3.522878745280337,-3.0,-2.698970004336019,-2.301029995663981,-2.0,-1.698970004336019,-1.301029995663981,-1.0,-0.6989700043360187,-0.3010299956639812,0.0,0.3010299956639812,0.6989700043360189,1.0,1.301029995663981,1.698970004336019,2.0,2.301029995663981,2.698970004336019,3.0,3.301029995663981,3.698970004336019,4.0,4.477121254719663,5.0,5.477121254719663,6.0]
ROP=[1.0E-2,3.0E-2,0.10,0.30,1.0,3.0]
TN=[2.0,2.090909090909091,2.181818181818182,2.272727272727273,2.363636363636364,2.454545454545455,2.545454545454545,2.636363636363636,2.727272727272728,2.818181818181818,2.909090909090909,3.0,3.090909090909091,3.181818181818182,3.272727272727273,3.363636363636364,3.454545454545455,3.545454545454545,3.636363636363636,3.727272727272728,3.818181818181818,3.909090909090909,4.0]
DN=[0.000,0.1666666666666667,0.3333333333333333,0.50,0.6666666666666666,0.8333333333333333,1.0,1.166666666666667,1.333333333333333,1.5,1.666666666666667,1.833333333333333,2.0,2.166666666666667,2.333333333333333,2.5,2.666666666666667,2.833333333333333,3.0,3.166666666666667,3.333333333333333,3.5,3.666666666666667,3.833333333333333,4.0,4.166666666666666,4.333333333333333,4.5,4.666666666666666,4.833333333333333,5.0,5.166666666666666,5.333333333333333,5.5,5.666666666666666,5.833333333333333,6.0,6.166666666666666,6.333333333333333,6.5,6.666666666666666,6.833333333333333,7.0,7.166666666666666,7.333333333333333,7.5,7.666666666666666,7.833333333333333,8.0,]

num=len(RH)*len(ROP)*len(TN)*len(DN)

WG = np.reshape(np.array(np.matrix(retxt('le_cube',3,39,0)).transpose()),(len(RH),len(ROP),len(TN),len(DN)))

def LambdaBre(T, nhII, nheII, nheIII, ne): #LAM(1)
	if T>5.e3:
		gff = 1.1+0.34*np.e**(-(5.5-np.log10(T))**2.0/3.0)
		L = 1.42e-27*gff*T**0.5*(nhII+nheII+4.0*nheIII)*ne
	else:
		L = 1.e-52
	return L

def LambdaIC(T, z=20, ne=1e4):
	return ne*5.41e-36*(1+z)**4*T #LAM(11)

def LambdaHeI(T, n, ne=1e4): #LAM(3)
	if T>8.e3:
		return n*ne*9.38e-22*T**0.5*np.e**(-285335.4/T)/(1+(T/10**5)**0.5)
	else:
		return 1.e-52

def LambdaHeII(T, n, ne=1e4):
	if T>1.e4:
		L1 = n*ne*4.95e-22*T**0.5*np.e**(-631515.0/T)/(1+(T/10**5)**0.5) #LAM(4)
		L2 = 0.0 #n*ne**2*5.01e-27*T**(-0.1687)*np.e**(-55338/T)/(1+(T/10**5)**0.5) #???
	else:
		L1, L2 = 1.e-52, 0.0
	if T>5.e3:
		L3 = n*ne*1.55e-26*T**0.3647 #LAM(6)
		LD = n*ne*1.24e-13*np.e**(-470000.0/T)*(1.0+0.3*np.e**(-94000.0/T))/T**1.5 #??? LAM(8)
	else:
		L3, LD = 1.e-52, 1.e-52
	if T>8.e3:
		L4 = n*ne*5.54e-17*T**(-0.397)*np.exp(-473638.0/T)/(1+(T/10**5)**0.5) #LAM(10)
	else:
		L4 = 1.e-52
	return L1+L2+L3+L4+LD

def LambdaHeIII(T, n, ne=1e4):
	if T>5.e3:
		L1 = n*ne*3.48e-26*T**0.5*(T/1000)**(-0.2)/(1+(T/10**6)**0.7) #LAM(7)
	else:
		L1 = 1.e-52
	if T>8.e3:
		L2 = 0.0 #n*ne**2*9.10e-27*T**(-0.1687)*np.e**(-13179/T)/(1+(T/10**5)**0.5) #???
	else:
		L2 = 0.0
	return L1+L2

def LambdaHI(T, n, ne=1e4):
	if T>5.e3:
		L1 = n*ne*1.27e-21*T**0.5*np.e**(-157809.1/T)/(1.0+(T/10**5)**0.5) #LAM(2)
		L2 = n*ne*7.5e-19*np.e**(-118348.0/T)/(1.0+(T/10**5)**0.5) #LAM(9)
		return L1+L2
	else:
		return 2.e-52

def LambdaHII(T, n, ne=1e4): #LAM(5)
	if T>5.e3:
		return n*ne*8.70e-27*T**0.5*(T/1000.0)**(-0.2)/(1.0+(T/1e6)**0.7)
	else:
		return 1.e-52

def LambdaHD(T, n, nh0, nh=1e8):
	aa, bb, omeg, phi, c1, c2, d1, d2 = -26.2982, -0.215807, 2.03657, 4.63258, 0.283978, -1.27333, -2.08189, 4.66288
	y = np.log10(np.min([np.max([nh,1]),1e8]))
	x = np.min([np.max([np.log10(T),np.log10(30)]),np.log10(3)+3])
	w = 0.5*y+aa*x**bb-(0.25*y**2+(c1*y+c2)*np.sin(omeg*x+phi)+(d1*y+d2))**0.5
	LHD = 10**w
	if 1: #y!=np.log10(nh) or x!=np.log10(T):
		w = -55.5725+56.649*np.log10(T)-37.9102*(np.log10(T))**2+12.698*(np.log10(T))**3-2.02424*(np.log10(T))**4+0.122393*(np.log10(T))**5
		LTE = 10**w	
		g10 = (4.4e-12+3.6e-13*T**0.77)/1.27
		g21 = (4.1e-12+2.1e-13*T**0.92)/1.27
		E10, E21 = 128.0, 255.0
		LOW = 2*g10*E10*kB*np.exp(-E10/T)+5.0/3.0*g21*E21*kB*np.exp(-E21/T)
		LOW = LOW*nh0
		LHD = LOW*LTE/(LOW+LTE)
	return LHD*n

def LambdaLiH(T, n, nh=1e3):
	a = np.ones(5,dtype='f')
	a[0], a[1], a[2], a[3], a[4] = -31.47, 8.817, -4.144, 0.8292, -0.04996
	Llow = 10**sum([a[x]*(np.log10(T))**x for x in range(len(a))])*nh
	a = np.ones(7,dtype='f')
	a[0], a[1], a[2], a[3], a[4], a[5], a[6] = -31.894, 34.3512, -31.0805, 14.9459, -3.72318, 0.455555, -0.0216129
	LLTE = 10**sum([a[x]*(np.log10(T))**x for x in range(len(a))])
	#ncrit = LLTE/Llow
	LLi = LLTE*Llow/(LLTE+Llow)
	return n*LLi

def LambdaH2(T, nh2, nh):
	WH2 = -103.0+97.59*np.log10(T)-48.05*(np.log10(T))**2+10.8*(np.log10(T))**3-0.9032*(np.log10(T))**4
	if WH2>-35:
		LOW = 10**WH2
	else:
		LOW = 0.0
	LOW = LOW*nh
	T3 = T/1000.0
	RLTE = (9.5e-22*T3**3.76*np.e**(-(0.13/T3)**3)/(1.0+0.12*T3**2.1)+3.e-24*np.e**(-0.51/T3))
	VLTE = (6.7e-19*np.e**(-5.86/T3)+1.6e-18*np.e**(-11.7/T3))
	LTE = RLTE+VLTE
	#XNCRIT = nh*(LTE/LOW)
	#LH2 = LTE/(1.0+XNCRIT/nh)
	LH2 = LOW*LTE/(LTE+LOW)
	return nh2*LH2

def ncrit(T):
	out = []
	WH2 = -103.0+97.59*np.log10(T)-48.05*(np.log10(T))**2+10.8*(np.log10(T))**3-0.9032*(np.log10(T))**4
	if WH2>-52:
		LOW = 10**WH2
	else:
		LOW = 0.0
	T3 = T/1000.0
	RLTE = (9.5e-22*T3**3.76*np.e**(-(0.13/T3)**3)/(1.0+0.12*T3**2.1)+3.e-24*np.e**(-0.51/T3))
	VLTE = (6.7e-19*np.e**(-5.86/T3)+1.6e-18*np.e**(-11.7/T3))
	LTE = RLTE+VLTE
	out.append(LTE/LOW)
	w = -55.5725+56.649*np.log10(T)-37.9102*(np.log10(T))**2+12.698*(np.log10(T))**3-2.02424*(np.log10(T))**4+0.122393*(np.log10(T))**5
	LTE = 10**w	
	g10 =4.4e-12+3.6e-13*T**0.77
	g21 = 4.1e-12+2.1e-13*T**0.92
	E10, E21 = 128.0, 255.0
	LOW = 2*g10*E10*kB*np.exp(-E10/T)+5.0/3.0*g21*E21*kB*np.exp(-E21/T)
	out.append(LTE/LOW)
	a = np.ones(5,dtype='f')
	a[0], a[1], a[2], a[3], a[4] = -31.47, 8.817, -4.144, 0.8292, -0.04996
	Llow = 10**sum([a[x]*(np.log10(T))**x for x in range(len(a))])
	a = np.ones(7,dtype='f')
	a[0], a[1], a[2], a[3], a[4], a[5], a[6] = -31.894, 34.3512, -31.0805, 14.9459, -3.72318, 0.455555, -0.0216129
	LLTE = 10**sum([a[x]*(np.log10(T))**x for x in range(len(a))])
	out.append(LLTE/Llow)
	return out

# old

def LambdaH20(T, nh2=1e-3, nh=1):
	rop = 9.0*np.e**(-170.5/T)
	if nh2>0:
		h = np.log10(np.min([np.max([nh/nh2,10**RH[0]]),10**RH[-1]]))
	else:
		h = -8.0
	d = np.log10(np.min([np.max([nh,10**DN[0]]),10**DN[-1]]))
	rop = np.min([np.max([rop,ROP[0]]),ROP[-1]])
	logT = np.min([np.max([np.log10(T),TN[0]]),TN[-1]])	
	for i in range(len(RH)):
		if h>=RH[i]:
			if h<=RH[i+1]:
				ih = i
				break;
	for i in range(len(ROP)):
		if rop>=ROP[i]:
			if rop<=ROP[i+1]:
				ir = i
				break;
	for i in range(len(TN)):
		if logT>=TN[i]:
			if logT<=TN[i+1]:
				it = i
				break;
	for i in range(len(DN)):
		if d>=DN[i]:
			if d<=DN[i+1]:
				iD = i
				break;
	y0000 = WG[ih][ir][it][iD]
	y1000 = WG[ih+1][ir][it][iD]
	y0100 = WG[ih][ir+1][it][iD]
	y1100 = WG[ih+1][ir+1][it][iD]
	y0010 = WG[ih][ir][it+1][iD]
	y1010 = WG[ih+1][ir][it+1][iD]
	y0110 = WG[ih][ir+1][it+1][iD]
	y1110 = WG[ih+1][ir+1][it+1][iD]
	y0001 = WG[ih][ir][it][iD+1]
	y1001 = WG[ih+1][ir][it][iD+1]
	y0101 = WG[ih][ir+1][it][iD+1]
	y1101 = WG[ih+1][ir+1][it][iD+1]
	y0011 = WG[ih][ir][it+1][iD+1]
	y1011 = WG[ih+1][ir][it+1][iD+1]
	y0111 = WG[ih][ir+1][it+1][iD+1]
	y1111 = WG[ih+1][ir+1][it+1][iD+1]

	t = (h-RH[ih])/(RH[ih+1]-RH[ih])
	u = (rop-ROP[ir])/(ROP[ir+1]-ROP[ir])
	v = (logT-TN[it])/(TN[it+1]-TN[it])
	w = (d-DN[iD])/(DN[iD+1]-DN[iD])
	un = 1.0
	WH2 = (un-t) * (un-u) * (un-v) * (un-w) * y0000 \
		+    t   * (un-u) * (un-v) * (un-w) * y1000 \
		+ (un-t) *    u   * (un-v) * (un-w) * y0100 \
		+    t   *    u   * (un-v) * (un-w) * y1100 \
		+ (un-t) * (un-u) *    v   * (un-w) * y0010 \
		+    t   * (un-u) *    v   * (un-w) * y1010 \
		+ (un-t) *    u   *    v   * (un-w) * y0110 \
		+    t   *    u   *    v   * (un-w) * y1110 \
		+ (un-t) * (un-u) * (un-v) *    w   * y0001 \
		+    t   * (un-u) * (un-v) *    w   * y1001 \
		+ (un-t) *    u   * (un-v) *    w   * y0101 \
		+    t   *    u   * (un-v) *    w   * y1101 \
		+ (un-t) * (un-u) *    v   *    w   * y0011 \
		+    t   * (un-u) *    v   *    w   * y1011 \
		+ (un-t) *    u   *    v   *    w   * y0111 \
		+    t   *    u   *    v   *    w   * y1111
	#print(WH2)
	LH2 = 10**WH2*nh2
	if d!=np.log10(nh) or logT!=np.log10(T): #or rop!=9.0*np.e**(-170.5/T) or h!=np.log10(nh/nh2):
		WH2 = -103.0+97.59*np.log10(T)-48.05*(np.log10(T))**2+10.8*(np.log10(T))**3-0.9032*(np.log10(T))**4
		if WH2>-30:
			LOW = 10**WH2
		else:
			LOW = 0.0
		LOW = LOW*nh
		T3 = T/1000.0
		RLTE = (9.5e-22*T3**3.76*np.e**(-(0.13/T3)**3)/(1.0+0.12*T3**2.1)+3.e-24*np.e**(-0.51/T3))
		VLTE = (6.7e-19*np.e**(-5.86/T3)+1.6e-18*np.e**(-11.7/T3))
		LTE = RLTE+VLTE
	#XNCRIT = nh*(LTE/LOW)
	#LH2 = LTE/(1.0+XNCRIT/nh)
		LH2 = nh2*LOW*LTE/(LTE+LOW)
	return LH2













