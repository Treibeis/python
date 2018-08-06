import h5py as H
from txt import *

#GRA = 6.672e-8
#GRV = 1.0
#BOL = 1.3806e-16
#BOL = 1.0
#PROTON = 1.6726e-24
#PROTON = 1.6726e-24/(2.176e-5)
#ELECTRON = 9.10938356e-28
#HBAR = 1.05457266e-27
#CHARGE = 4.80320451e-10
#H_FRAC = 0.76
#SPEEDOFLIGHT = 2.99792458e+10

UL = 3.085678e21
#UL = 1.0
UM = 1.989e43
#UM = 1.0
UV = 1.0e5
#UV = 1.0

UT = UL/UV
UD = UM/UL**3
UP = UM/UL/UT**2
UE = UM*UL**2/UT**2

G = GRA*UM*UT**2/UL**3

def temp(u, Y, X = H_FRAC, gamma = 5.0/3):
	M = PROTON*4.0/(3*X+1+4*X*Y)
	U = u*UE/UM
	tem = M*U*(gamma-1)/BOL
	return tem

def rot(a):
	return np.array(np.matrix(a).transpose())

def readf(fn='snapshot_005.hdf5'):
	f = H.File(fn, 'r')
	i0 = [x for x in f]
	#head = f[i0[0]]
	i1 = [[x for x in f[i0[i]]] for i in range(len(i0))]
	d = []
	for i in range(1,len(i0)):
		s = []
		for j in range(len(i1[i])):
			v = np.array(f[i0[i]][i1[i][j]])
			if i1[i][j]=='Coordinates' or i1[i][j]=='Velocities':
				vf = rot(v)
			else:
				vf = v
			s.append(vf)
		d.append(s) #[[np.array(f[i0[i]][i1[i][j]]) for j in range(len(i1[i]))] for i in range(1,len(i0))]
	f.flush()
	f.close()
	return {'l':d, 'i0': i0, 'i1': i1}#, 'h': head}
	
def merged(ld0 = []):
	if ld0==[]:
		return []
	else:
		n = len(ld0)
		ld = [ld0[i]['l'] for i in range(n)]
		out = [[np.hstack([ld[i][j][k] for i in range(n)]) for k in range(min([len(ld[i][j]) for i in range(n)]))] for j in range(min([len(ld[i]) for i in range(n)]))]
		return {'l':out, 'i0': ld0[0]['i0'], 'i1': ld0[0]['i1']}

def writef(fn, d):#, fr):
	f = H.File(fn, 'w')
	#ref = H.File(fr, 'r')
	#f[d['i0'][0]].create_dataset('init', data = ref[[x for x in ref][0]].values())
	g0 = [f.create_group(d['i0'][i]) for i in range(0,len(d['i0']))]
	for i in range(1,len(d['i0'])):
		for j in range(len(d['i1'][i])):
			v = d['l'][i-1][j]
			if d['i1'][i][j]=='Coordinates' or d['i1'][i][j]=='Velocities':
				vf = rot(v)
			else:
				vf = v
			g0[i][d['i1'][i][j]] = vf
	f.flush()
	f.close()
	#ref.close()
