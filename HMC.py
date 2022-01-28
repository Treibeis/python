import numpy as np
import struct
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import random
import time
from scipy.interpolate import *
from scipy.optimize import *
import txt 
from interp import *

# Units convertion

GRA = 6.672e-8
#GRV = 1.0
BOL = 1.3806e-16
#BOL = 1.0
PROTON = 1.6726e-24
#PROTON = 1.6726e-24/(2.176e-5)
ELECTRON = 9.10938356e-28
HBAR = 1.05457266e-27
CHARGE = 4.80320451e-10
H_FRAC = 0.76
SPEEDOFLIGHT = 2.99792458e+10

UL = 3.085678e24
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

# Meaningful items in the head for galaxy cluster simulation

head = ['npart[6]', 'mass[6]', 'time', 'redshift', 'Nall[6]', 'num_files', 'BoxSize', 'Omega0', 'OmegaLambda', 'HoubleParam']

name = ['Gas', 'Halo', 'Disk', 'Bulge', 'Stars', 'Bndry']

#REF1 = [-37.63071, -51.43581, 33.81053, 14.9574]
#REF2 = [-61.61415837, -16.29135888, 33.81053, 14.9574]

R_V = 2.4723322664903424
NORM = 72.144749111360142
MASS = (27.316245350212427*10**10)*(320.0/800.0)**3
IONIZATION = 0.0
BOXSIZE = 500.0
Coma = [370.0-37.63071, 370.0-51.43581, 30.0+33.81053, 14.9574]
#Coma = [370-61.61415837, 370-16.29135888,  30+33.81053, 14.9574]
Halo = [3.32205200e+02, 3.18591888e+02, 6.38388252e+01, 1.50029707e+01]
Earth = (370.0, 370.0, 30.0)
PHI = -126.18939931287795
THETA = 27.946604846341145
AVEP = (333.883698, 318.637817, 64.479408)
AVEP1 = (332.001007, 318.656891, 62.792908)
Shift = [AVEP[x] - Coma[x] for x in range(3)]
OMEGA0 = 0.258
OMEGA_B = 0.044
DEGREE = 1.2591634100184876 #h^{-1}Mpc

#D_Z = np.array(txt.retxt('/home/friede/python/d_z_WMAP5.txt', 2, 0, 0))
#ZD = interp1d(D_Z[0],D_Z[1], kind='cubic')
#H_Z = np.array(txt.retxt('/home/friede/python/H_z_WMAP5.txt', 2, 0, 0))
#HZ = interp1d(H_Z[1],H_Z[0], kind='cubic')

def CRITICAL(z):
	return 3*HZ(z)**2/8/np.pi/G
def matter(z):
	return OMEGA0*(1+z)**3/(OMEGA0*(1+z)**3+1-OMEGA0)

Tlist = [7.4893436088, 7.34968676055, 7.17056916489, 6.92443239074, 6.85329808398, 6.65483994532, 6.49941956572, 6.29488971946, 6.07869598583, 5.83977424921]
zlist = [0, 0.16442351131441257, 0.3558821137017867, 0.5788210117650425, 0.8384163062564247, 1.1406951705887787, 1.4926757871907927, 1.9025302475524395, 2.3797744625513055, 2.9354888471347733]
alist = [1/(1+zlist[x]) for x in range(10)]
alist1 = [0.53982201419, 0.578096747052, 0.619085254339, 0.662979949454, 0.709986888392, 0.760326737036, 0.814235807032, 0.871967165114, 0.933791821079, 0.97742156192]
zlist1 = [1/alist1[x]-1 for x in range(10)]
alist2 = [0.578096747052, 0.619085254339, 0.662979949454, 0.709986888392, 0.760326737036, 0.814235807032, 0.871967165114, 0.933791821079, 0.97742156192, 1]
zlist2 = [1/alist2[x]-1 for x in range(10)]
alist1.reverse()
alist2.reverse()
zlist1.reverse()
zlist2.reverse()

# Calculate the temperature based on internal energy u, metalicity X and degree of ionization Y for an adiabatic gas.

def temp(u, Y, X = H_FRAC, gamma = 5.0/3):
	M = PROTON*4.0/(3*X+1+4*X*Y)
	U = u*UE/UM
	tem = M*U*(gamma-1)/BOL
	return tem

# Read snapshot/IC files

def SZfac(dens, T, Y = 1, X = H_FRAC):
	rho = dens*UD
	M = PROTON*4.0/(3*X+1+4*X*Y)
	n = rho/M
	ni = n/((1/X-1)/4+1+Y)
	ne = Y*ni
	y = (ne*BOL*T/ELECTRON/SPEEDOFLIGHT**2)*8*np.pi*(CHARGE**2/ELECTRON/SPEEDOFLIGHT**2)**2/3
	return y

def Bremsstrahlung(dens, T, V, Y = 1, X = H_FRAC, Gaunt = 1.2):
	rho = dens*UD
	M = PROTON*4.0/(3*X+1+4*X*Y)
	n = rho/M
	ni = n/((1/X-1)/4+1+Y)
	ne = Y*ni
	J = 0.5*Gaunt*ni*ne*(CHARGE**6)*16*(2*np.pi*BOL*T/3/ELECTRON)**0.5/3/HBAR/ELECTRON/SPEEDOFLIGHT**3
	volume = V*UL**3
	return J*volume

def read(basename = 'snapshot', sn = 1, fn = 1, IC = 0, st = 0, so = -1, res = 4, tem = 1):
	output = []
	para = []
	for i in range(st, fn):
		if IC==0:
			l = load(basename+'_'+str(i).zfill(3),sn,IC,so,res,tem)
		else:
			l = load(basename,sn,IC,so,res,tem)
		output.append(l[1])
		para.append(l[0])
	d = {}
	d['head'] = [head, para]
	d['l'] = output
	return d  
# The outputs is a dictionary with 'head' leads to head items and their values at each snapshot (in the list d['head'][1][0] even if there is only one snapshot), 
# while 'l' leads to the list of the results at each snapshot.
		

# Transfer binary data blocks to numpy arrays 
	
def load(fname, fn = 1, IC = 0, so = -1, res = 4, tem = 1, ed = '<', dummy = 4):
	inte = ed+'i'
	real = ed+'f'
	linte = ed+'q'
	dreal = ed+'d'
	lpara, lp, lv, lID, lm, lU, lrho, lNe = [], [], [], [], [], [], [], []
	not_withmass = 0
	for fi in range(fn):
		if fn!=1:
			fname0 = fname+'.'+str(fi)
		else:
			fname0 = fname
		lp.append([])
		lv.append([])
		lID.append([])
		lm.append([])
		lNe.append([])
		lU.append([])
		lrho.append([])
		with open(fname0, 'rb') as f:
			# Read the head
			a = f.read(dummy)
			npart = [struct.unpack(inte, f.read(4))[0] for x in range(6)]
#			npart = [struct.unpack(linte, f.read(8))[0] for x in range(6)]
			mass = [struct.unpack(dreal, f.read(8))[0] for x in range(6)]
			time = struct.unpack(dreal, f.read(8))[0]
			Redshift = struct.unpack(dreal, f.read(8))[0]
			print ('Redshift, time = ', Redshift, time)
			flag_sfr = struct.unpack(inte, f.read(4))[0]	
			flag_feedback = struct.unpack(inte, f.read(4))[0]
			Nall = [struct.unpack(inte, f.read(4))[0] for x in range(6)]
#			Nall = [struct.unpack(linte, f.read(8))[0] for x in range(6)]
			flag_cooling = struct.unpack(inte, f.read(4))[0]
#			print ('flag_cooling', flag_cooling)
			num_files = struct.unpack(inte, f.read(4))[0]
			#if num_files!= fn:
			#	print('The number of files at each snapshot should be ',fn)
			#	return [[], []]
			BoxSize = struct.unpack(dreal, f.read(8))[0]
			Omega0 = struct.unpack(dreal, f.read(8))[0]
			OmegaLambda = struct.unpack(dreal, f.read(8))[0]
			HoubleParam = struct.unpack(dreal, f.read(8))[0]
			fill = 256-4*6-8*6-8*2-4*6-4-8*4-4*3
#			fill = 256-8*6-8*6-8*2-8*6-4-8*4-4*3
			a = f.read(fill)
			a = f.read(dummy)
			para = [npart, mass, time, Redshift, Nall, num_files, BoxSize, Omega0, OmegaLambda, HoubleParam]
			#print (para)
			# Data structure for individual files
			index = []
			for i in range(6):
				if (npart[i]!=0)and(i<res):
					index.append(i)
					lp[fi].append([])
					lv[fi].append([])
					lID[fi].append([])
					lm[fi].append([])
					if i==0:
						lU[fi].append([])
						lrho[fi].append([])
						lNe[fi].append([])
					if mass[i]==0: not_withmass += 1
			# Read positions			
			a = f.read(dummy)
			for k in range(len(index)):
				lp[fi][k] = [np.zeros(npart[index[k]], dtype='float') for x in range(3)]
				for l in range(npart[index[k]]):
					lp[fi][k][0][l] = struct.unpack(real, f.read(4))[0]
					lp[fi][k][1][l] = struct.unpack(real, f.read(4))[0]
					lp[fi][k][2][l] = struct.unpack(real, f.read(4))[0]
			for s in range(res, 6):
				if npart[s]!=0:
					for l in range(npart[s]):
						for m in range(3):
							a = f.read(4)
			a = f.read(dummy)
			# Read velocities
			a = f.read(dummy)
			for k in range(len(index)):
				lv[fi][k] = [np.zeros(npart[index[k]], dtype='float') for x in range(3)]
				for l in range(npart[index[k]]):
					lv[fi][k][0][l] = struct.unpack(real, f.read(4))[0]
					lv[fi][k][1][l] = struct.unpack(real, f.read(4))[0]
					lv[fi][k][2][l] = struct.unpack(real, f.read(4))[0]
			for s in range(res, 6):
				if npart[s]!=0:
					for l in range(npart[s]):
						for m in range(3):
							a = f.read(4)
			a = f.read(dummy)
			# Read particle IDs
			a = f.read(dummy)
			for k in range(len(index)):
				lID[fi][k] = np.zeros(npart[index[k]], dtype = 'int')
				for l in range(npart[index[k]]):
					lID[fi][k][l] = struct.unpack(inte, f.read(4))[0]
#					lID[fi][k][l] = struct.unpack(linte, f.read(8))[0]
			for s in range(res, 6):
				if npart[s]!=0:
					print(s)
					for l in range(npart[s]):
						a = f.read(4)
			a = f.read(dummy)
			# Read masses
			if not_withmass>0:
				a = f.read(dummy)
				for k in range(len(index)):
					if mass[index[k]]!=0:
						lm[fi][k] = np.ones(npart[index[k]], dtype = 'float')*mass[index[k]]
					else:
						lm[fi][k] = np.zeros(npart[index[k]], dtype = 'float')
						for l in range(npart[index[k]]):
							lm[fi][k][l] = struct.unpack(real, f.read(4))[0]
						for s in range(res, 6):
							if (npart[s]!=0)and(mass[s]==0):
								for l in range(npart[s]):
									a = f.read(4)
#						print('What?')
				a = f.read(dummy)
			else:
				for k in range(len(index)):
					lm[fi][k] = np.ones(npart[index[k]], dtype = 'float')*mass[index[k]]
			# Read gas properties
			if npart[0]!=0:
				lU[fi][0] = np.zeros(npart[0], dtype='float')
				lrho[fi][0] = np.zeros(npart[0], dtype='float')
				a = f.read(dummy)
				for l in range(npart[0]):
					lU[fi][0][l] =  struct.unpack(real, f.read(4))[0]
				a = f.read(dummy)
				if IC==0:
					a = f.read(dummy)
					for l in range(npart[0]):
						lrho[fi][0][l] = struct.unpack(real, f.read(4))[0]
					a = f.read(dummy)
				if flag_cooling!=1:
					lNe[fi][0] = np.ones(npart[0], dtype='float')*IONIZATION
				elif IC==0:
					lNe[fi][0] = np.zeros(npart[0], dtype='float')
					a = f.read(dummy)
					for l in range(npart[0]):
						lNe[fi][0][l] = struct.unpack(real, f.read(4))[0]
					a = f.read(dummy)
			for s in range(res, 6):
				para[0][s],para[4][s] = 0, 0
			#print(para)
			lpara.append(para)
	# Generate the output
	output = []
	for k in range(len(index)):
		if index[k]==0:
			if IC==0:
				if tem==1:
					output.append([[] for x in range(12)])
				else:
					output.append([[] for x in range(11)])
			else:
				if tem==1:
					output.append([[] for x in range(11)])
				else:
					output.append([[] for x in range(10)])
		else:
			output.append([[] for x in range(8)])
		output[k][0] = np.hstack([lp[fi][k][0] for fi in range(fn)])
		output[k][1] = np.hstack([lp[fi][k][1] for fi in range(fn)])
		output[k][2] = np.hstack([lp[fi][k][2] for fi in range(fn)])
		output[k][3] = np.hstack([lv[fi][k][0] for fi in range(fn)])
		output[k][4] = np.hstack([lv[fi][k][1] for fi in range(fn)])
		output[k][5] = np.hstack([lv[fi][k][2] for fi in range(fn)])
		output[k][6] = np.hstack([lID[fi][k] for fi in range(fn)])
		output[k][7] = np.hstack([lm[fi][k] for fi in range(fn)])
		if index[k]==0:
			output[0][8] = np.hstack([lU[fi][0] for fi in range(fn)])
			if IC==0:
				output[0][9] = np.hstack([lrho[fi][0] for fi in range(fn)])
				output[0][10] = np.hstack([lNe[fi][0] for fi in range(fn)])
				if tem==1:
					output[0][11] = np.zeros(np.shape(output[k][0])[0], dtype='float')
					for l in range(np.shape(output[k][0])[0]):
						output[k][11][l] = temp(output[k][8][l], output[k][10][l])
			else:
				output[0][9] = np.hstack([lNe[fi][0] for fi in range(fn)])
				if tem==1:
					output[0][10] = np.zeros(np.shape(output[k][0])[0], dtype='float')
					for l in range(np.shape(output[k][0])[0]):
						output[k][10][l] = temp(output[k][8][l], output[k][9][l])
		if so>=0:
			output[k] = sort(output[k],so)
		#output[k] = sort(output[k], 6)
	return [lpara, output]

def write(d, name = 'test_ics', IC = 0, ed = '<', dummy = 4):
	inte = ed+'i'
	real = ed+'f'
	linte = ed+'q'
	dreal = ed+'d'
	not_withmass = 0
	with open(name, 'wb') as f:
		a = f.write(struct.pack(inte, 256))
		a = [f.write(struct.pack(inte, x)) for x in d['head'][1][0][0][4]]
#		a = [f.write(struct.pack(linte, x)) for x in d['head'][1][0][0][4]]
		a = [f.write(struct.pack(dreal, x)) for x in d['head'][1][0][0][1]]
		a = f.write(struct.pack(dreal, d['head'][1][0][0][2]))
		a = f.write(struct.pack(dreal, d['head'][1][0][0][3]))
		a = f.write(struct.pack(inte, 0))
		a = f.write(struct.pack(inte, 0))
		a = [f.write(struct.pack(inte, x)) for x in d['head'][1][0][0][4]]
#		a = [f.write(struct.pack(linte, x)) for x in d['head'][1][0][0][4]]
		a = f.write(struct.pack(inte, 0))
		a = f.write(struct.pack(inte, 1))
		a = f.write(struct.pack(dreal, d['head'][1][0][0][6]))
		a = f.write(struct.pack(dreal, d['head'][1][0][0][7]))
		a = f.write(struct.pack(dreal, d['head'][1][0][0][8]))
		a = f.write(struct.pack(dreal, d['head'][1][0][0][9]))
		fill = 256-4*6-8*6-8*2-4*6-4-8*4-4*3
#		fill = 256-8*6-8*6-8*2-8*6-4-8*4-4*3
		a = [f.write(struct.pack(inte, 0)) for x in range(int(fill/4))]
		a = f.write(struct.pack(inte, 256))
		index = []
		for i in range(6):
			if d['head'][1][0][0][4][i]!=0:
				index.append(i)
				if d['head'][1][0][0][1][i]==0: not_withmass += 1
		a = f.write(struct.pack(inte, sum(d['head'][1][0][0][4])*3))
		for i in range(len(index)):
			for k in range(d['head'][1][0][0][4][index[i]]):
				a = [f.write(struct.pack(real, d['l'][0][i][x][k])) for x in range(3)]
		a = f.write(struct.pack(inte, sum(d['head'][1][0][0][4])*3))
		a = f.write(struct.pack(inte, sum(d['head'][1][0][0][4])*3))
		for i in range(len(index)):
			for k in range(d['head'][1][0][0][4][index[i]]):
				a = [f.write(struct.pack(real, d['l'][0][i][3+x][k])) for x in range(3)]
		a = f.write(struct.pack(inte, sum(d['head'][1][0][0][4])*3))
		a = f.write(struct.pack(inte, sum(d['head'][1][0][0][4])))
		for i in range(len(index)):
			for k in range(d['head'][1][0][0][4][index[i]]):
				a = f.write(struct.pack(inte, d['l'][0][i][6][k]))
#				a = f.write(struct.pack(linte, d['l'][0][i][6][k]))
		a = f.write(struct.pack(inte, sum(d['head'][1][0][0][4])))
		if not_withmass>0:
			a = f.write(struct.pack(inte, sum(d['head'][1][0][0][4])))
			for k in range(len(index)):
				if d['head'][1][0][0][1][index[i]]!=0:
					a = [f.write(struct.pack(real, d['head'][1][0][0][1][index[i]])) for x in range(d['head'][1][0][0][4][i])]
				else:
					a = [f.write(struct.pack(real, d['l'][0][i][7][x])) for x in range(d['head'][1][0][0][4][i])]
			a = f.write(struct.pack(inte, sum(d['head'][1][0][0][4])))
		
		if d['head'][1][0][0][4][0]!=0:
			a = f.write(struct.pack(inte, d['head'][1][0][0][4][0]))
			a = [f.write(struct.pack(real, d['l'][0][0][8][x])) for x in range(d['head'][1][0][0][4][0])] # energy
			a = f.write(struct.pack(inte, d['head'][1][0][0][4][0]))
			if IC==0:
				a = f.write(struct.pack(inte, d['head'][1][0][0][4][0]))
				a = [f.write(struct.pack(real, d['l'][0][0][9][x])) for x in range(d['head'][1][0][0][4][0])] # density
				a = f.write(struct.pack(inte, d['head'][1][0][0][4][0]))
				#a = f.write(struct.pack(inte, d['head'][1][0][0][4][0]))
				#a = [f.write(struct.pack(real, d['l'][0][0][10][x])) for x in range(d['head'][1][0][0][4][0])] # ionization
				#a = f.write(struct.pack(inte, d['head'][1][0][0][4][0]))
					
def estimation(res, radius = 10.0, fac = 200.0, a = 1.05, b = 2.1, boxsize=BOXSIZE):
	nhigh = res**3*fac*4*np.pi*(a*radius)**3/3.0/boxsize**3
	nmiddle = (res/2)**3*4*np.pi*((b*radius)**3-(a*radius)**3)/3.0/boxsize**3
	nlow = (res/4)**3*(boxsize**3-4*np.pi*(b*radius)**3/3.0)/boxsize**3
	l = [nhigh, nmiddle, nlow]
	out = l
	out.append(sum(l)**(1/3))
	return out

def projection(ra, dec, vx, vy, vz):
	phi = ra*np.pi/180.0
	theta = dec*np.pi/180.0
	nx = np.cos(theta)*np.cos(phi)
	ny = np.cos(theta)*np.sin(phi)
	nz = np.sin(theta)
	vin = nx*vx+ny*vy+nz*vz
	return -vin



def prof(draw, dh, sq, label, scale = 1, fac1 = 1/10**2, fac2 = 3.0, nbin = 0.06, bs = 0, tola = 1):
	lpart = draw['l'][0]
	info = dh['info'][sq]
	d0 = profile(lpart[0], info, -1, scale, fac1, fac2, nbin, bs, tola)
	v0 = profile(lpart[0], info, -2, scale, fac1, fac2, nbin, bs, tola)
	d1 = profile(lpart[1], info, -1, scale, fac1, fac2, nbin, bs, tola)
	v1 = profile(lpart[1], info, -2, scale, fac1, fac2, nbin, bs, tola)
	t0 = profile(lpart[0], info, 11, scale, fac1, fac2, nbin, bs, tola)
	data = [d1[0][0], d1[1][0], v1[1][0], v1[1][1], v1[1][2], v1[1][3], v1[1][4], d0[1][0], v0[1][0], v0[1][1], v0[1][2], v0[1][3], v0[1][4], t0[1][0]]
	error = [d1[0][1], d1[1][1], v1[1][5], v1[1][6], v1[1][7], v1[1][8], v1[1][9], d0[1][1], v0[1][5], v0[1][6], v0[1][7], v0[1][8], v0[1][9], t0[1][1]]
	head1 = ['r', 'DM_dens', 'DM_vr', 'DM_sigma1', 'DM_sigma2', 'DM_sigma3', 'DM_beta', 'gas_dens', 'gas_vr', 'gas_sigma1', 'gas_sigma2', 'gas_sigma3', 'gas_beta', 'T']
	head2 = [str(info[x]) for x in range(6)]+[str(info[6][x]) for x in range(3)]+[str(info[7][x]) for x in range(3)]+[str(info[8])]
	txt.totxt('profile/pro_'+label+str(sq), data, head1, 1, 0)
	txt.totxt('profile/std_'+label+str(sq), error, head2, 1, 0)
	return [data, error]
		
def prof_sky(lpart, cen, index = 0, a = 1, fac0 = 1/10**2, fac2 = 10.0, bsize = 0.1, bs = 0, tola = 1, dr = 10.0, blos = 40):
	loc = np.array(cen)-np.array(Shift)+np.array(AVEP)
	print(loc)
	loc0 = loc - np.array(Earth)	
	loc1 = np.dot(rot_matrix3d(THETA,PHI), loc0)
	r0 = np.linalg.norm(loc1)
	dec0 = math.asin(loc1[2]/r0)
	if loc1[1]>=0:
		ra0 =  math.acos(loc1[0]/(loc1[0]**2+loc1[1]**2)**0.5)
	else:
		ra0 = - math.acos(loc1[0]/(loc1[0]**2+loc1[1]**2)**0.5)
	print(ra0, dec0)
	m0 = len(lpart)
	num = len(lpart[0])
	cata = [lpart[x] for x in range(m0)] + [np.zeros(len(lpart[0]), dtype='float') for x in range(3)]
	for i in range(num):
		dis = np.array([lpart[x][i] for x in range(3)])
		r = np.linalg.norm(dis)
		dec = math.asin(dis[2]/r)
		if dis[1]>=0:
			ra =  math.acos(dis[0]/(dis[0]**2+dis[1]**2)**0.5)
		else:
			ra = - math.acos(dis[0]/(dis[0]**2+dis[1]**2)**0.5)
		cata[m0][i] = dec
		cata[m0+1][i] = ((ra-ra0)**2+(dec-dec0)**2)**0.5*180*60/np.pi
		cata[m0+2][i] = r
	min_dis = np.min(cata[m0+1])
	if min_dis>=fac0:
		print(min_dis)
		fac1 = 1.5*min_dis
	else:
		fac1 = fac0
	cata = np.array(np.matrix(cata).transpose())
	cata = sorted(cata, key=lambda cata:cata[m0+1])
	nbin = int(fac2/bsize+0.5)
	if bs==0:
		axis1 = 10**np.linspace(np.log10(fac1), np.log10(fac2), nbin)
		axis2 = [0]+[axis1[x] for x in range(nbin)]
	else:
		axis2 = np.linspace(0, fac2, nbin+1)
	data = [np.zeros(nbin, dtype='float') for x in range(2)]
	buf = [[] for x in range(nbin)]
	label = 0
	for j in range(len(cata)):
		if cata[j][m0+1]<axis2[label+1]:
			buf[label].append(cata[j])
		else:
			label += 1
		if label>nbin-1:
			break
	for i in range(nbin):
		count = len(buf[i])
		if count<tola:
			print("Error: The bin is too small.",i)
			return []
		if index==0:
			part = np.array(np.matrix(buf[i]).transpose())
			area = 2*np.pi*(UL*NORM*np.pi/60/180)**2*(axis2[i+1]**2-axis2[i]**2)
			HB = Bremsstrahlung(part[9], part[11], part[7]/part[9])/np.cos(part[m0])
			data[0][i] = np.sum(HB)/area/(2*np.pi)/(180*60/np.pi)**2
			data[1][i] = data[0][i]/np.sqrt(count)
		elif index==1:
			part = np.array(np.matrix(buf[i]).transpose())
			data[0][i] = np.average(part[11])*BOL/10**7/(1.6/10**19)/1000.0
			data[1][i] = data[0][i]/np.sqrt(count)
			#if count>1:
			#	data[1][i] = np.std(part[11])
			#else:
			#	data[1][i] = part[11][0]*BOL/10**7/(1.6/10**19)/1000.0
		else:
			part = buf[i]
			part = sorted(part, key=lambda part:part[m0+2])
			ddr = dr/blos
			axis3 = np.linspace(NORM-dr/2+ddr/2,NORM+dr/2-ddr/2,blos)
			line = [[] for x in range(blos)]
			to_int = np.zeros(blos, dtype='float')
			sq = 0
			for k in range(count):
				if (part[k][m0+2]<axis3[sq]+ddr/2)and(part[k][m0+2]>=axis3[sq]-ddr/2):
					line[sq].append(part[k])
				else:
					if part[k][m0+2]>=axis3[sq]+ddr/2:
						sq += 1
				if sq==blos:
					break
			for s in range(blos):
				if len(line[s])>0:
					to_int[s] = np.average([SZfac(line[s][x][9],line[s][x][11]) for x in range(len(line[s]))])
			data[0][i] = -2*np.sum(to_int)*UL*a*ddr*2.7315*10**6
			data[1][i] = abs(data[0][i]/np.sqrt(count))
	axis4 = np.array([(axis2[x+1]+axis2[x])/2.0 for x in range(nbin)])
	xerr = np.array([(axis2[x+1]-axis2[x])/2.0 for x in range(nbin)])
	return 	[[axis4, xerr], data]
		

def profile(lpart, info, index = -1, a = 1, fac0 = 1/10**2, fac2 = 3.0, bsize = 0.1, bs = 0, tola = 1):
	rv = info[4]
	M = 10**(info[0]-10)
	#num = int(M/lpart[7][0])+1
	num = len(lpart[0])
	velc = np.array(info[7])
	print(rv)
	nbin = int(a*rv*fac2/bsize+0.5)
	print(nbin)
	centre = np.array([info[1+x] for x in range(3)])

	ref = np.zeros(len(lpart[0]), dtype='float')
	for k in range(len(lpart[0])):
		dis = np.array([lpart[x][k] for x in range(3)]) - centre
		ref[k]=np.dot(dis, dis)**0.5
	cata = [lpart[x] for x in range(len(lpart))] + [ref]
	cata = np.array(np.matrix(cata).transpose())
	cata = sorted(cata, key=lambda cata:cata[len(lpart)])
	min_dis = np.min(ref)
	if min_dis>=fac0*rv:
		print(min_dis/rv)
		fac1 = 1.5*min_dis/rv
	else:
		fac1 = fac0
	if bs==0:
		axis1 = 10**np.linspace(np.log10(fac1*rv), np.log10(fac2*rv), nbin)
		#print(axis1)
		axis2 = [0]+[axis1[x] for x in range(nbin)]
	else:
		axis2 = np.linspace(0, fac2*rv, nbin+1)
	if index==-2:
		buf = [[np.zeros(num, dtype='float'), np.zeros(num, dtype='float'), np.zeros(num, dtype='float'), 0] for i in range(nbin)]
		data = [np.zeros(nbin, dtype='float') for x in range(10)]	
	else:
		buf = [[np.zeros(num, dtype='float'), 0] for x in range(nbin)]
		data = [np.zeros(nbin, dtype='float') for x in range(2)]
	
	label = 0
	for j in range(len(cata)):
		if index==-1:
			V = (a**3)*4*np.pi*(axis2[label+1]**3-axis2[label]**3)/3.0
			if cata[j][len(lpart)]<axis2[label+1]:
				buf[label][0][buf[label][1]] = cata[j][7]/V
				buf[label][1] += 1
			else:
				label += 1
		elif index==-2:
			if cata[j][len(lpart)]<axis2[label+1]:
				dis = np.array([cata[j][x] for x in range(3)]) - centre
				if np.linalg.norm(dis)>=min_dis:
					vel = np.array([cata[j][x+3] for x in range(3)])-velc
					dis = dis/np.linalg.norm(dis)
					#print(dis)
					if dis[2]>1:
						dis[2]=1
					elif dis[2]<-1:
						dis[2]=-1
					theta = np.arccos(dis[2])
					if theta==np.nan:
						print(j)
						continue
					if dis[1]>=0:
						phi =  np.arccos(dis[0]/np.sin(theta))
					else:
						phi = 2*np.pi-np.arccos(dis[0]/np.sin(theta))
					n = []
					n.append(np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)]))
					n.append(np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), -np.sin(theta)]))
					n.append(np.array([-np.sin(phi), np.cos(phi), 0]))
					#if (np.dot(n[0],vel)!=np.nan)and(np.dot(n[1],vel)!=np.nan)and (np.dot(n[2],vel)!=np.nan):
					for l in range(3):
						buf[label][l][buf[label][3]] = np.dot(n[l],vel)
					buf[label][3] += 1
					#else:
						#print(label)
			else:
				label += 1
		else:
			if cata[j][len(lpart)]<axis2[label+1]:
				buf[label][0][buf[label][1]] = cata[j][index]
				buf[label][1] += 1
			else:
				label += 1
		if label>nbin-1:
			break
	if index==-2:
		for i in range(nbin):
			if buf[i][3]<tola:
				print("Error: The bin is too small.",i)
				return []
			data[0][i] = np.nanmean([buf[i][0][x] for x in range(buf[i][3])])
			if buf[i][3]>1:
				data[1][i] = np.nanstd([buf[i][0][x] for x in range(buf[i][3])])
				data[2][i] = np.nanstd([buf[i][1][x] for x in range(buf[i][3])])
				data[3][i] = np.nanstd([buf[i][2][x] for x in range(buf[i][3])])
			else:
				data[1][i] = abs(buf[i][0][0])
				data[2][i] = abs(buf[i][1][0])
				data[3][i] = abs(buf[i][2][0])
			data[4][i] = 1-(data[2][i]**2+data[3][i]**2)/data[1][i]**2/2.0
			#data[4][i] = ((data[0][i]/np.sqrt(buf[i][2]))**2+data[1][i]**2)**0.5
			data[5][i] = abs(data[0][i])/np.sqrt(buf[i][3])
			data[6][i] = abs(data[1][i])/np.sqrt(buf[i][3])
			data[7][i] = abs(data[2][i])/np.sqrt(buf[i][3])
			data[8][i] = abs(data[3][i])/np.sqrt(buf[i][3])
			data[9][i] = abs((1-data[4][i]))*np.sqrt(12/buf[i][3])
	else:
		for i in range(nbin):
			if buf[i][1]<tola:
				print("Error: The bin is too small.",i)
				return []
			if index==-1:
				data[0][i] = np.sum([buf[i][0][x] for x in range(buf[i][1])])
				data[1][i] = data[0][i]/np.sqrt(buf[i][1])
			elif index==9:
				volume = np.array([cata[0][7] for x in range(buf[i][1])])/np.array([buf[i][0][x] for x in range(buf[i][1])])
				data[0][i] = cata[0][7]*buf[i][1]/np.sum(volume)
				data[1][i] = data[0][i]*((np.nanstd([buf[i][0][x] for x in range(buf[i][1])])/np.nanmean([buf[i][0][x] for x in range(buf[i][1])]))**2+1/buf[i][1])**0.5
			else:
				data[0][i] = np.nanmean([buf[i][0][x] for x in range(buf[i][1])])
				data[1][i] = ((data[0][i]/np.sqrt(buf[i][1]))**2+np.nanstd([buf[i][0][x] for x in range(buf[i][1])])**2)**0.5
	axis3 = np.array([(axis2[x+1]+axis2[x])/2.0/rv for x in range(nbin)])
	xerr = np.array([(axis2[x+1]-axis2[x])/2.0/rv for x in range(nbin)])
	return 	[[axis3, xerr], data]
					
def Virial(M, z = 0, t = 0, fac = 200.0):
	m = 10**(M-10)
	H = HZ(z)
	if t==0:
		rho = 3*H**2/8/np.pi/G
		r = ((1+z)**4*m*3/fac/4/np.pi/rho)**(1/3)
	else:
		rho = 3*H**2/8/np.pi/G
		r = ((1+z)**3*m*3/fac/4/np.pi/rho)**(1/3)
	return r

def V200(part, cen, r0, m0, z, C = 3, tola = 0.01):
	m = 10**(m0-10)
	part.append(np.zeros(len(part[0]), dtype='float'))
	for i in range(len(part[0])):
		dis = np.array([part[y][i]-cen[y] for y in range(3)])
		part[len(part)-1][i] = np.linalg.norm(dis)
	ind = len(part)-1
	part = np.array(np.matrix(part).transpose())
	part = sorted(part, key=lambda part:part[ind])
	def error(r):
		count = 0
		M = 0
		while part[count][ind]<=r:
			count += 1
			M += part[count][7]
			if count>len(part):
				break
		return M-200*CRITICAL(z)*(4*np.pi*(r/(1+z))**3/3)*(OMEGA0-OMEGA_B)/OMEGA0
	rv = r0*C
	er = error(r0)
	count = 0
	while abs(er)>=tola*m:
		if er>0:
			rv += tola*r0
		else:
			rv = rv - tola*r0
		er = error(rv)
		count += 1
		#print(rv)
		if count>=C/tola:
			break
	mv = 200*CRITICAL(z)*(4*np.pi*(rv/(1+z))**3/3)
	return [rv, mv]


def NFW(x, c0, z):
	H = HZ(z)
	c = c0*(1+z)**(1/3)
	delta = 200.0*c0**3/3/(np.log(1+c0)-c0/(1+c0))
	rho = (3*H**2/8/np.pi/G)*delta/(c*x*(1+c*x)**2)/(1+z)**3
	return rho

def forf(z, refx, refy, style=1, x0 = 1, start = 0.05, end = 1.0, fac=10**4):
	def ft(c0):
		return fitting(c0, z, refx, refy, style, start, end, fac)
	#x_min = brent(ft)
	x_min = fmin_bfgs(ft, x0)[0]
	print(x_min)
	return [x_min, np.array([NFW(x, x_min, z) for x in refx])]

def fitting(c0, z, refx, refy, style=1, start = 0.05, end = 1.0, fac=10**4):
	out = 0
	index = [i for i in range(len(refx)) if (refx[i]>=start)and(refx[i]<=end)]
	slope = (np.log10(refy[index[0]])-np.log10(refy[index[len(index)-1]]))/(np.log10(refx[index[0]])-np.log10(refx[index[len(index)-1]]))
	def refrho(x):
		out = np.log10(refy[index[0]])+slope*(np.log10(x)-np.log10(refx[index[0]]))
		return 10**out
	for x in index:
		if style==0:
			out += (NFW(refx[x],c0,z)/fac-refy[x]/fac)**2
		elif style==1:
			out += abs(np.log10(NFW(refx[x],c0,z))-np.log10(refy[x]))**2
		elif style==2:
			out += (NFW(refx[x],c0,z)/refrho(refx[x])-refy[x]/refrho(refx[x]))**2
	return out

def pickrand(raw, fac):
	num = int(fac*len(raw[0]))
	seq = [random.random() for x in range(len(raw[0]))]
	cata = []
	for i in range(len(raw)):
		cata.append(raw[i])
	cata.append(seq)
	a = np.array(np.matrix(cata).transpose())
	a = sorted(a, key=lambda a:a[len(raw)])
	b = []
	for j in range(num):
		b.append(a[j])
	out = np.array(np.matrix(b).transpose())
	return out

def energy1(pos, vel, ref, fac, a = 1):
	mass = ref[7][0]
	ki = 0.5*np.dot(vel, vel)*mass
	#ki = 0
	gra = 0.0
	for i in range(len(ref[0])):
		dis = pos-np.array([ref[x][i] for x in range(3)])
		r = a*np.dot(dis, dis)**0.5
		if r!=0:
			gra += -G*ref[7][i]/r
	return ki+gra*fac

def energy2(pos, ref, a = 1):
	mass = ref[7][0]
	gra = 0.0
	for i in range(len(ref[0])):
		dis = pos-np.array([ref[x][i] for x in range(3)])
		r = a*np.dot(dis, dis)**0.5
		if r!=0:
			gra += -G*ref[7][i]/r
	return gra

def find_centre(d, scale = 1, fac1 = 0.05, fac2 = 0.2, lim = 1000, start = 5, run = 20, count = 10, style = 0, index = 1, shi = np.array([AVEP[x] - Coma[x] for x in range(3)])):
	out = []
	for i in range(len(d['part'])):
		rv = Virial(d['info'][i][2], 1/scale-1)
		cen = np.array([d['info'][i][3+x] for x in range(3)])+shi
		velc = (np.array([np.nanmean(d['part'][i][1][3+x]) for x in range(3)])*(OMEGA0-OMEGA_B)*len(d['part'][i][1][0])+ np.array([np.nanmean(d['part'][i][0][3+x]) for x in range(3)])*OMEGA_B*len(d['part'][i][0][0]))/((OMEGA0-OMEGA_B)*len(d['part'][i][1][0])+OMEGA_B*len(d['part'][i][0][0]))
		print(d['info'][i][2], cen, velc)
		store = []
		fac = min(fac2, lim/len(d['part'][0][index][0]))
		#ref = pickrand(d['part'][i][index], fac)
	# initialization
		ref0 = pickrand(d['part'][i][index], fac)
		offset = 0 
		label = 0
		for s in range(len(ref0[0])):
			pos = np.array([ref0[0][s], ref0[1][s], ref0[2][s]])
			if style==0:
				vel = np.array([ref0[3+x][s] for x in range(3)])-velc
				temp = energy1(pos, vel, ref0, 1.0/fac, scale)
			else:
				temp = energy2(pos, ref0, scale)/fac
			if temp<offset:
				offset = temp
				label = s
		cen = np.array([ref0[x][label] for x in range(3)])
		print(cen)
		refof = 0
		for j in range(run):
			ref = pickrand(d['part'][i][index], fac)
			num = 0
			cen00 = cen.copy()
			while num<count:
				#ref = pickrand(d['part'][i][index], fac)
				offset = 0
				label = 0
				cen0 = cen.copy()
				EBAR = 0.0
				npart = 0
				for k in range(len(d['part'][i][index][0])):
					x = d['part'][i][index][0][k]
					y = d['part'][i][index][1][k]
					z = d['part'][i][index][2][k]
					pos = np.array([x, y, z])
					if np.dot(pos-cen, pos-cen)**0.5<=rv*fac1:
						if style==0:
							vel = np.array([d['part'][i][index][3+x][k] for x in range(3)])-velc
							temp = energy1(pos, vel, ref, 1.0/fac, scale)
						else:
							temp = energy2(pos, ref, scale)/fac
						npart += 1
						EBAR += temp
						if temp<offset:
							offset = temp
							label = k
				cen = np.array([d['part'][i][index][x][label] for x in range(3)])
				#print(cen,num,j,i)
				dis = cen-cen0
				if np.dot(dis,dis)==0:
					EBAR = abs(EBAR/npart)
					if EBAR==0:
						EBAR = abs(refof)
					if j>=start:
						judge = min(1, np.exp((refof-offset)/EBAR))
						#print(judge)
						rad = random.random()
						if rad>judge:
							cen = cen00.copy()
						store.append(cen)
					break
				num += 1
			if num==count:
				print ("Insufficient number of steps for run", j, "of halo", i)
			print(cen,num,j,i)
		store = np.array(np.matrix(store).transpose())
		final = np.array([np.nanmean(store[x]) for x in range(3)])
		error = np.array([np.nanstd(store[x]) for x in range(3)])
		out.append([d['info'][i][2],final[0], final[1], final[2],rv,d['info'][i][6]/d['info'][i][7],error,velc,len(store)/float(run-start)])
	d0 = {}
	d0['part'] = d['part']
	d0['info'] = out
	return d0

def overall2(d0, draw, scale = 1, C = 3.0, upper = 3.0):
	d = d0.copy()
	out = [[] for x in range(len(d['part']))]
	for i in range(len(d['part'])):
		rv = d['info'][i][4]
		cen = np.array([d['info'][i][1+x] for x in range(3)])
		velc = d['info'][i][7]
		I0 = np.zeros(3, dtype='float')
		I1 = np.zeros(3, dtype='float')
		dis = 0
		for j in range(len(d['part'][i][0][0])):
			pos = np.array([d['part'][i][0][x][j] for x in range(3)])-cen
			vel = np.array([d['part'][i][0][3+x][j] for x in range(3)])-velc
			I0 += scale*np.cross(pos,vel)
			if np.linalg.norm(pos)>dis:
				dis = np.linalg.norm(pos)
		I0 = I0/len(d['part'][i][0][0])
		for j in range(len(d['part'][i][1][0])):
			pos = np.array([d['part'][i][1][x][j] for x in range(3)])-cen
			vel = np.array([d['part'][i][1][3+x][j] for x in range(3)])-velc
			I1 += scale*np.cross(pos,vel)
			if np.linalg.norm(pos)>dis:
				dis = np.linalg.norm(pos)
		I1 = I1/len(d['part'][i][1][0])
		gas = draw['l'][0][0]
		#lim = max(dis, C*rv)
		lim = upper*rv
		if dis>C*rv:
			print (dis/rv)
		L = 0.0
		temperature = np.zeros(len(gas[0]),dtype='float')
		count = 0
		for k in range(len(gas[0])):
			pos = np.array([gas[x][k] for x in range(3)])-cen
			if np.linalg.norm(pos)<=lim:
				L += 2*Bremsstrahlung(gas[9][k],gas[11][k],gas[7][k]/gas[9][k])
				temperature[count] = gas[11][k]
				count += 1
		T = np.nanmean([temperature[i] for i in range(count)])
		#Sigma = L/4/np.pi/scale**2/lim**2/UL**2
		
		#I0_ = np.dot(rot_matrix3d(THETA,PHI), I0)
		#I1_ = np.dot(rot_matrix3d(THETA,PHI), I1)
		angle = np.arccos(np.dot(I0,I1)/np.linalg.norm(I0)/np.linalg.norm(I1))*180/np.pi
		out[i].append(L)
		#out[i].append(Sigma)
		out[i].append(d['info'][i][5])
		out[i].append(I0[0])
		out[i].append(I0[1])
		out[i].append(I0[2])
		out[i].append(I1[0])
		out[i].append(I1[1])
		out[i].append(I1[2])
		out[i].append(angle)
		out[i].append(cen[0]-250)
		out[i].append(cen[1]-250)
		out[i].append(cen[2]-250)
		out[i].append(T)
		#out[i].append(I0_[0])
		#out[i].append(I0_[1])
		#out[i].append(I0_[2])
		#out[i].append(I1_[0])
		#out[i].append(I1_[1])
		#out[i].append(I1_[2])
	return out

def vel(d0, dx = 10, dy = 16, dz = 16):
	d=d0.copy()
	part=d['l'][0].copy()
	vc = [(np.average(part[0][3+x])*len(part[0][3+x])*OMEGA_B+np.average(part[1][3+x])*len(part[1][3+x])*(OMEGA0-OMEGA_B))/(len(part[0][3+x])*OMEGA_B+len(part[1][3+x])*(OMEGA0-OMEGA_B)) for x in range(3)]
	HB = np.sum(Bremsstrahlung(part[0][9],part[0][11],part[0][7]/part[0][9]))
	m0 = np.sum(part[0][7])
	m1 = np.sum(part[1][7])
	mref = CRITICAL(0)*OMEGA0*dx*dy*dz
	print (vc)
	for x in range(3):
		part[0][3+x] = part[0][3+x]-vc[x]
		part[1][3+x] = part[1][3+x]-vc[x]
	vx0 = np.average(-np.sign(part[0][0])*part[0][3])
	vy0 = np.average(-np.sign(part[0][1])*part[0][4])
	vz0 = np.average(-np.sign(part[0][2])*part[0][5])
	vx1 = np.average(-np.sign(part[1][0])*part[1][3])
	vy1 = np.average(-np.sign(part[1][1])*part[1][4])
	vz1 = np.average(-np.sign(part[1][2])*part[1][5])
	vx0d = np.std(-np.sign(part[0][0])*part[0][3])
	vy0d = np.std(-np.sign(part[0][1])*part[0][4])
	vz0d = np.std(-np.sign(part[0][2])*part[0][5])
	vx1d = np.std(-np.sign(part[1][0])*part[1][3])
	vy1d = np.std(-np.sign(part[1][1])*part[1][4])
	vz1d = np.std(-np.sign(part[1][2])*part[1][5])
	vin0 = np.average(-(part[0][0]*part[0][3]+part[0][1]*part[0][4]+part[0][2]*part[0][5])/(part[0][0]**2+part[0][1]**2+part[0][2]**2)**0.5)
	vin1 = np.average(-(part[1][0]*part[1][3]+part[1][1]*part[1][4]+part[1][2]*part[1][5])/(part[1][0]**2+part[1][1]**2+part[1][2]**2)**0.5)
	vin0d = np.std(-(part[0][0]*part[0][3]+part[0][1]*part[0][4]+part[0][2]*part[0][5])/(part[0][0]**2+part[0][1]**2+part[0][2]**2)**0.5)
	vin1d = np.std(-(part[1][0]*part[1][3]+part[1][1]*part[1][4]+part[1][2]*part[1][5])/(part[1][0]**2+part[1][1]**2+part[1][2]**2)**0.5)
	Tem = np.average(part[0][11])
	Temd = np.std(part[0][11])
	return [HB/10**44, m0, m1, (m1+m0)/mref]+[vin0, vx0, vy0, vz0, vin1, vx1, vy1, vz1, np.log10(Tem), vx0d, vy0d, vz0d, vin0d, vx1d, vy1d, vz1d, vin1d, Temd]+vc
	

def fof_cata(fname, ed = '<'):
	inte = ed+'i'
	real = ed+'f'
	linte = ed+'q'
	dreal = ed+'d'
	out = {}
	data = []
	with open(fname, 'rb') as f:
		nhalo = struct.unpack(inte, f.read(4))[0]
		for k in range(2):
			data.append(np.zeros(nhalo, dtype='int'))
			for i in range(nhalo):
				data[k][i] = struct.unpack(inte, f.read(4))[0]
		for k in range(4):
			data.append(np.zeros(nhalo, dtype='float'))
		for i in range(nhalo):
			data[2][i] = np.log10(struct.unpack(real, f.read(4))[0])+10
		for i in range(nhalo):
			for r in range(3):
				data[3+r][i] = struct.unpack(real, f.read(4))[0]
		for k in range(3):
			data.append(np.zeros(nhalo, dtype='int'))
		for i in range(nhalo):
			for r in range(3):
				data[6+r][i] = struct.unpack(inte, f.read(4))[0]
	out['nhalo'] = nhalo
	out['l'] = data
	return out

def fof_index(fname, ed = '<'):
	inte = ed+'i'
	out = {}
	with open(fname, 'rb') as f:
		out['npart'] = struct.unpack(inte, f.read(4))[0]
		out['l'] = np.zeros(out['npart'], dtype='int')
		for i in range(out['npart']):
			out['l'][i] = struct.unpack(inte, f.read(4))[0]
	return out	

def mergetree(d1, d2, style = 0):
	nhalo1 = len(d1['info'])
	nhalo2 = len(d2['info'])
	out0, out1 = [np.zeros(nhalo2, dtype='float') for x in range(nhalo1)], [np.zeros(nhalo2, dtype='float') for x in range(nhalo1)]
	ref = [np.zeros(sum([len(d1['part'][x][0][0])+len(d1['part'][x][1][0]) for x in range(nhalo1)])) for y in range(3)]
	tar = [np.zeros(sum([len(d2['part'][x][0][0])+len(d2['part'][x][1][0]) for x in range(nhalo2)])) for y in range(3)]
	#print (ref, tar)
	count = 0
	for i in range(nhalo1):
		for j in range(len(d1['part'][i][0][0])):
			ref[0][count] = d1['part'][i][0][6][j]
			ref[1][count] = i
			ref[2][count] = 0
			count += 1
		for k in range(len(d1['part'][i][1][0])):
			ref[0][count] = d1['part'][i][1][6][k]
			ref[1][count] = i
			ref[2][count] = 1
			count += 1
	ref = np.array(np.matrix(ref).transpose())
	ref = sorted(ref, key=lambda ref:ref[0])
	count = 0
	for i in range(nhalo2):
		for j in range(len(d2['part'][i][0][0])):
			tar[0][count] = d2['part'][i][0][6][j]
			tar[1][count] = i
			tar[2][count] = 0
			count += 1
		for k in range(len(d2['part'][i][1][0])):
			tar[0][count] = d2['part'][i][1][6][k]
			tar[1][count] = i
			tar[2][count] = 1
			count += 1
	tar = np.array(np.matrix(tar).transpose())
	tar = sorted(tar, key=lambda tar:tar[0])
	count = 0
	for i in range(len(tar)):
		while tar[i][0]>ref[count][0]:
				count+=1
		if ref[count][0]==tar[i][0]:
			if tar[i][2]==0:
				out0[int(ref[count][1]+0.5)][int(tar[i][1]+0.5)]+=1
			else:
				out1[int(ref[count][1]+0.5)][int(tar[i][1]+0.5)]+=1
		if count==len(ref):
			break
	for i in range(nhalo1):
		for j in range(nhalo2):
			if style==0:
				out0[i][j] = out0[i][j]/float(d1['info'][i][6])
				out1[i][j] = out1[i][j]/float(d1['info'][i][7])
			else:
				out0[i][j] = out0[i][j]/float(d2['info'][j][6])
				out1[i][j] = out1[i][j]/float(d2['info'][j][7])
	return [out0, out1]

def collect(d, cataf, indexf):
	start = time.time()
	nhalo = cataf['nhalo']
	npart = len(indexf['l'])
	out = [[[] for y in range(len(d))] for x in range(nhalo)]
	raw = [np.array(np.matrix(d[k]).transpose()) for k in range(len(d))]
	label = -1*np.ones(npart, dtype='int')
	num = [0 for x in range(nhalo)]
	ref = []
	for i in range(npart):
		for j in range(nhalo):
			if (i>=cataf['l'][1][j]) and (i<cataf['l'][1][j]+cataf['l'][0][j]):
				label[i]=j
				#num[j] += 1
			#else:
			#	label[i]=-1
		ref.append((indexf['l'][i], label[i]))
	ref = sorted(ref, key=lambda ref:ref[0])
	print('Construction of the index list: done', time.time()-start, 's')
	#return num
	data = []
	for k in range(len(d)):
		for i in range(len(d[k][0])):
			data.append(raw[k][i])
	offset = 0
	judge = []
	for k in range(len(d)):
		judge.append(offset)
		offset += len(d[k][0])
	print(judge)
	data = sorted(data, key=lambda data:data[6])
	#for j in range(nhalo):
	#	for i in range(len(ref)):
	#		if ref[i][1]==j:
	#			a = 1
	#			num[j] += 1
	#return [ref, num]
	count=0
	for i in range(len(data)):
		if data[i][6]==ref[count][0]:
			for k in range(len(d)):
				if (data[i][6]>=judge[k]) and (data[i][6]<judge[k]+len(d[k][0])):
					if ref[count][1]!=-1:
						out[ref[count][1]][k].append(data[i])
					break
			count +=1
		if count==npart:
			break
	print('Selection: done', time.time()-start,'s', count, i)
	for i in range(nhalo):
		for k in range(len(d)):
			out[i][k] = np.array(np.matrix(out[i][k]).transpose())
	d0 = {}
	d0['part'] = out
	d0['info'] = np.array(np.matrix(cataf['l']).transpose())
	return d0
					
def overall1(d0, scale = 1, C = 1.5, shi = np.array([AVEP[x] - Coma[x] for x in range(3)])):
	d = d0.copy()
	out = [[] for x in range(len(d['part']))]
	for i in range(len(d['part'])):
		rv = Virial(d['info'][i][2], 1/scale-1)*scale
		cen = np.array([d['info'][i][3+x] for x in range(3)])+shi-np.array([250.0,250.0,250.0])
		velc = (np.array([np.nanmean(d['part'][i][1][3+x]) for x in range(3)])*(OMEGA0-OMEGA_B)*len(d['part'][i][1][0])+ np.array([np.nanmean(d['part'][i][0][3+x]) for x in range(3)])*OMEGA_B*len(d['part'][i][0][0]))/((OMEGA0-OMEGA_B)*len(d['part'][i][1][0])+OMEGA_B*len(d['part'][i][0][0]))
		out[i].append(d['info'][i][2])
		out[i].append(rv)
		out[i].append(cen[0])
		out[i].append(cen[1])
		out[i].append(cen[2])
		out[i].append(velc[0])
		out[i].append(velc[1])
		out[i].append(velc[2])
		#print(d['info'][i][2], rv, cen, velc)
	return out
		
def find_halo(tar, std = [5.0, 10.0,10.0,10.0], ref=[Coma[3]]+[BOXSIZE/2.0 for x in range(3)]):
	a = np.copy(tar['l'])
	out = []
	nhalo = tar['nhalo']
	b = np.array(np.matrix(a).transpose())
	count = 0
	for i in range(nhalo):
		t = 0
		for k in range(4):
			if abs(b[i][2+k]-ref[k])<=std[k]:
				t += 1
		if t==4:
			count += 1
			out.append(b[i])
	c = np.array(out)
	#c = sorted(c, key=-lambda c:c[2])
	d = {}
	d['nhalo'] = count
	d['l'] = np.array(np.matrix(c).transpose())
	return d
		
def cutbox(d0, obs = Coma, box = (10.0, 40.0, 40.0), boxsize=BOXSIZE):
	d = d0.copy()
	head = d['head'][0]
	out = []
	lpara = []
	for i in range(len(d['l'])):
		out.append([])
		for k in range(len(d['l'][i])):
			out[i].append([])
			index = []
			for j in range(len(d['l'][i][k][0])):
				t = 0
				if (periodic_pos(d['l'][i][k][0][j],obs[0],boxsize)>obs[0]-box[0]/2.0)and(periodic_pos(d['l'][i][k][0][j],obs[0],boxsize)<obs[0]+box[0]/2.0):
					t += 1
				if (periodic_pos(d['l'][i][k][1][j],obs[1],boxsize)>obs[1]-box[1]/2.0)and(periodic_pos(d['l'][i][k][1][j],obs[1],boxsize)<obs[1]+box[1]/2.0):
					t += 1
				if (periodic_pos(d['l'][i][k][2][j],obs[2],boxsize)>obs[2]-box[2]/2.0)and(periodic_pos(d['l'][i][k][2][j],obs[2],boxsize)<obs[2]+box[2]/2.0):
					t += 1
				if t==3:
					index.append(j)
			for m in range(len(d['l'][i][k])):
				out[i][k].append(np.zeros(len(index), dtype=d['l'][i][k][m].dtype))
			for n in range(len(index)):
				for s in range(len(d['l'][i][k])):
					if s<=2:
						out[i][k][s][n] = periodic_pos(d['l'][i][k][s][index[n]],obs[s],boxsize)-obs[s]
					else:
						out[i][k][s][n] = d['l'][i][k][s][index[n]]
		lpara.append([])
		lpara[i].append([])
		lpara[i][0]=d['head'][1][i][0]
		count = 0
		for r in range(6):
			if d['head'][1][i][0][4][r]!=0: 
				lpara[i][0][0][r]=len(out[i][count][0])
				lpara[i][0][4][r]=len(out[i][count][0])
				count += 1
		lpara[i][0][6]=box[0]
	dout = {}
	dout['head'] = [head, lpara]
	dout['l'] = out
	return dout

def cutsphere(d0, obs = Coma, radius = [0.0, 50.0], sky = [[-90.0,90.0], [-180.0,180.0]], boxsize=BOXSIZE):
	d = d0.copy()
	head = d['head'][0]
	out = []
	lpara = []
	for i in range(len(d['l'])):
		out.append([])
		for k in range(len(d['l'][i])):
			out[i].append([])
			index = []
			for j in range(len(d['l'][i][k][0])):
				dis = np.array([(periodic_pos(d['l'][i][k][x][j],obs[x],boxsize)-obs[x]) for x in range(3)])
				mod = np.dot(dis,dis)**0.5
				t = 0
				dec = math.asin(dis[2]/mod)*180.0/math.pi
				if dis[1]>=0:
					ra =  math.acos(dis[0]/(dis[0]**2+dis[1]**2)**0.5)*180.0/math.pi
				else:
					ra = - math.acos(dis[0]/(dis[0]**2+dis[1]**2)**0.5)*180.0/math.pi
				if (mod<=radius[1]) and (mod>=radius[0]):
					t += 1
				if (dec<=sky[0][1]) and (dec>=sky[0][0]):
					t += 1
				if (ra<=sky[1][1]) and (ra>=sky[1][0]):
					t += 1
				if t==3:
					index.append(j)
			for m in range(len(d['l'][i][k])):
				out[i][k].append(np.zeros(len(index), dtype=d['l'][i][k][m].dtype))
			for n in range(len(index)):
				for s in range(len(d['l'][i][k])):
					if s<=2:
						out[i][k][s][n] = periodic_pos(d['l'][i][k][s][index[n]],obs[s],boxsize)-obs[s]
					else:
						out[i][k][s][n] = d['l'][i][k][s][index[n]]
		lpara.append([])
		lpara[i].append([])
		lpara[i][0]=d['head'][1][i][0]
		count = 0
		for r in range(6):
			if d['head'][1][i][0][4][r]!=0: 
				lpara[i][0][0][r]=len(out[i][count][0])
				lpara[i][0][4][r]=len(out[i][count][0])
				count += 1
		lpara[i][0][6]=radius*2
	dout = {}
	dout['head'] = [head, lpara]
	dout['l'] = out
	return dout

def periodic_pos(a, b, c):
	if a-b>c/2.0:
		return a-c
	elif a-b<-c/2.0:
		return a+c
	else:
		return a

def periodic(a, b):
	if a-b>0:
		return a-b
	elif a<0:
		return a+b
	else:
		return a

def rot_matrix3d(theta=0.0, phi=39.0):
	out = np.zeros((3,3), dtype='float')
	out[0][0] = np.cos((theta/180)*np.pi)*np.cos((phi/180)*np.pi)
	out[0][1] = np.cos((theta/180)*np.pi)*np.sin((phi/180)*np.pi)
	out[0][2] = np.sin((theta/180)*np.pi)
	out[1][0] = -np.sin((phi/180)*np.pi)
	out[1][1] = np.cos((phi/180)*np.pi)
	out[2][0] = -np.sin((theta/180)*np.pi)*np.cos((phi/180)*np.pi)
	out[2][1] = -np.sin((theta/180)*np.pi)*np.sin((phi/180)*np.pi)
	out[2][2] = np.cos((theta/180)*np.pi)
	return out

def LoS1(d0, avep = AVEP, box = (10.0, 16.0, 16.0), phi = PHI, theta = THETA):
	d = d0.copy()
	d1 = shift(d, np.array(avep)-np.array([BOXSIZE/2.0,BOXSIZE/2.0,BOXSIZE/2.0]))
#	d1 = shift(cutsphere(d, (BOXSIZE/2.0,BOXSIZE/2.0,BOXSIZE/2.0), [0.0, NORM]), np.array(avep))
	d2 = rotate(d1, theta, phi, Earth)
	d3 = cutbox(d2, np.array(Earth)+np.array([NORM,0.0,0.0]), box)
	for i in range(len(d3['l'])):
		for k in range(len(d3['l'][i])):
			d3['l'][i][k][1] *= -1
			d3['l'][i][k][4] *= -1
	return d3

def LoS3(d0, avep = AVEP, region = (10.0,12.0,12.0), phi = PHI, theta = THETA):
	d = d0.copy()
	d1 = shift(d, np.array(avep)-np.array([BOXSIZE/2.0,BOXSIZE/2.0,BOXSIZE/2.0]))
#	d1 = shift(cutsphere(d, (BOXSIZE/2.0,BOXSIZE/2.0,BOXSIZE/2.0), [0.0, NORM]), np.array(avep))
	d2 = cutsphere(d1, np.array(Earth), [NORM-region[0]/2.0, NORM+region[0]/2.0], [[-region[1]/2.0+theta, region[1]/2.0+theta], [-region[2]/2.0+phi,phi+region[2]/2.0]])
	return d2

def LoS2(d0, avep = AVEP, region = (10.0, 30.0, 30.0), phi = PHI, theta = THETA):
	d = d0.copy()
#	d1 = shift(d, np.array(avep)-np.array([BOXSIZE/2.0,BOXSIZE/2.0,BOXSIZE/2.0]))
	d1 = shift(cutsphere(d, (BOXSIZE/2.0,BOXSIZE/2.0,BOXSIZE/2.0), [0.0, NORM]), np.array(avep))
	d2 = rotate(d1, theta, phi, Earth)
	d3 = cutsphere(d2, np.array(Earth), [NORM-region[0]/2.0, NORM+region[0]/2.0], [[-region[1]/2.0,region[1]/2.0], [-region[2]/2.0,region[2]/2.0]])
	#for i in range(len(d3['l'])):
	#	for k in range(len(d3['l'][i])):
	#		d3['l'][i][k][1] *= -1
	#		d3['l'][i][k][4] *= -1
	return d3

def shift(d0, avep = Shift, boxsize = BOXSIZE):
	d = d0.copy()
	head = d['head'][0]
	out = []
	lpara = d['head'][1]
	for i in range(len(d['l'])):
		out.append([])
		for k in range(len(d['l'][i])):
			out[i].append([])
			for m in range(len(d['l'][i][k])):
				out[i][k].append(np.zeros(len(d['l'][i][k][0]), dtype=d['l'][i][k][m].dtype))
			for n in range(len(d['l'][i][k][0])):
				for s in range(len(d['l'][i][k])):
					if s<=2:
						out[i][k][s][n] = periodic(d['l'][i][k][s][n]+avep[s],boxsize)
					else:
						out[i][k][s][n] = d['l'][i][k][s][n]
	dout = {}
	dout['head'] = [head, lpara]
	dout['l'] = out
	return dout

def rotate(d0, theta=0.0, phi=39.0, obs=Earth):
	d = d0.copy()
	head = d['head'][0]
	out = []
	lpara = np.copy(d['head'][1])
	for i in range(len(d['l'])):
		out.append([])
		for k in range(len(d['l'][i])):
			out[i].append([])
			pos = np.zeros((len(d['l'][i][k][0]),3), dtype='float')
			vel = np.zeros((len(d['l'][i][k][0]),3), dtype='float')
			for m in range(len(d['l'][i][k])):
				out[i][k].append(np.zeros(len(d['l'][i][k][0]), dtype=d['l'][i][k][m].dtype))
			for j in range(len(d['l'][i][k][0])):
				pos0 = np.array([d['l'][i][k][x][j]-obs[x] for x in range(3)], dtype='float')
				vel0 = np.array([d['l'][i][k][3+x][j] for x in range(3)])
				pos[j] = np.dot(rot_matrix3d(theta,phi), pos0)
				vel[j] = np.dot(rot_matrix3d(theta,phi), vel0)
			for n in range(len(d['l'][i][k][0])):
				for s in range(len(d['l'][i][k])):
					if s<=2:
						out[i][k][s][n] = pos[n][s] + obs[s]
					elif (2<s) and (s<=5):
						out[i][k][s][n] = vel[n][s-3]
					else:
						out[i][k][s][n] = d['l'][i][k][s][n]
	dout = {}
	dout['head'] = [head, lpara]
	dout['l'] = out
	return dout

def addgas(d0, res = 256.0, Omega_b = OMEGA_B, boxsize=BOXSIZE):
	d = d0.copy()
	head = d['head'][0]
	out = []
	lpara = d['head'][1]
	dis = 3**0.5*boxsize/res/2
	ratio = Omega_b/d['head'][1][0][0][7]
	for i in range(len(d['l'])):
		out.append([])
		out[i].append([])
		out[i].append([])
		for s in range(len(d['l'][i][0])):
			out[i][0].append(np.zeros(len(d['l'][i][0][0]), dtype=d['l'][i][0][s].dtype))
			out[i][1].append(np.zeros(len(d['l'][i][0][0]), dtype=d['l'][i][0][s].dtype))
		out[i][0].append(np.ones(len(d['l'][i][0][0]), dtype='float'))
		for m in range(len(d['l'][i][0][0])):
			r1, r2 = random.random(), random.random()
			split = dis*np.array([np.sin(r1*np.pi)*np.cos(r2*2*np.pi),np.sin(r1*np.pi)*np.sin(r2*2*np.pi),np.cos(r1*np.pi)])
			for n in range(3):
				out[i][0][n][m]=d['l'][i][0][n][m] + split[n]*(1-ratio)
				out[i][1][n][m]=d['l'][i][0][n][m] - split[n]*ratio
				out[i][0][n+3][m]=d['l'][i][0][n+3][m]
				out[i][1][n+3][m]=d['l'][i][0][n+3][m]
			out[i][0][6][m] = m
			out[i][1][6][m] = d['l'][i][0][6][m]+len(d['l'][i][0][0])
			out[i][0][7][m] = d['head'][1][0][0][1][1]*ratio
			out[i][1][7][m] = d['head'][1][0][0][1][1]*(1-ratio)
		for k in range(len(d['l'][i])-1):
			out[i].append([])
			out[i][k+2] = d['l'][i][k+1]
			out[i][k+2][6] = out[i][k+2][6]+len(d['l'][i][0][0])
		lpara.append([])
		lpara[i].append([])
		lpara[i][0]=d['head'][1][i][0]
		lpara[i][0][0][0] = d['head'][1][i][0][4][1]
		lpara[i][0][4][0] = d['head'][1][i][0][4][1]
		lpara[i][0][1][0] = d['head'][1][0][0][1][1]*ratio
		lpara[i][0][1][1] = d['head'][1][0][0][1][1]*(1-ratio)
	dout = {}
	dout['head'] = [head, lpara]
	dout['l'] = out
	return dout
		
def mindis(d, ref = (BOXSIZE/2.0, BOXSIZE/2.0, BOXSIZE/2.0), boxsize=BOXSIZE):
	Rmin = boxsize
	pos = ref
	for i in range(len(d[0])):
		dis = np.array([periodic_pos(d[m][i],ref[m],boxsize)-ref[m] for m in range(3)])
		norm = np.dot(dis,dis)**0.5
		if norm<Rmin:
			Rmin = norm
			pos = dis
	return [Rmin, pos]

def sort(l, index = 7):
	print('Begin sorting')
	dim = len(l)
	if index>dim:
		print ('Error')
		return l
	a = np.matrix(l)
	b = np.array(a.transpose())
	b = sorted(b, key=lambda b:b[index])
	c = np.array(np.matrix(b).transpose())
	print('Done')
	return c
#	lis[k].append(l[dim][k])


# Codes below will not be used.
def read_fof_old(fname, mass = MASS, fn = 1, ed = '<'):
	inte = ed+'i'
	real = ed+'f'
	linte = ed+'q'
	dreal = ed+'d'
	out = {}
	lpara, data = [], []
	for fi in range(fn):
		lpara.append([])
		data.append([])
		if fn!=1:
			fname0 = fname+'.'+str(fi)
		else:
			fname0 = fname

		with open(fname0, 'rb') as f:
			nhalo = struct.unpack(inte, f.read(4))[0]
			nids = struct.unpack(inte, f.read(4))[0]
			totngroups = struct.unpack(inte, f.read(4))[0]
			nfile = struct.unpack(inte, f.read(4))[0]
			#if nfile!=fn:
			#	print('Error: incorrect file number')
			#	return {}
			lpara[fi]=[nhalo,nids,totngroups,nfile]
			data[fi]=[np.zeros(nhalo, dtype='float') for x in range(5)]
			data[fi][4] = np.zeros(nhalo, dtype='int')
			for i in range(nhalo):
				data[fi][4][i] = struct.unpack(inte, f.read(4))[0]
			data[fi][3] = np.log10(mass*data[fi][4])
			for i in range(nhalo):
				a = f.read(4)
			for i in range(nhalo):
				for k in range(3):
					data[fi][k][i] = struct.unpack(real, f.read(4))[0]
	out['head'] = lpara
	output = [np.hstack([data[fi][x] for fi in range(fn)]) for x in range(5)]
	out['l'] = output
	return out

def read_highres(basename = 'snapshot', sn = 4, fn = 1, IC = 0, st = 0, so = -1, res = 1, tem = 1):
	output = []
	para = []
	for i in range(st, fn):
		buf = []
		if IC==0:
			for j in range(sn):
				if sn!=1:
					l = load(basename+'_'+str(i).zfill(3)+'.'+str(j),1,IC,so,4,tem)
				else:
					l = load(basename+'_'+str(i).zfill(3),1,IC,so,4,tem)
				if l[0][0][4][0]==0:
					del l[1][1+res]
				else:
					del l[1][2+res]
				buf.append(l) 
				#buf.append([[l[1][m][s] for s in range(8)]])
			if buf[0][0][0][4][0]==0:
				output.append([[np.hstack([buf[x][1][m][s] for x in range(sn)]) for s in range(8)] for m in range(1+res)])
			else:
				output.append([[np.hstack([buf[x][1][0][s] for x in range(sn)]) for s in range(11+tem)]] + [[np.hstack([buf[x][1][m+1][s] for x in range(sn)]) for s in range(8)] for m in range(1+res)])
		else:
			for j in range(sn):
				if sn!=1:
					l = load(basename+'.'+str(j),1,IC,so,4,tem)
				else:
					l = load(basename,1,IC,so,4,tem)
				if l[0][0][4][0]==0:
					del l[1][1+res]
				else:
					del l[1][2+res]
				buf.append(l)
			if buf[0][0][0][4][0]==0:
				output.append([[np.hstack([buf[x][1][m][s] for x in range(sn)]) for s in range(8)] for m in range(2)])
			else:
				output.append([[np.hstack([buf[x][1][0][s] for x in range(sn)]) for s in range(10+tem)]] + [[np.hstack([buf[x][1][m+1][s] for x in range(sn)]) for s in range(8)] for m in range(1+res)])

		para.append([buf[x][0][0] for x in range(sn)])
		for k in range(2-res):
			for x in range(sn):
				para[i-st][x][4][2+tem-k], para[i-st][x][0][2+tem-k] = 0, 0
	d = {}
	d['head'] = [head, para]
	d['l'] = output
	return d  


