import numpy as np
import struct
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

# Units convertion

GRA = 6.672e-8
#GRV = 1.0
BOL = 1.3806e-16
#BOL = 1.0
PROTON = 1.6726e-24
#PROTON = 1.6726e-24/(2.176e-5)

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

# Meaningful items in the head for galaxy cluster simulation

head = ['npart[6]', 'mass[6]', 'time', 'redshift', 'Nall[6]', 'num_files', 'BoxSize', 'Omega0', 'OmegaLambda', 'HoubleParam']

name = ['Gas', 'Halo', 'Disk', 'Bulge', 'Stars', 'Bndry']

# Calculate the temperature based on internal energy u, metalicity X and degree of ionization Y for an adiabatic gas.

IONIZATION = 0.0

def temp(u, Y, X = 0.76, gamma = 5.0/3):
	M = PROTON*4.0/(3*X+1+4*X*Y)
	U = u*UE/UM
	tem = M*U*(gamma-1)/BOL
	return tem

# Read snapshot/IC files

def read(basename = 'snapshot', sn = 1, fn = 1, IC = 0, st = 0, so = -1, tem = 0):
	output = []
	para = []
	for i in range(st, fn):
		if IC==0:
			l = load(basename+'_'+str(i).zfill(3),sn,IC,so,tem)
		else:
			l = load(basename,sn,IC,so,tem)
		output.append(l[1])
		para.append(l[0])
	d = {}
	d['head'] = [head, para]
	d['l'] = output
	return d  
# The outputs is a dictionary with 'head' leads to head items and their values at each snapshot (in the list d['head'][1][0] even if there is only one snapshot), 
# while 'l' leads to the list of the results at each snapshot.

# Transfer binary data blocks to numpy arrays 
	
def load(fname, fn = 1, IC = 0, so = -1, tem = 1, ed = '<', dummy = 4):
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
			mass = [struct.unpack(dreal, f.read(8))[0] for x in range(6)]
			time = struct.unpack(dreal, f.read(8))[0]
			Redshift = struct.unpack(dreal, f.read(8))[0]
			print ('Redshift, time = ', Redshift, time)
			flag_sfr = struct.unpack(inte, f.read(4))[0]	
			flag_feedback = struct.unpack(inte, f.read(4))[0]
			Nall = [struct.unpack(inte, f.read(4))[0] for x in range(6)]
			flag_cooling = struct.unpack(inte, f.read(4))[0]
			print ('flag_cooling', flag_cooling)
			num_files = struct.unpack(inte, f.read(4))[0]
			#if num_files!= fn:
			#	print('The number of files at each snapshot should be ',fn)
			#	return [[], []]
			BoxSize = struct.unpack(dreal, f.read(8))[0]
			Omega0 = struct.unpack(dreal, f.read(8))[0]
			OmegaLambda = struct.unpack(dreal, f.read(8))[0]
			HoubleParam = struct.unpack(dreal, f.read(8))[0]
			fill = 256-4*6-8*6-8*2-4*6-4-8*4-4*3
			a = f.read(fill)
			a = f.read(dummy)
			para = [npart, mass, time, Redshift, Nall, num_files, BoxSize, Omega0, OmegaLambda, HoubleParam]
			lpara.append(para)
			# Data structure for individual files
			index = []
			for i in range(6):
				if npart[i]!=0:
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
			a = f.read(dummy)
			# Read velocities
			a = f.read(dummy)
			for k in range(len(index)):
				lv[fi][k] = [np.zeros(npart[index[k]], dtype='float') for x in range(3)]
				for l in range(npart[index[k]]):
					lv[fi][k][0][l] = struct.unpack(real, f.read(4))[0]
					lv[fi][k][1][l] = struct.unpack(real, f.read(4))[0]
					lv[fi][k][2][l] = struct.unpack(real, f.read(4))[0]
			a = f.read(dummy)
			# Read particle IDs
			a = f.read(dummy)
			for k in range(len(index)):
				lID[fi][k] = np.zeros(npart[index[k]], dtype = 'uint')
				for l in range(npart[index[k]]):
					lID[fi][k][l] = struct.unpack(inte, f.read(4))[0]
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
					output[0][11] = np.zeros(Nall[0], dtype='float')
					for l in range(Nall[0]):
						output[k][11][l] = temp(output[k][8][l], output[k][10][l])
			else:
				output[0][9] = np.hstack([lNe[fi][0] for fi in range(fn)])
				if tem==1:
					output[0][10] = np.zeros(Nall[0], dtype='float')
					for l in range(Nall[0]):
						output[k][10][l] = temp(output[k][8][l], output[k][9][l])
		if so>=0:
			output[k] = sort(output[k],so)
		#output[k] = sort(output[k], 6)
	return [lpara, output]
		
def write(d, name = 'test_ics', ed = '<', dummy = 4):
	inte = ed+'i'
	real = ed+'f'
	linte = ed+'q'
	dreal = ed+'d'
	not_withmass = 0
	with open(name, 'wb') as f:
		a = f.write(struct.pack(inte, 256))
		a = [f.write(struct.pack(inte, x)) for x in d['head'][1][0][0][4]]
		a = [f.write(struct.pack(dreal, x)) for x in d['head'][1][0][0][1]]
		a = f.write(struct.pack(dreal, d['head'][1][0][0][2]))
		a = f.write(struct.pack(dreal, d['head'][1][0][0][3]))
		a = f.write(struct.pack(inte, 0))
		a = f.write(struct.pack(inte, 0))
		a = [f.write(struct.pack(inte, x)) for x in d['head'][1][0][0][4]]
		a = f.write(struct.pack(inte, 0))
		a = f.write(struct.pack(inte, 1))
		a = f.write(struct.pack(dreal, d['head'][1][0][0][6]))
		a = f.write(struct.pack(dreal, d['head'][1][0][0][7]))
		a = f.write(struct.pack(dreal, d['head'][1][0][0][8]))
		a = f.write(struct.pack(dreal, d['head'][1][0][0][9]))
		fill = 256-4*6-8*6-8*2-4*6-4-8*4-4*3
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
			a = [f.write(struct.pack(real, d['l'][0][0][8][x])) for x in range(d['head'][1][0][0][4][0])]
			a = f.write(struct.pack(inte, d['head'][1][0][0][4][0]))
					
def cut(d, box = (100.0, 100.0, 100.0), obs = (200.0,200.0,200.0)):
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
				if (d['l'][i][k][0][j]>obs[0])and(d['l'][i][k][0][j]<obs[0]+box[0]):
					t += 1
				if (d['l'][i][k][1][j]>obs[1])and(d['l'][i][k][1][j]<obs[1]+box[1]):
					t += 1
				if (d['l'][i][k][2][j]>obs[2])and(d['l'][i][k][2][j]<obs[2]+box[2]):
					t += 1
				if t==3:
					index.append(j)
			for m in range(len(d['l'][i][k])):
				out[i][k].append(np.zeros(len(index), dtype=d['l'][i][k][m].dtype))
			for n in range(len(index)):
				for s in range(len(d['l'][i][k])):
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
	lis[k].append(l[dim][k])

