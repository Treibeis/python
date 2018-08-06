import numpy as np
#from scipy.interpolate import *
#from txt import *
import math
import time
import csv
#import random
import struct 
import matplotlib.pyplot as plt
from interp import *

mp_r = 129391565095.19
mp_l = 146268717741.28
NORM = 72.144749111360142

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

def sample(cata, mag_0 = 20.5, step = 0.01, n_max = 4*10**4):
	out = cata
	m = mag_0
	while len(out[0])>n_max:
		m -= step
		raw = [[], [], [], [], []]
		for i in range(len(out[0])):
			if out[3][i]<=m:
				raw[0].append(out[0][i])
				raw[1].append(out[1][i])
				raw[2].append(out[2][i])
				raw[3].append(out[3][i])
				raw[4].append(out[4][i])
		out = raw
	print ('mag_r_cut:', m)
	return out	

def cut(d, xr = [100.0,140.0], yr = [0.0,40.0], zr = [0.0,0.4]):
	out = [[], [], []]
	for i in range(len(d[0])):
		t1, t2, t3 = 0, 0, 0
		x, y, z = d[0][i], d[1][i], d[2][i]
		if ((xr[0]<=x)and(x<=xr[1])):
			t1 = 1
		if ((yr[0]<=y)and(y<=yr[1])):
			t2 = 1
		if ((zr[0]<=z)and(z<=zr[1])):
			t3 = 1
		if ((t1+t2+t3))==3:
			out[0].append(d[0][i])
			out[1].append(d[1][i])
			out[2].append(d[2][i])
	print (len(out[0]))
	return out


def redeuss(s2 = 'rpcdmw5', s1 = 'fof_cone2592_n2048', num = 512, s3 = 'masst'):
	out = [[], [], [], [], []]
	for i in range(num):
		s4 = str(i).zfill(5)
		s = s1+'_'+s2+'_'+s3+'_'+s4
		o = ri(s)
		out[0] += o[0]
		out[1] += o[1]
		out[2] += o[2]
		out[3] += o[3]
		out[4] += o[4]
	print (len(out[0]))
	return out

def ri(s, mp = 1, box = [2592.0, 2592.0, 2592.0], ed = '>'):
	out = [[], [], [], [], []]
	inte = ed+'i'
	real = ed+'f'
	linte = ed+'q'
	with open(s, 'rb') as f:
		a = f.read(4)
		num = struct.unpack(inte,f.read(4))[0]
		print (num)
		a = f.read(4)
		for i in range(num):
			a = f.read(4)
			ID = struct.unpack(linte,f.read(8))[0]
			npart = struct.unpack(inte, f.read(4))[0]*mp
			x = struct.unpack(real, f.read(4))[0]*box[0]
			y = struct.unpack(real, f.read(4))[0]*box[1]
			z = struct.unpack(real, f.read(4))[0]*box[2]
			a = f.read(4)
			out[0].append(x)
			out[1].append(y)
			out[2].append(z)
			out[3].append(npart)
			out[4].append(ID)
	return out

def recsv(s, n, k = 1, t = 1):
	out = []
	j = 0
	for i in range(n):
		out.append([])
	with open(s, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			if j<k:
				print (row)
			else:
				for i in range(n):
					out[i].append(float(row[i]))
			j = j+1
	if t!=0:
		for i in range(n):
			out[i].reverse()
			out[i] = np.array(out[i])
	return np.array(out)

def retxt(s, n, k = 1, t = 1):
	out = []
	for i in range(n):
		out.append([])
	j = 0
	with open(s, 'r') as f:
		for line in f:
			lst = line.split()
			if j<k:
				print (lst[0])
			else:
				for i in range(n):
					out[i].append(float(lst[i]))
			j = j+1
	if t!=0:
		for i in range(n):
			out[i].reverse()
			#out[i] = np.array(out[i])
	return np.array(out)

def totxt(s, l, ls, t = 0, k = 0):
	j = 0
	with open(s, 'w') as f:
		if t!=0:
			for r in range(len(ls)):
				f.write(ls[r])
				f.write(' ')
			f.write('\n')
		for i in range(len(l[0])):
			if j<k:
				print (l[0][i])
			else:
				for s in range(len(l)):
					f.write(str(l[s][i]))
					f.write(' ')
				f.write('\n')
			j = j+1

# The codes above are meant for file I/O.

RATIO = 0.6884*math.sin(70.0*math.pi/180.0)/4.0
box_size = 500.0


D_Z_L = np.array(txt.retxt('/home/friede/python/d_z_L.txt', 2, 0, 0))
D_Z_L = np.array([np.array(D_Z_L[0])*Unit1, np.array(D_Z_L[1])*Unit2])
H_Z_L = txt.retxt('/home/friede/python/H_z_L.txt', 2, 0, 0)
D_Z_R = txt.retxt('/home/friede/python/d_z_R.txt', 2, 0, 0)
H_Z_R = txt.retxt('/home/friede/python/H_z_R.txt', 2, 0, 0)

def generate(out = 'a1.txt', ra = [100.0,280.0], dec = [0.0,70.0], b1 = 1.0, b2 = 1.0, num = 100.0):
	n1, n2 = int((ra[1]-ra[0])/b1)+1, int((dec[1]-dec[0])/b2)+1
	lra, ldec = np.linspace(ra[0], ra[1], n1), np.linspace(dec[0], dec[1], n2)
	dec_cord = [(ldec[i+1]+ldec[i])/2.0 for i in range(n2-1)]
	#grid = np.zero((n1-1,n2-1))
	s0 = b1*b2*(math.pi/180.0)**2
	cata = [[], [], [], [], []]
	count = 0
	for i in range(n1-1):
		for j in range(n2-1):
			s = b1*(math.sin(ldec[j+1]*math.pi/180.0)-math.sin(ldec[j]*math.pi/180.0))*math.pi/180.0
			n = int(num*s/s0)
			for k in range(n):
				count += 1
				r1, r2 = np.random.random(), np.random.random()
				cata[0].append(lra[i]+(lra[i+1]-lra[i])*r1)
				cata[1].append(ldec[j]+(ldec[j+1]-ldec[j])*r2)
				cata[2].append(count)
				cata[3].append(0)
				cata[4].append(0)
	totxt(out, cata, 0, 0 ,0)
	return cata

def num_dis(z):
	return (1.55-1.165*z)/10**4

def sky(cata, area = [[0.0, 360.0], [-90.0, 90.0]], lz = [0.2,0.6]):
	out =  []
	raw = np.matrix(cata).transpose()
	for i in range(len(cata[0])):
		s = 0
		if ((cata[0][i]<=area[0][1])and(cata[0][i]>=area[0][0])):
			s = s+1
		if ((cata[1][i]<=area[1][1])and(cata[1][i]>=area[1][0])):
			s = s+1
		if ((cata[2][i]>=lz[0])and(cata[2][i]<=lz[1])):
			s = s+1
		if (s==3):
			out.append(raw)
	out = np.array(np.matrix(raw).transpose())
	return out

def Atracer(nd = num_dis, d_z = D_Z_L, zr = [0.2,0.6], nub = 40):
	z_bin = (zr[1]-zr[0])/nub
	nub += 1
	edge = np.linspace(zr[0], zr[1], nub)
	z_cord = [(edge[i+1]+edge[i])/2.0 for i in range(nub-1)]
	density = [nd(z_cord[i]) for i in range(nub-1)]
	dz = interp1d(d_z[1],d_z[0])#,kind='cubic')
	ratio = RATIO
	volume = [ratio*(dz(edge[i+1])**3-dz(edge[i])**3)*4.0*math.pi/3.0 for i in range(nub-1)]
	his = [int((volume[i]*density[i])) for i in range(nub-1)]
	bat = [[edge[i] for i in range(len(edge)-1)], [edge[i+1] for i in range(len(edge)-1)], density, his, volume]
	return [his, edge, z_cord, density, bat]

def poisson(l):
	out = l
	n = len(l)
	num = len(l[0])
	for i in range(n-1):
		up, down = [], []
		for j in range(num):
			up.append(l[i+1][j]+abs(l[i+1][j])**0.5)
			down.append(l[i+1][j]-abs(l[i+1][j])**0.5)
		out.append(up)
		out.append(down)
	return out

def numfuc(cata, rr = [0.0, 70.0], nub = 70):
	nub = nub+1
	his, edge = np.histogram(cata[3], np.linspace(rr[0], rr[1], nub))
	r_cord = [(edge[i+1]+edge[i])/2.0 for i in range(nub-1)]
	r_cord.reverse()
	his = his[::-1]
	hisa = np.cumsum(his)
	return [r_cord, his, hisa]

def tracer(cata, d_z = D_Z_L, zr = [0.2,0.6], nub = 40):
	z_bin = (zr[1]-zr[0])/nub
	nub += 1
	his, edge = np.histogram(cata[2], np.linspace(zr[0], zr[1], nub))
	z_cord = [(edge[i+1]+edge[i])/2.0 for i in range(nub-1)]
	dz = interp1d(d_z[1],d_z[0])#,kind='cubic')
	ratio = RATIO
	volume = [ratio*(dz(edge[i+1])**3-dz(edge[i])**3)*4.0*math.pi/3.0 for i in range(nub-1)]
	density = [his[i]/volume[i] for i in range(nub-1)]
	return [his, edge, z_cord, density]

def modify(cata, ref, dm = 10, t = 0):
	edge = ref[1]
	his = ref[0]
	raw0 = np.array(np.matrix(cata).transpose())
	count = 0
	num0 = len(cata[0])
	for k in range(num0):
		if cata[2][k]>=ref[1][0]:
			raw.append(raw0[k])
			count += 1
	raw = sorted(raw, key=lambda raw:raw[2])
	nub = len(ref[0])
	out = []
	j = 0
	num = count
	for i in range(nub):
		sub = []
		while ((j<num)and(raw[j][2]<=edge[i+1])):
			mod = (raw[j][3]+dm*np.random.normal(), raw[j])
			sub.append(mod)
			j += 1
		if len(sub)<his[i]:
			print ('There are too many galaxies in the redshift range:', edge[i], edge[i+1])
			return []
		sub = sorted(sub, key=lambda sub:sub[0])
		if t==0:
			sub.reverse()
		if sub!=[]:
			for k in range(his[i]):
				out.append(sub[k][1])
	out = np.array(np.matrix(out).transpose())
	return out

def select(cata, cataf, zr = [0.2, 0.6], ra = 1.0/10**5, dec = 1.0/10**5, z = 1.0/10**5):
	raw = np.array(np.matrix(cata).transpose())
	raw = sorted(raw, key=lambda raw:raw[2])
	ref = np.array(np.matrix(cataf).transpose())
	ref = sorted(ref, key=lambda ref:ref[2])
	num1 = len(raw)
	num2 = len(ref)
	print('n1, n2 =', num1, num2)
	print('Sampled and sorted: done')	
	out = []
	k = 0
	mis = []
	for i in range(num1):
		head = k
		#print(head)
		while (k<num2):	
			t = 0
			if abs(ref[k][2]-raw[i][2])<=z:
				t += 1
			if abs(ref[k][1]-raw[i][1])<=dec:
				t += 1
			if abs(ref[k][0]-raw[i][0])<=ra:
				t += 1
			if t==3:
				out.append(ref[k])
				k = head+1
				break
			if ref[k][2]-raw[i][2]>z:
				print('Not match at z=', raw[i][2])
				mis.append(k)
				k = head
				break
			k += 1
	out = np.array(np.matrix(out).transpose())
	print(len(out[0]))
	return [out, mis]

def batch_mod1(ref, re, s1, n, dm = 10, T = 0, d = 5, s2 = '.txt', k = 0, t = 0, fill = 0, fn = 4):
	for i in range(n):
		if (fill==0):
			s3 = str(i+1)
		else:
			s3 = str(i+1).zfill(fn)
		totxt('modified_'+s1+'_'+s3+s2, modify(retxt(re+s1+'_'+s3+s2, d, k, t), ref, dm, T), 0, 0, 0)
	return 0

def batch_mod2(ref, re, s1, n, dm = 10, T = 0, d = 5, s2 = '.txt', k = 0, t = 0, fill = 0, fn = 4):
	for i in range(n):
		if (fill==0):
			s3 = str(i+1)
		else:
			s3 = str(i+1).zfill(fn)
		totxt('modified_'+s1+'_'+s3+s2, modify(retxt(re+s1+s2, d, k, t), ref, (i+1)*dm, T), 0, 0, 0)
	return 0

			
def rand_cata(num = 10**7, zr = [0.2, 0.6], d_z = D_Z_L, ob = (1800.0, 1800.0, 1800.0), box = (3600.0, 3600.0, 3600.0), sky = [[0.0,360.0], [-90.0,90.0]], ko = 0):
	real = [[], [], [], [], []]
	for i in range(num):
		cord = [box[k]*np.random.random() for k in range(3)]
		real[0].append(cord[0])
		real[1].append(cord[1])
		real[2].append(cord[2])
		real[3].append(500)
		real[4].append(i)
	d = convert2(real, zr, 'rand_cata.txt', 'random', d_z, ob, box, sky)
	d['real'] = real
	return d
		
	
def reconvert(lred, zr = [0.0, ZD(box_size)], fo = 'DR21.txt', label = 'obs', d_z = D_Z, ob = (0.0, 0.0, 0.0), box = (box_size, box_size, box_size), sky = [[0.0,360.0], [-90.0,90.0]], ko = 0):
	start = time.time()
	dz = interp1d(d_z[1],d_z[0])#,kind='cubic')
	#zd = interp1d(d_z[0],d_z[1])#,kind='cubic')
	#Hz = interp1d(H_z[1],H_z[0])#,kind='cubic')
	print ('Interpolation: done', time.time()-start, 's')
	dmax = dz(zr[1])
	if dmax>min([-ob[i]+box[i] for i in range(3)]+[ob[j] for j in range(3)]):
		print ('The box is too small.')
		return {}
	raw = np.array(np.matrix(lred).transpose())
	dim = len(lred)
	for i in range(len(lred[0])):
		if raw[i][2]>=0:
			r = dz(raw[i][2])
			dec = raw[i][1]*math.pi/180.0
			ra = raw[i][0]*math.pi/180.0
			raw[i][0] = r*math.cos(dec)*math.cos(ra)+ob[0]
			raw[i][1] = r*math.cos(dec)*math.sin(ra)+ob[1]
			raw[i][2] = r*math.sin(dec)+ob[2]
	real = []
	for i in range(len(raw)):
		t1, t2, t3 = 0, 0, 0
		if (0<=raw[i][0]) and (raw[i][0]<=box[0]):
			t1 = 1
		if (0<=raw[i][1]) and (raw[i][1]<=box[1]):
			t2 = 1
		if (0<=raw[i][2]) and (raw[i][2]<=box[2]):
			t3 = 1
		if (t1+t2+t3==3):
			real.append(raw[i])
	a = np.matrix(real)
	output = np.array(a.transpose())
	#output = [[], [], [], [], []]
	#for i in range(len(real)):
	#	output[0].append(real[i][0])
	#	output[1].append(real[i][1])
	#	output[2].append(real[i][2])
	#	output[3].append(real[i][3])
	#	output[4].append(real[i][4])
	print ('Sampling: done', time.time()-start, 's')
	#totxt(fo, output, 0, 0, ko)
	d = {}
	d['output'], d['ob'], d['box'], d['label'], d['num'], d['l'] = real, ob, box, label, len(real), output
	print ('All done', time.time()-start, 's')
	return d

# The code below is to convert real boxes to lightcones.

def convert2(lreal, zr = [0.0,ZD(box_size)], fo = 'out1.txt', label = 'Lambda', d_z = D_Z, ob = (0.0,0.0,0.0), box = (box_size,box_size,box_size), sky = [[0.0,360.0],[-90.0,90.0]], ko = 0):
	start = time.time()
	dz = interp1d(d_z[1],d_z[0])#,kind='cubic')
	zd = interp1d(d_z[0],d_z[1])#,kind='cubic')
	#Hz = interp1d(H_z[1],H_z[0])#,kind='cubic')
	print ('Readin and interpolation: done', time.time()-start, 's')
	dmax = dz(zr[1])
	real = np.array(np.matrix(lreal).transpose())
	#edge_x = dmax+ob[0]
	#edge_y = dmax+ob[1]
	print ('Construction of the catalogue in real space: done', time.time()-start, 's')
	#for i in range(len(lreal[0])):
		#if real[i][0]>edge_x:
			#displace = (real[i][0]-box[0], real[i][1], real[i][2], real[i][3], real[i][4], real[i][5])
			#real.append(displace)
		#elif real[i][1]>edge_y:
			#displace = (real[i][0], real[i][1]-box[1], real[i][2], real[i][3], real[i][4], real[i][5])
			#real.append(displace)
	#print ('Arrangement of peioridical boundary condition: done', time.time()-start, 's') 
	for i in range(len(real)):
		x, y, z = real[i][0]-ob[0], real[i][1]-ob[1], real[i][2]-ob[2]
		r = (x**2+y**2+z**2)**0.5
		#if (r>dz(zr[0])) and (r<dz(zr[1])):
		z_real = zd(r)
		dec = math.asin(z/r)*180.0/math.pi
		if y>=0:
			ra =  math.acos(x/(x**2+y**2)**0.5)*180.0/math.pi
		else:
			ra = 360.0 - math.acos(x/(x**2+y**2)**0.5)*180.0/math.pi
		real[i][0], real[i][1], real[i][2] = ra, dec, z_real
	print ('Generatation of raw output: done', time.time()-start, 's')
	redshift = []
	for i in range(len(real)):
		t1, t2, t3 = 0, 0, 0
		if (sky[0][0]<real[i][0]) and (sky[0][1]>real[i][0]):
			t1 = 1
		if (sky[1][0]<real[i][1]) and (sky[1][1]>real[i][1]):
			t2 = 1
		if (zr[0]<real[i][2]) and (zr[1]>real[i][2]):
			t3 = 1
		if (t1+t2+t3==3):
			redshift.append(real[i])
	redshift = sorted(redshift, key=lambda redshift:redshift[2])
	a = np.matrix(redshift).transpose()
	output = np.array(a)
	print ('Sampling: done', time.time()-start, 's')
	#totxt(fo, output, 0, 0, 0)
	d = {}
	d['output'], d['zr'], d['sky'], d['label'], d['num'], d['l'] = redshift, zr, sky, label, len(redshift), output
	print ('All done', time.time()-start, 's')
	return d

def convert1(lreal, zr = [0.0,ZD(box_size)], fo = 'out1.txt', label = 'R-P', d_z = D_Z, H_z = H_Z, ob = (0.0, 0.0, 0.0), box = (box_size,box_size,box_size), sky = [[0.0,360.0],[-90.0,90.0]], ko = 0):
	"""zr is the redshift range. \
	lreal is the list (table) of the real space data. \
	fo is the name of the output file. \
	d_z and H_z are lists of comoving distance/Hubble parameter and redshift \
	ob is the position of the observer \
	box is the size of the simulation box \
	sky is the area in the sky, whose default value corresponds to \
	0<dec<90, 0<ra<360 (unit: degree) \
	the first three columns of lreal are the coordinates in Mpc h^-1\
	while the following three are the velocities in km s^-1 \
	the last column is the halo mass
	ra, dec, z, mass correspond to the 1st, 3rd, 2nd and 4th column of the output file \
	the output dec and ra are in degree"""
	start = time.time()
	dz = interp1d(d_z[1],d_z[0])#,kind='cubic')
	zd = interp1d(d_z[0],d_z[1])#,kind='cubic')
	Hz = interp1d(H_z[1],H_z[0])#,kind='cubic')
	print ('Readin and interpolation: done', time.time()-start, 's')
	dmax = dz(zr[1])
	real = np.array(np.matrix(lreal).transpose())
	#edge_x = dmax+ob[0]
	#edge_y = dmax+ob[1]
	print ('Construction of the catalogue in real space: done', time.time()-start, 's')
	#for i in range(len(lreal[0])):
		#if real[i][0]>edge_x:
			#displace = (real[i][0]-box[0], real[i][1], real[i][2], real[i][3], real[i][4], real[i][5])
			#real.append(displace)
		#elif real[i][1]>edge_y:
			#displace = (real[i][0], real[i][1]-box[1], real[i][2], real[i][3], real[i][4], real[i][5])
			#real.append(displace)
	#print ('Arrangement of peioridical boundary condition: done', time.time()-start, 's') 
	for i in range(len(real)):
		x, y, z = real[i][0]-ob[0], real[i][1]-ob[1], real[i][2]-ob[2]
		r = (x**2+y**2+z**2)**0.5
		#if (r>dz(zr[0])) and (r<dz(zr[1])):
		z_real = zd(r)
		vx, vy ,vz = real[i][3], real[i][4], real[i][5]
		vlos = (vx*x+vy*y+vz*z)/r
		s = r + vlos*(1+z_real)/Hz(z_real)
			#if (s>dz(zr[0])) and (s<dz(zr[1])):
		z_dis = zd(s)
		dec = math.asin(z/r)*180.0/math.pi
		if y>=0:
			ra = math.acos(x/(x**2+y**2)**0.5)*180.0/math.pi
		else:
			ra = 360.0 - math.acos(x/(x**2+y**2)**0.5)*180.0/math.pi
		real[i][0], real[i][1], real[i][2] = ra, dec, z_dis
	print ('Generatation of raw output: done', time.time()-start, 's')
	redshift = []
	for i in range(len(real)):
		t1, t2, t3 = 0, 0, 0
		if (sky[0][0]<real[i][0]) and (sky[0][1]>real[i][0]):
			t1 = 1
		if (sky[1][0]<real[i][1]) and (sky[1][1]>real[i][1]):
			t2 = 1
		if (zr[0]<real[i][2]) and (zr[1]>real[i][2]):
			t3 = 1
		if (t1+t2+t3==3):
			redshift.append(real[i])
	redshift = sorted(redshift, key=lambda redshift:redshift[2])
	a = np.matrix(redshift).transpose()
	output = np.array(a)
	print ('Sampling: done', time.time()-start, 's')
	#totxt(fo, output, 0, 0, 0)
	d = {}
	d['output'], d['zr'], d['sky'], d['label'], d['num'], d['l'] = redshift, zr, sky, label, len(redshift), output
	print ('All done', time.time()-start, 's')
	return d
		
def combine(d1, d2, delta = [1.0, 1.0, 0.05], fo = 'combine.txt'):
	"""d1, d2 are supposed to be outputs of function convert or combine \
	d2 corresponds to the shell with higher redshift \
	fo is the name of output file \
	ra, dec, z, mass correspond to the 1st, 2nd, 3rd and 4th column of the output file \
	the output dec and ra are in degree"""
	start = time.time()
	overlap = [d2['zr'][0], d1['zr'][1]]
	zr = [d1['zr'][0],d2['zr'][1]]
	raw = d1['output']
	for i in range(len(d2['output'])):
		raw.append(d2['output'][i])
	raw = sorted(raw, key=lambda raw:raw[2])
	redshift = []
	mark = np.zeros(len(raw))
	for i in range(len(raw)-1):
		t1, t2, t3 = 0, 0, 0
		if abs(raw[i][0]-raw[i+1][0])<delta[0]:
			t1 = 1
		if abs(raw[i][1]-raw[i+1][1])<delta[1]:
			t2 = 1
		if abs(raw[i][2]-raw[i+1][2])<delta[2]:
			t3 = 1
		if (t1+t2+t3==3):
			mark[i] = 1
		if mark[i]==0:
			redshift.append(raw[i])
	#print ('Number of halos removed: ', sum(mark))
	redshift.append(raw[len(raw)-1])
	print ('Number of repeat: ',sum(mark))
	output = np.array(np.matrix(redshift).transpose())
	print ('Sampling: done', time.time()-start, 's')
	#totxt(fo, output, 0, 0, 0)
	d = {}
	d['output'], d['zr'], d['sky'], d['label'], d['num'], d['l'] = redshift, zr, d1['sky'], d1['label'], len(redshift), output
	return d




			
