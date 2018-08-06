import struct
import numpy as np

def select(d, box = (100.0, 100.0, 100.0), obs = (200.0,200.0,200.0)):
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
		for r in range(6):
			count = 0
			if d['head'][1][i][0][4][r]!=0: 
				lpara[i][0][0][r]=len(out[i][count][0])
				lpara[i][0][4][r]=len(out[i][count][0])
				count += 1
	dout = {}
	dout['head'] = [head, lpara]
	dout['l'] = out
	return dout
		
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
					
				
		
