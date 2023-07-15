import random as rd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

def logistic(x0, r, N):
	x = x0
	for i in range(N):
		x = r*x*(1-x)
	return [r, x]

if __name__=="__main__":
	N0 = 1000
	S0 = 100
	lr = np.linspace(0,4.0,1000)
	ncore = 4
	npc = int(S0/ncore)
	manager = mp.Manager()
	output = manager.Queue()
	def sess():
		out = np.vstack([ [logistic(rd.random(), r, N0) for i in range(npc)] for r in lr]).T
		output.put(out)
	processes = [mp.Process(target=sess) for i in range(ncore)]
	for p in processes:
		p.start()
	for p in processes:
		p.join()
	out = np.hstack([output.get() for p in processes])
	plt.figure()
	plt.plot(out[0], out[1], '.', markersize=0.1)
	plt.ylim(0,1)
	plt.xlim(0,4.0)
	plt.ylabel(r'$x_{'+str(N0)+'}$')
	plt.xlabel(r'$r$')
	plt.title(r'$x_{n+1}=rx_{n}(1-x_{n})$')
	plt.tight_layout()
	plt.savefig('logistic.png', dpi=300)
	plt.close()
	#plt.show()
