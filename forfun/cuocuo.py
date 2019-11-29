import numpy as np
import matplotlib.pyplot as plt

top = lambda x: (1-(np.abs(x)-1)**2)**0.5

bottom0 = lambda x: -np.arccos(np.abs(x)-1)

y0 = -1.2
beta = 1.2
slope = -1/np.sin(y0)
x0 = 1 + np.cos(-y0)
a = slope/x0**(beta-1)/beta
b = y0 - a*x0**beta
bottom1 = lambda x: a*np.abs(x)**beta + b
	
def bottom(x):
	b0 = bottom0(x)
	b1 = bottom1(x)
	b = b0 * np.logical_or(x>=x0, x<=-x0) + b1 * np.logical_and(x<x0, x>-x0)
	return b

if __name__=="__main__":
	lx = np.linspace(-2, 2, 521)
	ly1 = top(lx)
	ly2 = bottom(lx)

	asp = 1.0 #3./np.pi

	plt.figure()
	ax = plt.subplot(111)
	ax.set_aspect('equal')
	plt.fill_between(lx, ly1, ly2*asp, facecolor='r', alpha=0.5,
			edgecolor='red')
	#plt.plot(lx, bottom0(lx), '--', color='r', alpha=0.5, label='Before')
	#plt.plot(lx, bottom(lx), '-', color='r', alpha=0.5, label='After')
	#plt.legend(loc=3)
	plt.tick_params(top=False, bottom=False, left=False, right=False, 
			labelleft=False, labelbottom=False)
	#plt.title('I love U')
	plt.tight_layout()
	plt.savefig('IloveU.pdf')
	plt.show()
	plt.close()
