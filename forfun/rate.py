import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def rate(x, x1=-0.5, x2=2, x3=2.8, y0=-1, y1=1, y2=5, y3=13):
	y = y0*(x<x1)
	y += y1*(x>=x1)*(x<x2)
	y += y2*(x>=x2)*(x<x3)
	y += y3*(x>x3)
	return y

if __name__=="__main__":
	x1, x2 = -3, 3
	lx = np.linspace(x1, x2, 1000)
	
	y1, y2 = -2, 15
	fig, ax1 = plt.subplots(figsize=(4,6))
	ax1.plot(lx, norm.pdf(lx)*30, 'r--')
	ax1.plot([x1,x2], [0]*2, 'k:')
	ax1.set_xlabel(r'property')
	ax1.set_ylabel(r'probability density', color='r')
	ax1.tick_params(axis="y", labelcolor='r')
	ax1.set_xlim(x1,x2)
	ax1.set_ylim(y1, y2)
	ax2 = ax1.twinx()
	ax2.plot(lx, rate(lx), 'b-')
	ax2.set_ylabel(r'rating', color='b')
	ax2.tick_params(axis="y", labelcolor='b')
	ax2.set_xlim(ax1.get_xlim())
	ax2.set_ylim(ax1.get_ylim())
	plt.tight_layout()
	plt.savefig('rate.png', dpi=300)
	plt.close()
