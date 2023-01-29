import numpy as np
import matplotlib.pyplot as plt

phi = (1+5**0.5)/2

def P2C(r, theta):
	x = r*np.cos(theta)
	y = r*np.sin(theta)
	return x, y
	
def s2r(s, a = 1, c = phi**(2/np.pi)):
	return a*c**s
	
npoint = 1000
ncircle = 3.25
ls = np.linspace(0, ncircle*2*np.pi, npoint)
lr = s2r(ls)

lx, ly = P2C(lr, ls)

c = '#009E73'
plt.figure()
ax = plt.subplot(111)
ax.set_aspect(aspect=1)
plt.plot(lx, ly, color=c)
plt.title(r'Golden spiral')#: $N_{c}='+str(ncircle)+'$')
plt.tight_layout()
plt.savefig('nautilus.pdf')
plt.show()
plt.close()
