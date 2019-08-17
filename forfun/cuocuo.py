import numpy as np
import matplotlib.pyplot as plt

top = lambda x: (1-(np.abs(x)-1)**2)**0.5
bottom = lambda x: -np.arccos(np.abs(x)-1)

lx = np.linspace(-2, 2, 521)
ly1 = top(lx)
ly2 = bottom(lx)

asp = 3./np.pi

plt.figure()
ax = plt.subplot(111)
ax.set_aspect('equal')
plt.fill_between(lx, ly1, ly2*asp, facecolor='r', alpha=0.5,
				edgecolor='red')
plt.title('I love U')
plt.tight_layout()
plt.savefig('IloveU.pdf')
plt.show()
plt.close()
