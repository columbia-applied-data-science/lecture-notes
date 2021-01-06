import scipy as sp
import matplotlib.pyplot as plt
from pylab import Arrow

delta = 0.95
r = sp.log(delta / (1 - delta))

z1 = sp.randn(8, 2) + sp.array([1, 1])
z2 = sp.randn(8, 2) + sp.array([2, 3])
z = sp.r_[z1, z2]
w = sp.array([1, 1])
mask = z.dot(w) > r

fig, axes = plt.subplots(1, 1)

plt.plot(z[mask,0], z[mask,1], 'r.', ms=15, mew=7, label='Class 1')
plt.plot(z[~mask,0], z[~mask,1], 'bx', ms=15, mew=7, label='Class 0')

axshape = plt.axis()

# Plot the normal
t = sp.linspace(-3, 9, 3)
plt.plot(t, r - t, 'k-', lw=8)

# Reset the axis to the correct size
plt.axis(axshape)
plt.axis([0, 4, 0, 4])

# Legend
plt.legend(loc='best')

# Make an arrow
arr = Arrow(1, r - 1, 1, 1, label='w')
ax = plt.gca()
ax.add_patch(arr)
ax.text(1, r - 0.6, 'w', fontsize=50)



#plt.xlabel('$x$', fontsize=20)
#plt.ylabel('$y$', fontsize=20)

plt.show()
