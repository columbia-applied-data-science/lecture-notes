import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

lw=5

x = sp.linspace(-2, 2, 100)

likelihood = stats.norm.pdf(x, loc=1, scale=0.5)
prior = stats.norm.pdf(x, loc=0, scale=1)
posterior = stats.norm.pdf(x, loc=4/5., scale=5**(-0.5))

plt.clf()

plt.plot(x, prior, 'r--', label='prior', lw=lw)
plt.plot(x, likelihood, 'b-', label='likelihood', lw=lw)
plt.plot(x, posterior, 'g', marker='>', ms=10, label='posterior', lw=lw)

plt.xlabel('x')
plt.ylabel('density')
plt.legend(loc='best')
plt.annotate('MAP estimate', (4/5., 0), xytext=(-1.5, 0.5), fontsize=20, arrowprops=dict(facecolor='black'))

plt.show()
