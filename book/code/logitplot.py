import scipy as sp
import matplotlib.pyplot as plt

def p(z):
    expz = sp.exp(z)

    return expz / (1 + expz)

x = sp.linspace(-5, 5, 100)
y = p(x)

fig, axes = plt.subplots(1, 1)

plt.plot(x, y, lw=8)
plt.axis([-5, 5, 0, 1])
plt.xlabel('$z$', fontsize=20)
plt.ylabel('$\\sigma(z)$', fontsize=20)

plt.show()
