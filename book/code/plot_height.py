import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

age = np.arange(80)

height = np.ones(80) * np.nan

height[:18] = age[:18] * 0.1
height[18:60] = height[17]
height[60:] = height[59] + 0.01 * age[59] - 0.01 * age[60:]

plt.figure(1)
plt.xlabel('Age')

#plt.plot(age, height, lw=7, label='Mean height')

height += 0.3 * sp.randn(len(height))
plt.plot(age, height, 'g.', ms=10, label='Height')

z = sp.polyfit(age, height, 1)
p = np.poly1d(z)
plt.plot(age, p(age), 'r--', lw=7, label='Linear Fit')

plt.legend(loc='best')

p1 = np.poly1d(sp.polyfit(age[:18], height[:18], 1))
p2 = np.poly1d(sp.polyfit(age[18:60], height[18:60], 1))
p3 = np.poly1d(sp.polyfit(age[60:], height[60:], 1))

plt.figure(2)
plt.plot(age, height, 'g.', ms=10, label='Height')
plt.plot(age[:18], p1(age[:18]), 'k', lw=7, label='Segment 1')
plt.plot(age[18:60], p2(age[18:60]), 'k', lw=7, label='Segment 2')
plt.plot(age[60:], p3(age[60:]), 'k', lw=7, label='Segment 3')
plt.legend(loc='best')


plt.show()
