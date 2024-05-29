# T formula

import numpy as np
import matplotlib.pyplot as plt

generations = np.arange(0, 500)

temperatures = np.maximum(np.exp(-generations / 100), 0.1)

plt.figure(figsize=(10, 6))
plt.plot(generations, temperatures, label='T = max(e^{-g/100}, 0.1)')
plt.title('Exponential Decay of Temperature over Generations with Lower Bound')
plt.xlabel('Generation (g)')
plt.ylabel('Temperature (T)')
plt.grid(True)
plt.show()