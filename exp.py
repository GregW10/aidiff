import numpy as np
import matplotlib.pyplot as plt

start_beta = 0.00001
stop_beta  = 0.025
steep      = 4

x = np.linspace(0, 1, 1_000)

betas = (((np.exp(steep*x) - 1)/(np.e**steep - 1)))*(stop_beta - start_beta) + start_beta

print(f"Min beta: {np.min(betas)}\nMax. beta: {np.max(betas)}")

plt.plot(x, betas)
plt.show()

