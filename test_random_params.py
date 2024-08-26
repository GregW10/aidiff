import numpy as np
import torch
import matplotlib.pyplot as plt
import math

def lognormal_to_gauss(mu_ln: float, sigma_ln: float) -> tuple[float, float]:
    sig_over_mu = sigma_ln / mu_ln
    squared_p1 = 1 + sig_over_mu * sig_over_mu
    return (math.log(mu_ln / math.sqrt(squared_p1)), math.sqrt(math.log(squared_p1)))

np.random.seed(2)

mu, sig = lognormal_to_gauss(0.005, 4.5)
muz, sigz = lognormal_to_gauss(0.02, 4.5)

num_samples = 100
params = torch.zeros(size=(num_samples, 3))

for i in range(num_samples):
    ap = np.random.lognormal(mean=mu, sigma=sig)
    wav = np.random.lognormal(mean=mu, sigma=sig)
    while wav > ap or wav * 20 < ap:
        ap = np.random.lognormal(mean=mu, sigma=sig)
        wav = np.random.lognormal(mean=mu, sigma=sig)
    zd = np.random.lognormal(mean=muz, sigma=sigz)
    params[i][0] = ap
    params[i][1] = wav
    params[i][2] = zd

print("Extrema for 'ap':")
print(f"  Min: {params[:, 0].min().item()}")
print(f"  Max: {params[:, 0].max().item()}")

print("\nExtrema for 'wav':")
print(f"  Min: {params[:, 1].min().item()}")
print(f"  Max: {params[:, 1].max().item()}")

print("\nExtrema for 'zd':")
print(f"  Min: {params[:, 2].min().item()}")
print(f"  Max: {params[:, 2].max().item()}")

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.scatter(params[:, 0], params[:, 1], c='blue', label='ap vs wav')
plt.xlabel('ap')
plt.ylabel('wav')
plt.title('ap vs wav')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(params[:, 0], params[:, 2], c='green', label='ap vs zd')
plt.xlabel('ap')
plt.ylabel('zd')
plt.title('ap vs zd')
plt.legend()

plt.tight_layout()
plt.show()
