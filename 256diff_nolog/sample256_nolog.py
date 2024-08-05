import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import struct
from unet256_nolog import UNet
from tqdm import tqdm

import math

def lognormal_to_gauss(mu_ln: float, sigma_ln: float) -> tuple[float, float]:
    sig_over_mu = sigma_ln / mu_ln
    squared_p1 = 1 + sig_over_mu * sig_over_mu
    return (math.log(mu_ln / math.sqrt(squared_p1)), math.sqrt(math.log(squared_p1)))

class DiffusionModel:
    def __init__(self, T: int, model: nn.Module, width: int = 256, height: int = 256, num_channels: int = 1, device: str = "cpu"):
        self.T = T
        self.model = model
        self.width = width
        self.height = height
        self.num_channels = num_channels
        self.device = device

        self.betas = torch.linspace(start=0.0001, end=0.02, steps=T, device=device)
        self.alphas = (1 - self.betas).to(device)
        self.bar_alphas = torch.cumprod(self.alphas, dim=0).to(device)

    @torch.no_grad
    def sample(self, extrema, num_samples=1, params=None):
        if params is None:
            mu, sig = lognormal_to_gauss(0.01, 5)
            params = torch.zeros(size=(num_samples, 3), device=self.device)
            for i in range(num_samples):
                ap = np.random.lognormal(mean=mu, sigma=sig, size=1)
                wav = np.random.lognormal(mean=mu, sigma=sig, size=1)
                while wav > ap:
                    ap = np.random.lognormal(mean=mu, sigma=sig, size=1)
                    wav = np.random.lognormal(mean=mu, sigma=sig, size=1)
                zd = np.random.lognormal(mean=mu, sigma=sig, size=1)
                params[i][0] = ap
                params[i][1] = wav
                params[i][2] = zd
        normed_params = params
        normed_params[:, 0] /= extrema["aperture_max"]
        normed_params[:, 1] /= extrema["wavelength_max"]
        normed_params[:, 2] /= extrema["distance_max"]
        x = torch.randn((num_samples, self.num_channels, self.width, self.height)).to(self.device)
        pfunc = tqdm if torch.cuda.device_count() == 0 else lambda x: x

        for t in pfunc(range(self.T, 0, -1)):
            z = torch.randn_like(x).to(self.device) if t > 1 else torch.zeros_like(x).to(self.device)
            x = (1 / torch.sqrt(self.alphas[t - 1, None, None, None]) * \
                 (x - ((1 - self.alphas[t - 1, None, None, None]) / torch.sqrt(1 - self.bar_alphas[t - 1, None, None, None])) * self.model(x, torch.Tensor([t] * num_samples).to(self.device), normed_params)) + \
                 torch.sqrt(self.betas[t - 1, None, None, None]) * z)
        return x, params

def load_extrema(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    extrema_values = struct.unpack('d' * (len(data) // 8), data)
    extrema_keys = ["intensity_min", "intensity_max", "aperture_min", "aperture_max", "wavelength_min", "wavelength_max", "distance_min", "distance_max"]
    return dict(zip(extrema_keys, extrema_values))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load('model.pth'))
    extrema = load_extrema("extrema.xtr")
    diff_model = DiffusionModel(T=1_000, model=model, width=256, height=256, device=device)

    num_samples = 20
    samples, params = diff_model.sample(extrema, num_samples=num_samples)

    for i in range(num_samples):
        plt.imshow(samples[i][0].cpu().numpy(), cmap="gray")
        plt.title(f"Aperture: {params[i][0].item()}, Wavelength: {params[i][1].item()}, Distance: {params[i][2].item()}")
        plt.savefig(f"sample_{i}.png")
        plt.clf()

if __name__ == "__main__":
    main()
