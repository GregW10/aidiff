import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import struct
from unet64_log import UNet
from tqdm import tqdm
import os
import sys

import math


def lognormal_to_gauss(mu_ln: float, sigma_ln: float) -> tuple[float, float]:
    sig_over_mu = sigma_ln / mu_ln
    squared_p1 = 1 + sig_over_mu * sig_over_mu
    return (math.log(mu_ln / math.sqrt(squared_p1)), math.sqrt(math.log(squared_p1)))


class DiffusionModel:
    def __init__(self, T: int, model: nn.Module, width: int = 64, height: int = 64, num_channels: int = 1, device: str = "cpu"):
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
                ap = np.random.lognormal(mean=mu, sigma=sig)
                wav = np.random.lognormal(mean=mu, sigma=sig)
                while wav > ap or wav*20 < ap:
                    ap = np.random.lognormal(mean=mu, sigma=sig)
                    wav = np.random.lognormal(mean=mu, sigma=sig)
                zd = np.random.lognormal(mean=mu, sigma=sig)
                params[i][0] = ap
                params[i][1] = wav
                params[i][2] = zd
        log_params = torch.log(params)
        log_norm_params = (log_params - torch.tensor([extrema['log_aperture_min'], extrema['log_wavelength_min'], extrema['log_distance_min']]).to(self.device)) / \
                          (torch.tensor([extrema['log_aperture_max'], extrema['log_wavelength_max'], extrema['log_distance_max']]).to(self.device) - \
                           torch.tensor([extrema['log_aperture_min'], extrema['log_wavelength_min'], extrema['log_distance_min']]).to(self.device))
        x = torch.randn((num_samples, self.num_channels, self.width, self.height)).to(self.device)
        pfunc = tqdm if os.isatty(sys.stdout.fileno()) else lambda x: x

        for t in pfunc(range(self.T, 0, -1)):
            z = torch.randn_like(x).to(self.device) if t > 1 else torch.zeros_like(x).to(self.device)
            x = (1 / torch.sqrt(self.alphas[t - 1, None, None, None]) * \
                 (x - ((1 - self.alphas[t - 1, None, None, None]) / torch.sqrt(1 - self.bar_alphas[t - 1, None, None, None])) * self.model(x, torch.Tensor([t] * num_samples).to(self.device), log_norm_params)) + \
                 torch.sqrt(self.betas[t - 1, None, None, None]) * z)
        
        # Undo normalization on intensity
        norm_x = x * (extrema['log_intensity_max'] - extrema['log_intensity_min']) + extrema['log_intensity_min']
        original_x = torch.exp(norm_x)
        
        return original_x, params

def load_extrema(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    extrema_values = struct.unpack('d' * (len(data) // 8), data)
    extrema_keys = ["log_intensity_min", "log_intensity_max", "log_aperture_min", "log_aperture_max", "log_wavelength_min", "log_wavelength_max", "log_distance_min", "log_distance_max"]
    return dict(zip(extrema_keys, extrema_values))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load('model.pth'))
    extrema = load_extrema("extrema.xtr")
    diff_model = DiffusionModel(T=1_000, model=model, width=64, height=64, device=device)

    num_samples = 20
    samples, params = diff_model.sample(extrema, num_samples=num_samples)

    for i in range(num_samples):
        plt.imshow(samples[i][0].cpu().numpy(), cmap="gray")
        plt.title(f"Aperture: {params[i][0].item()}, Wavelength: {params[i][1].item()}, Distance: {params[i][2].item()}")
        plt.savefig(f"ap{params[i][0].item()}_wav{params[i][1].item()}_dist{params[i][2].item()}.png")
        plt.clf()


if __name__ == "__main__":
    main()
