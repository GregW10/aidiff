import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import struct
from simple_cnn import SimpleCNN
import os
import sys
import math


def lognormal_to_gauss(mu_ln: float, sigma_ln: float) -> tuple[float, float]:
    sig_over_mu = sigma_ln / mu_ln
    squared_p1 = 1 + sig_over_mu * sig_over_mu
    return (math.log(mu_ln / math.sqrt(squared_p1)), math.sqrt(math.log(squared_p1)))


def load_extrema(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    extrema_values = struct.unpack('d' * (len(data) // 8), data)
    extrema_keys = ["intensity_min", "intensity_max", "aperture_min", "aperture_max", "wavelength_min", "wavelength_max", "distance_min", "distance_max"]
    return dict(zip(extrema_keys, extrema_values))


def generate_random_params(extrema, num_samples=1):
    mu, sig = lognormal_to_gauss(0.005, 4)
    muz, sigz = lognormal_to_gauss(0.01, 3)
    params = torch.zeros(size=(num_samples, 3))
    """
    for i in range(num_samples):
        # ap = np.random.uniform(extrema["aperture_min"], extrema["aperture_max"])
        # wav = np.random.uniform(extrema["wavelength_min"], extrema["wavelength_max"])
        # zd = np.random.uniform(extrema["distance_min"], extrema["distance_max"])
        # params[i][0] = ap
        # params[i][1] = wav
        # params[i][2] = zd
        params[i][0] = np.random.lognormal(mu, sig)
        params[i][1] = np.random.lognormal(mu, sig)
        params[i][2] = np.random.lognormal(muz, sigz)
    """
    for i in range(num_samples):
        ap = np.random.lognormal(mean=mu, sigma=sig)
        wav = np.random.lognormal(mean=mu, sigma=sig)
        while wav > ap or wav*20 < ap:
            ap = np.random.lognormal(mean=mu, sigma=sig)
            wav = np.random.lognormal(mean=mu, sigma=sig)
        zd = np.random.lognormal(mean=muz, sigma=sigz)
        params[i][0] = ap
        params[i][1] = wav
        params[i][2] = zd
    return params


def normalize_params(params, extrema):
    normed_params = params.clone()
    normed_params[:, 0] /= extrema["aperture_max"]
    normed_params[:, 1] /= extrema["wavelength_max"]
    normed_params[:, 2] /= extrema["distance_max"]
    return normed_params


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('simple_cnn.pth', map_location=torch.device(device)))
    extrema = load_extrema("extrema.xtr")

    num_samples = 20
    params = generate_random_params(extrema, num_samples=num_samples)
    normed_params = normalize_params(params, extrema).to(device)

    model.eval()
    with torch.no_grad():
        samples = model(normed_params)

    for i in range(num_samples):
        plt.imshow(samples[i][0].cpu().numpy(), cmap="gray")
        plt.title(f"Aperture: {params[i][0].item()}, Wavelength: {params[i][1].item()}, Distance: {params[i][2].item()}")
        plt.savefig(f"ap{params[i][0].item()}_wav{params[i][1].item()}_dist{params[i][2].item()}.png")
        plt.clf()


if __name__ == "__main__":
    main()
