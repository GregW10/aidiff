import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import struct
from simple_cnn import SimpleCNN
from main32_nolog import lognormal_to_gauss #, load_extrema, save_dffr, run_dffrcc_simulation, load_dffr
from sampleANDcompare32_nolog import *
from tqdm import tqdm
import os
import sys
import subprocess
import math
import tempfile


def generate_random_params(extrema, num_samples=1):
    mu, sig = lognormal_to_gauss(0.005, 5)
    muz, sigz = lognormal_to_gauss(0.01, 4)
    params = torch.zeros(size=(num_samples, 3))
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
    model.load_state_dict(torch.load("model_simple_cnn.pth", map_location=torch.device(device)))
    extrema = load_extrema("extrema.xtr")

    num_samples = 2
    params = generate_random_params(extrema, num_samples=num_samples)
    normed_params = normalize_params(params, extrema).to(device)

    model.eval()
    with torch.no_grad():
        samples = model(normed_params)
        samples = samples*extrema["intensity_max"]

    total_true_distance = 0.0
    total_norm_distance = 0.0

    f = open("distances_simple_cnn.csv", "w")
    f.write("aperture_size,wavelength,distance,true_distance,norm_distance\n")

    for i in range(num_samples):
        ap = params[i][0].item()
        wav = params[i][1].item()
        dist = params[i][2].item()

        # Generate paths for the output files
        model_dffr_path = f"model_simple_cnn_ap{ap}_wav{wav}_dist{dist}.dffr"
        model_bmp_path = f"model_simple_cnn_ap{ap}_wav{wav}_dist{dist}.bmp"
        dffrcc_dffr_path = f"dffrcc_simple_cnn_ap{ap}_wav{wav}_dist{dist}.dffr"
        dffrcc_bmp_path = f"dffrcc_simple_cnn_ap{ap}_wav{wav}_dist{dist}.bmp"

        # Save model-generated data as .dffr file
        model_pattern = samples[i][0].cpu().numpy()
        save_dffr(model_dffr_path, model_pattern, params[i], detector_size=256)

        # Run the dffrcc simulator and generate the corresponding .dffr file
        run_dffrcc_simulation(ap, wav, dist, dffrcc_dffr_path, dffrcc_bmp_path, nx=256, ny=256)

        # Load the dffrcc-generated pattern
        dffrcc_pattern = load_dffr(dffrcc_dffr_path)

        # Calculate the L2 distance between the two patterns
        true_distance = l2_true_distance(model_pattern, dffrcc_pattern)
        total_true_distance += true_distance

        norm_distance = l2_norm_distance(model_pattern, dffrcc_pattern)
        total_norm_distance += norm_distance

        f.write(f"{ap},{wav},{dist},{true_distance},{norm_distance}\n")

        print(f"Sample {i+1}/{num_samples}:\n\tL2 True Distance: {true_distance}\n\tL2 Norm. Distance: {norm_distance}")

        # Optionally, save comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(model_pattern, cmap="gray")
        axes[0].set_title("Model-Generated")
        axes[1].imshow(dffrcc_pattern, cmap="gray")
        axes[1].set_title("dffrcc-Generated")
        plt.suptitle(f"Aperture: {ap} m, Wavelength: {wav} m,\nDistance: {dist} m")
        plt.savefig(f"comparison_simple_cnn_ap{ap}_wav{wav}_dist{dist}.png")
        plt.close()

    print(f"Average L2 True Distance: {total_true_distance / num_samples}")
    print(f"Average L2 Norm. Distance: {total_norm_distance / num_samples}")

    f.close()

if __name__ == "__main__":
    main()
