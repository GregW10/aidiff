import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import struct
from unet256_nolog import UNet
from tqdm import tqdm
import os
import sys
import subprocess
import math
import tempfile


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
            mu, sig = lognormal_to_gauss(0.005, 5)
            muz, sigz = lognormal_to_gauss(0.01, 4)
            params = torch.zeros(size=(num_samples, 3), device=self.device)
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
        normed_params = params.clone()
        normed_params[:, 0] /= extrema["aperture_max"]
        normed_params[:, 1] /= extrema["wavelength_max"]
        normed_params[:, 2] /= extrema["distance_max"]
        x = torch.randn((num_samples, self.num_channels, self.width, self.height)).to(self.device)
        pfunc = tqdm if os.isatty(sys.stdout.fileno()) else lambda x: x

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

"""
def run_dffrcc_simulation(aperture, wavelength, distance, output_dffr_path, output_bmp_path):
    command = [
        "dffrcc", "--nx", "256", "--ny", "256", "--lam", str(wavelength),
        "--xa", str(-aperture/2), "--xb", str(aperture/2),
        "--ya", str(-aperture/2), "--yb", str(aperture/2),
        "--z", str(distance), "--w", str(1.5 * distance), "--l", str(1.5 * distance),
        "--I0", "10000", "--ptol_x", "1e-15", "--ptol_y", "1e-15", "--atol_x", "1e-10", "--atol_y", "1e-10",
        "--rtol_x", "1e-10", "--rtol_y", "1e-10", "-f", "lf",
        "--dffr", output_dffr_path, "--bmp", output_bmp_path, "--ptime", "1", "--cmap", "grayscale"
    ]
    subprocess.run(command, check=True)
"""


def run_dffrcc_simulation(aperture, wavelength, distance, output_dffr_path, output_bmp_path):
    # Constants derived from gen.sh
    l_max = -8
    l_b = -5
    z_b = math.exp(l_b * math.log(10))
    min_tol = math.exp(l_max * math.log(10))
    tol_cut = 1e-12
    lams = -8.5

    # Calculate tolerance values
    logz = math.log(distance) / math.log(10)
    lamdiff = lams + (math.log(wavelength) / math.log(10)) - logz

    if distance <= z_b and lamdiff >= l_max:
        tol = min_tol
    else:
        zdiff = l_max + 1.5 * (l_b - logz)
        if zdiff <= lamdiff:
            tol = math.exp(zdiff * math.log(10))
        else:
            tol = math.exp(lamdiff * math.log(10))

    # Determine the floating point precision
    ftype = "Lf" if tol < tol_cut else "lf"

    # Adjust detector width and length if necessary
    wl = 1.5 * distance
    one_p5_apl = 1.5 * aperture
    if wl < one_p5_apl:
        wl = one_p5_apl

    command = [
        "dffrcc", "--nx", "256", "--ny", "256", "--lam", str(wavelength),
        "--xa", str(-aperture/2), "--xb", str(aperture/2),
        "--ya", str(-aperture/2), "--yb", str(aperture/2),
        "--z", str(distance), "--w", str(wl), "--l", str(wl),
        "--I0", "10000", "--ptol_x", "1e-15", "--ptol_y", "1e-15",
        "--atol_x", str(tol), "--atol_y", str(tol),
        "--rtol_x", str(tol), "--rtol_y", str(tol),
        "-f", ftype, "--dffr", output_dffr_path, "--bmp", output_bmp_path,
        "--ptime", "1", "--cmap", "grayscale", "-v"
    ]
    subprocess.run(command, check=True)
    if ftype == "Lf":
        subprocess.run(["mkdir", "shitdir"], check=True)
        subprocess.run(["fconv", "lf", output_dffr_path, "-o", "shitdir"], check=True)
        subprocess.run(["mv", "-v", f"shitdir/{os.listdir('shitdir')[0]}", f"./{output_dffr_path}"], check=True)
        subprocess.run(["rm", "-rfv", "shitdir"], check=True)


def load_dffr(file_path):
    with open(file_path, "rb") as f:
        header = f.read(4)
        sizeof_T = struct.unpack('i', f.read(4))[0]
        mantissa_digits = struct.unpack('i', f.read(4))[0]
        wavelength = struct.unpack('d', f.read(8))[0]
        z_distance = struct.unpack('d', f.read(8))[0]
        detector_width = struct.unpack('d', f.read(8))[0]
        detector_length = struct.unpack('d', f.read(8))[0]
        incident_intensity = struct.unpack('d', f.read(8))[0]
        horizontal_resolution = struct.unpack('q', f.read(8))[0]
        vertical_resolution = struct.unpack('q', f.read(8))[0]
        aperture_id = struct.unpack('i', f.read(4))[0]
        lower_x_limit = struct.unpack('d', f.read(8))[0]
        upper_x_limit = struct.unpack('d', f.read(8))[0]
        lower_y_limit = struct.unpack('d', f.read(8))[0]
        upper_y_limit = struct.unpack('d', f.read(8))[0]
        data = np.fromfile(f, dtype=np.float64).reshape((vertical_resolution, horizontal_resolution))
    return data


def save_dffr(file_path, data, params, detector_size=256):
    with open(file_path, "wb") as f:
        # Header and metadata
        f.write(b'DFFR')
        f.write(struct.pack('i', 4))  # sizeof(float)
        f.write(struct.pack('i', 24))  # DBL_MANT_DIG
        f.write(struct.pack('f', params[1]))  # Wavelength
        f.write(struct.pack('f', params[2]))  # z_distance
        f.write(struct.pack('f', 1.5 * params[2]))  # Detector width
        f.write(struct.pack('f', 1.5 * params[2]))  # Detector length
        f.write(struct.pack('f', 10_000))  # Incident light intensity
        f.write(struct.pack('q', detector_size))  # Horizontal resolution
        f.write(struct.pack('q', detector_size))  # Vertical resolution
        f.write(struct.pack('i', 0))  # Rectangular aperture
        f.write(struct.pack('f', -params[0]/2))  # Lower x-limit
        f.write(struct.pack('f', params[0]/2))  # Upper x-limit
        f.write(struct.pack('f', -params[0]/2))  # Lower y-limit
        f.write(struct.pack('f', params[0]/2))  # Upper y-limit

        # Intensity data
        data.tofile(f)


def l2_true_distance(pattern1, pattern2):
    return np.linalg.norm(pattern1 - pattern2)


def l2_norm_distance(pattern1, pattern2):
    return np.linalg.norm((pattern1/np.max(pattern1)) - (pattern2/np.max(pattern2)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load('model.pth'))
    extrema = load_extrema("extrema.xtr")
    diff_model = DiffusionModel(T=1_000, model=model, width=256, height=256, device=device)

    num_samples = 1
    samples, params = diff_model.sample(extrema, num_samples=num_samples)
    
    total_true_distance = 0.0
    total_norm_distance = 0.0

    f = open("distances.csv", "w")
    f.write("aperture_size,wavelength,distance,true_distance,norm_distance\n")

    for i in range(num_samples):
        ap = params[i][0].item()
        wav = params[i][1].item()
        dist = params[i][2].item()

        # Generate paths for the output files
        model_dffr_path = f"model_ap{ap}_wav{wav}_dist{dist}.dffr"
        model_bmp_path = f"model_ap{ap}_wav{wav}_dist{dist}.bmp"
        dffrcc_dffr_path = f"dffrcc_ap{ap}_wav{wav}_dist{dist}.dffr"
        dffrcc_bmp_path = f"dffrcc_ap{ap}_wav{wav}_dist{dist}.bmp"

        # Save model-generated data as .dffr file
        model_pattern = samples[i][0].cpu().numpy()
        save_dffr(model_dffr_path, model_pattern, params[i])

        # Run the dffrcc simulator and generate the corresponding .dffr file
        run_dffrcc_simulation(ap, wav, dist, dffrcc_dffr_path, dffrcc_bmp_path)

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
        plt.savefig(f"comparison_ap{ap}_wav{wav}_dist{dist}.png")
        plt.close()

    print(f"Average L2 True Distance: {total_true_distance / num_samples}")
    print(f"Average L2 Norm. Distance: {total_norm_distance / num_samples}")

    f.close()


if __name__ == "__main__":
    main()
