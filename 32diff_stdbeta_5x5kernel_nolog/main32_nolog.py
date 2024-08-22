import torch
import torch.nn as nn
import os
import signal
import sys
from itertools import cycle
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from unet32_5x5 import UNet
from tqdm import tqdm
import numpy as np
import struct

from torch.utils.data import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

import math
import time


def lognormal_to_gauss(mu_ln: float, sigma_ln: float) -> tuple[float, float]:
    sig_over_mu = sigma_ln / mu_ln
    squared_p1 = 1 + sig_over_mu * sig_over_mu
    return (math.log(mu_ln / math.sqrt(squared_p1)), math.sqrt(math.log(squared_p1)))


def is_symmetric(image, threshold=0.01):
    if image.shape[0] != image.shape[1]:
        raise ValueError("Error: image must be square for this symmetry check.")
    norm_img = image/np.max(image)
    flipped_lr = np.fliplr(norm_img)
    flipped_ud = np.flipud(norm_img)
    symm_yx = np.sqrt(np.sum(np.abs(norm_img**2 - flipped_lr.T**2)))
    symm_ynegx = np.sqrt(np.sum(np.abs(norm_img**2 - flipped_ud.T**2)))
    return symm_yx < threshold and symm_ynegx < threshold


def load_extrema(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    extrema_values = struct.unpack('d' * (len(data) // 8), data)
    extrema_keys = ["intensity_min", "intensity_max", "aperture_min", "aperture_max", "wavelength_min", "wavelength_max", "distance_min", "distance_max"]
    return dict(zip(extrema_keys, extrema_values))


class DFFRDataset(Dataset):
    def __init__(self, data_dir, device, Nx=32, Ny=32, threshold=0.01, rnk=0):
        # super(DFFRDataset, self).__init__()
        self.data_dir = data_dir
        self.device = device
        self.threshold = threshold
        self.nx = Nx
        self.ny = Ny
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dffr')]
        self.discarded_files = []
        if threshold >= 0.0:
            if os.path.exists("nonsym_files.list"):
                with open("nonsym_files.list", "r") as badf:
                    for line in badf:
                        self.discarded_files.append(line.strip())
                f = []
                for file in self.files:
                    if file not in self.discarded_files:
                        f.append(file)
                self.files = f
            else:
                self.files = self.filter_symmetric_files()
                if rnk == 0:
                    with open("nonsym_files.list", "w") as badf:
                        for file in self.discarded_files:
                            badf.write(f"{file}\n")
            if rnk == 0:
                print(f"Number of discarded files: {len(self.discarded_files)}")
        if os.path.exists("extrema.xtr"):
            self.extrema = load_extrema("extrema.xtr")
        else:
            self.extrema = self.find_extrema()
            if rnk == 0:
                serialised_data = bytearray()
                for value in self.extrema.values():
                    serialised_data.extend(struct.pack('d', value))
                with open("extrema.xtr", "wb") as f:
                    f.write(serialised_data)
        print(f"Number of files: {len(self.files)}")
        print(f"Data Extremes:\n{self.extrema}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        return self.load_dffr(file_path)

    def load_dffr(self, file_path):
        with open(file_path, 'rb') as f:
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
            aperture_size = upper_x_limit - lower_x_limit
            data = np.fromfile(f, dtype=np.float64).reshape((vertical_resolution, horizontal_resolution))

        assert data.shape == (self.ny, self.nx), f"Shape is not {self.nx}x{self.ny}"

        # data = average_pooling(data, 32)

        norm_data = data / self.extrema['intensity_max']
        norm_params = np.array([
            aperture_size / self.extrema['aperture_max'],
            wavelength / self.extrema['wavelength_max'],
            z_distance / self.extrema['distance_max']
        ])

        return torch.tensor(norm_data, dtype=torch.float32).unsqueeze(0).to(self.device), torch.tensor(norm_params, dtype=torch.float32).to(self.device)

    def filter_symmetric_files(self):
        valid_files = []
        discarded_dir = 'discarded_pats'
        os.makedirs(discarded_dir, exist_ok=True)
        for file_path in tqdm(self.files, desc='Filtering files for symmetry'):
            with open(file_path, 'rb') as f:
                f.seek(104)
                data = np.fromfile(f, dtype=np.float64)
                if data.size != self.nx*self.ny:
                    print(f"Error: unexpected size for file {file_path}: {data.size}")
                    self.discarded_files.append(file_path)
                    continue
                data = data.reshape((self.nx, self.ny))
                # data = average_pooling(data, 32)
                if is_symmetric(data, self.threshold):
                    valid_files.append(file_path)
                else:
                    self.discarded_files.append(file_path)
                    base_name = os.path.basename(file_path).replace('.dffr', '.png')
                    plt.imshow(data, cmap='gray')
                    plt.title(f'Discarded: {base_name}')
                    plt.savefig(os.path.join(discarded_dir, base_name))
                    plt.clf()
        plt.close()
        print("Discarded Files:")
        for file_path in self.discarded_files:
            print(file_path)
        return valid_files

    def find_extrema(self):
        intensity_min = float('inf')
        intensity_max = float('-inf')
        aperture_min = float('inf')
        aperture_max = float('-inf')
        wavelength_min = float('inf')
        wavelength_max = float('-inf')
        distance_min = float('inf')
        distance_max = float('-inf')
        for file_path in tqdm(self.files, desc='Finding extrema'):
            with open(file_path, 'rb') as f:
                f.seek(12)
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
                aperture_size = upper_x_limit - lower_x_limit
                data = np.fromfile(f, dtype=np.float64).reshape((vertical_resolution, horizontal_resolution))

                # data = average_pooling(data, 32)

                intensity_min = min(intensity_min, np.min(data))
                intensity_max = max(intensity_max, np.max(data))
                aperture_min = min(aperture_min, aperture_size)
                aperture_max = max(aperture_max, aperture_size)
                wavelength_min = min(wavelength_min, wavelength)
                wavelength_max = max(wavelength_max, wavelength)
                distance_min = min(distance_min, z_distance)
                distance_max = max(distance_max, z_distance)

        return {
            'intensity_min': intensity_min,
            'intensity_max': intensity_max,
            'aperture_min': aperture_min,
            'aperture_max': aperture_max,
            'wavelength_min': wavelength_min,
            'wavelength_max': wavelength_max,
            'distance_min': distance_min,
            'distance_max': distance_max
        }


def inverse_sigmoid(y):
    return np.log(y / (1 - y))


class DiffusionModel(nn.Module):
    def __init__(self,
                 T: int,
                 width: int = 32,
                 height: int = 32,
                 num_channels: int = 1,
                 device="cpu",
                 rnk=0):
        super(DiffusionModel, self).__init__()
        self.T = T
        self.model = UNet().to(device)
        self.width = width
        self.height = height
        self.num_channels = num_channels
        self.device = device

        # self.betas = torch.linspace(start=0.0001, end=0.02, steps=T, device=device)
        # self.alphas = (1 - self.betas).to(device)
        # self.bar_alphas = torch.cumprod(self.alphas, dim=0).to(device)

        # target_array = np.linspace(start=0.0001, stop=0.02, num=T)
        # inv_array = inverse_sigmoid(target_array)

        # start_beta = 0.00001
        # stop_beta  = 0.025
        # steep      = 4

        # x = np.linspace(0, 1, 1_000)

        # betas = (((np.exp(steep*x) - 1)/(np.e**steep - 1)))*(stop_beta - start_beta) + start_beta

        self.raw_betas = torch.linspace(start=0.0001, end=0.02, steps=T, device=device, dtype=torch.float32)

        # Define betas as a learnable parameter, I initialise with DDPMs paper values
        # self.betas = nn.Parameter(torch.linspace(start=0.0001, end=0.02, steps=T, device=device))
        # self.register_buffer("betas", torch.nn.functional.sigmoid(self.raw_betas).to(device))
        # self.betas = torch.nn.functional.sigmoid(self.raw_betas).to(device)

        # self.register_buffer('alphas', (1 - self.betas).to(device))  # Alphas depend on betas
        # self.register_buffer('bar_alphas', torch.cumprod(self.alphas, dim=0).to(device))
        # self.alphas = (1 - self.betas).to(device)
        # self.bar_alphas = torch.cumprod(self.alphas, dim=0).to(device)

        if rnk == 0 and __name__ == "__main__":
            signal.signal(signal.SIGHUP, self.handle_signal)
            signal.signal(signal.SIGINT, self.handle_signal)
            signal.signal(signal.SIGTERM, self.handle_signal)
            signal.signal(signal.SIGQUIT, self.handle_signal)

        # self.rank = dist.get_rank() if dist.is_initialized() else 0  # Get the rank of the process
        self.rank = rnk

    def handle_signal(self, signum, frame):
        print(f"Received signal {signum}. Saving model...")
        self.save_model()
        sys.exit(0)

    def save_model(self):
        if self.rank == 0:  # Only save the model from process with rank 0
            torch.save(self.state_dict(), f"interrupted_model_{self.rank}.pth")
            print(f"Model saved.")

    def train(self,
              dataloader,
              lr=1e-5,
              num_epochs=10_000,
              csv_f="losses.csv",
              model_path="model.pth",
              accumulation_steps=4):
        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        if self.rank == 0:
            _tt = int(time.time())
            model_dir = f"model_chkps{_tt}"
            noise_dir = f"noise_{_tt}"
            betas_dir = f"betas_{_tt}"
            os.mkdir(model_dir)
            os.mkdir(noise_dir)
            os.mkdir(betas_dir)
        if self.rank == 0:
            cf = open(csv_f, "w")
            cf.write("epoch,iteration,loss\n")

        # pfunc = tqdm if (self.rank == 0 and os.isatty(sys.stdout.fileno())) else lambda x: x

        optimiser.zero_grad()

        num_its = len(dataloader)

        elen = len(str(num_epochs))
        dlen = len(str(num_its))

        acloss = torch.tensor(0.0, device=self.device)
        accs = 0

        betas = self.raw_betas.clone()
        alphas = (1 - betas).to(self.device)
        bar_alphas = torch.cumprod(alphas, dim=0).to(self.device)

        for epoch in range(1, num_epochs + 1):
            # last_iter = 0
            for iteration, (x_0, params) in enumerate(dataloader, start=1):
                # for _ in range(accumulation_steps):
                # x_0, params = next(dataloader)

                x_0 = x_0.to(self.device)
                params = params.to(self.device)
                t = torch.randint(low=1, high=self.T + 1, size=(x_0.size(0),)).to(self.device)
                eps = torch.randn_like(x_0).to(self.device)
                x_t = torch.sqrt(bar_alphas[t - 1, None, None, None]) * x_0 + torch.sqrt(1 - bar_alphas[t - 1, None, None, None]) * eps
                # loss = torch.nn.functional.mse_loss(self.model(x_t, t, params), eps) / accumulation_steps
                loss = torch.nn.functional.mse_loss(self.model(x_t, t, params), eps)

                # loss.backward()

                acloss += loss  # torch.nn.functional.mse_loss(self.model(x_t, t, params), eps)
                accs += 1

                if accs == accumulation_steps:
                    acloss /= accumulation_steps

                    acloss.backward()

                    optimiser.step()
                    optimiser.zero_grad()

                    acloss = torch.tensor(0.0, device=self.device)
                    accs = 0

                if self.rank == 0 and iteration % 100 == 0:
                    print(f"Epoch {epoch}/{num_epochs}, it. {iteration}/{num_its}, Loss: {loss.item()}", flush=True)
                    cf.write(f"{epoch},{iteration},{loss.item()}\n")
                    cf.flush()
                    # if iteration % 100 == 0:
                    #     torch.save(self.state_dict(),
                    #                f"{model_dir}/model_epoch{epoch:0{elen}d}_it{iteration:0{dlen}d}.pth")
                    # if iteration % 10_000: # just to check, but is not necessary as they are not meant to be changing
                        # self.raw_betas.cpu().detach().numpy().tofile(
                            # f"{betas_dir}/rawbetas_epoch{epoch:0{elen}d}_it{iteration:0{dlen}d}.rbetas")
                        # betas.cpu().detach().numpy().tofile(
                            # f"{betas_dir}/betas_epoch{epoch:0{elen}d}_it{iteration:0{dlen}d}.betas")

                    if iteration % 10_000 == 0:
                        for (x0v, xtv, tv) in zip(x_0, x_t, t):
                            fig, axes = plt.subplots(1, 2)
                            axes[0].imshow(x0v[0].cpu().detach().numpy(), cmap="gray")
                            axes[0].set_title("t = 0")
                            axes[1].imshow(xtv[0].cpu().detach().numpy(), cmap="gray")
                            axes[1].set_title(f"t = {tv.item()}")
                            fig.savefig(f"{noise_dir}/epoch{epoch:0{elen}d}_it{iteration:0{dlen}d}_t{tv.item()}.png", dpi=600)
                            plt.clf()
                            plt.close()

            # actual_acums = last_iter % accumulation_steps

            if accs != 0:
                # for par in self.parameters():
                #     if par.grad is not None:
                #         par.grad.data.mul_(accumulation_steps/actual_acums)
                acloss /= accs

                acloss.backward()

                optimiser.step()
                optimiser.zero_grad()

                acloss = torch.tensor(0.0, device=self.device)
                accs = 0

            if self.rank == 0:
                torch.save(self.state_dict(), f"{model_dir}/model_epoch{epoch:0{elen}d}.pth")

        if dist.is_initialized():
            dist.barrier()
        if self.rank == 0:
            cf.close()
            torch.save(self.state_dict(), f"{model_dir}/model_path")

    @torch.no_grad()
    def sample(self, extrema, num_samples=1, params=None):
        betas = self.raw_betas.clone()
        alphas = ((1 - betas).to(self.device))
        bar_alphas = (torch.cumprod(alphas, dim=0).to(self.device))
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
        x = torch.randn((num_samples, self.num_channels, self.height, self.width)).to(self.device)
        pfunc = tqdm if (self.rank == 0 and os.isatty(sys.stdout.fileno())) else lambda x: x

        for t in pfunc(range(self.T, 0, -1)):
            z = torch.randn_like(x).to(self.device) if t > 1 else torch.zeros_like(x).to(self.device)
            x = (1 / torch.sqrt(alphas[t - 1, None, None, None]) * \
                 (x - ((1 - alphas[t - 1, None, None, None]) / torch.sqrt(1 - bar_alphas[t - 1, None, None, None])) * self.model(x, torch.Tensor([t] * num_samples).to(self.device), normed_params)) + \
                 torch.sqrt(betas[t - 1, None, None, None]) * z)
        return x*extrema["intensity_max"], params


def main_single_gpu():
    print("Single/zero-GPU execution.")
    data_dir = "data/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sym_threshold = 0.01 # higher now because I'm using Euclidean distance
    dataset = DFFRDataset(data_dir, device, threshold=sym_threshold)
    batch_size = 4  # Reduced batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # infinite_dataloader = cycle(dataloader)

    # sys.exit(0)

    T = 1_000

    # model = UNet().to(device)

    diff_model = DiffusionModel(T, width=32, height=32, device=device)

    if len(sys.argv) > 1:
        diff_model.load_state_dict(torch.load(sys.argv[1]))

    lr = 1e-5
    # optim = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 1_000_000

    acc_steps = 4

    diff_model.train(dataloader, lr=lr, num_epochs=epochs, accumulation_steps=acc_steps)


def main():
    print("Single GPU")
    main_single_gpu()


if __name__ == "__main__":
    main()
