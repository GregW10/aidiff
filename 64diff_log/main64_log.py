import torch
import torch.nn as nn
import os
import signal
import sys
from itertools import cycle
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from unet64_log import UNet
from tqdm import tqdm
import numpy as np
import struct

from torch.utils.data import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

import math


def lognormal_to_gauss(mu_ln: float, sigma_ln: float) -> tuple[float, float]:
    sig_over_mu = sigma_ln / mu_ln
    squared_p1 = 1 + sig_over_mu * sig_over_mu
    return (math.log(mu_ln / math.sqrt(squared_p1)), math.sqrt(math.log(squared_p1)))


def is_symmetric(image, threshold=0.001):
    """
    Check if the image is symmetric about y=x and y=-x diagonals.
    Args:
        image (np.ndarray): 2D array representing the image.
        threshold (float): Threshold for asymmetry.
    Returns:
        bool: True if the image is symmetric, False otherwise.
    """
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image must be square for this symmetry check.")
    
    flipped_lr = np.fliplr(image)
    flipped_ud = np.flipud(image)
    symm_yx = np.abs(image - flipped_lr.T).mean()
    symm_ynegx = np.abs(image - flipped_ud.T).mean()
    
    return symm_yx < threshold and symm_ynegx < threshold


def average_pooling(data, size=64):
    """
    Downsample the data from 256x256 to size x size using average pooling.
    """
    pooled_data = data.reshape(size, data.shape[0] // size, size, data.shape[1] // size).mean(axis=(1, 3))
    return pooled_data


class DFFRDataset(Dataset):
    def __init__(self, data_dir, device, threshold=0.00001):
        self.data_dir = data_dir
        self.device = device
        self.threshold = threshold
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dffr')]
        self.discarded_files = []
        if threshold >= 0.0:
            self.files = self.filter_symmetric_files()
        self.extrema = self.find_extrema()
        print("Data Extremes:", self.extrema)
        serialised_data = bytearray()
        for value in self.extrema.values():
            serialised_data.extend(struct.pack('d', value))
        with open("extrema.xtr", "wb") as f:
            f.write(serialised_data)

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

        assert data.shape == (256, 256), "Shape is not 256x256"

        # Downsample to 64x64 using average pooling
        data = average_pooling(data, size=64)

        # Normalize data using log values
        log_data = np.log(data)
        log_norm_data = (log_data - self.extrema['log_intensity_min']) / (self.extrema['log_intensity_max'] - self.extrema['log_intensity_min'])
        log_norm_params = np.array([
            (np.log(aperture_size) - self.extrema['log_aperture_min']) / (self.extrema['log_aperture_max'] - self.extrema['log_aperture_min']),
            (np.log(wavelength) - self.extrema['log_wavelength_min']) / (self.extrema['log_wavelength_max'] - self.extrema['log_wavelength_min']),
            (np.log(z_distance) - self.extrema['log_distance_min']) / (self.extrema['log_distance_max'] - self.extrema['log_distance_min'])
        ])
        return torch.tensor(log_norm_data, dtype=torch.float32).unsqueeze(0).to(self.device), torch.tensor(log_norm_params, dtype=torch.float32).to(self.device)

    def filter_symmetric_files(self):
        valid_files = []
        discarded_dir = 'discarded_pats'
        os.makedirs(discarded_dir, exist_ok=True)
        
        for file_path in tqdm(self.files, desc='Filtering files for symmetry'):
            with open(file_path, 'rb') as f:
                f.seek(104)
                data = np.fromfile(f, dtype=np.float64)
                if data.size != 256 * 256:
                    print(f"Unexpected size for file {file_path}: {data.size}")
                    self.discarded_files.append(file_path)
                    continue
                
                data = data.reshape((256, 256))
                # Downsample before checking symmetry
                data = average_pooling(data, size=64)
                if is_symmetric(data, self.threshold):
                    valid_files.append(file_path)
                else:
                    self.discarded_files.append(file_path)
                    base_name = os.path.basename(file_path).replace('.dffr', '.png')
                    plt.imshow(data, cmap='gray')
                    plt.title(f'Discarded: {base_name}')
                    plt.savefig(os.path.join(discarded_dir, base_name))
                    plt.clf()
        
        print("Discarded Files:")
        for file_path in self.discarded_files:
            print(file_path)
        
        return valid_files

    def find_extrema(self):
        min_intensities = []
        max_intensities = []
        apertures = []
        wavelengths = []
        distances = []

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

                data = average_pooling(data)

                min_intensities.append(np.min(data))
                max_intensities.append(np.max(data))
                apertures.append(aperture_size)
                wavelengths.append(wavelength)
                distances.append(z_distance)

        log_min_intensities = np.log(min_intensities)
        log_max_intensities = np.log(max_intensities)
        log_apertures = np.log(apertures)
        log_wavelengths = np.log(wavelengths)
        log_distances = np.log(distances)

        return {
            'log_intensity_min': np.min(log_min_intensities),
            'log_intensity_max': np.max(log_max_intensities),
            'log_aperture_min': np.min(log_apertures),
            'log_aperture_max': np.max(log_apertures),
            'log_wavelength_min': np.min(log_wavelengths),
            'log_wavelength_max': np.max(log_wavelengths),
            'log_distance_min': np.min(log_distances),
            'log_distance_max': np.max(log_distances)
        }


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

        signal.signal(signal.SIGHUP, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGQUIT, self.handle_signal)
        self.rank = dist.get_rank() if dist.is_initialized() else 0  # Get the rank of the process

    def handle_signal(self, signum, frame):
        print(f"Received signal {signum}. Saving model...")
        self.save_model()
        sys.exit(0)

    def save_model(self):
        if self.rank == 0:  # Only save the model from process with rank 0
            torch.save(self.model.state_dict(), "interrupted_model.pth")
            print(f"Model saved.")

    def train(self, dataloader, optimiser, num_epochs=10_000, csv_f="losses.csv", model_path="model.pth", accumulation_steps=4):
        if self.rank == 0:
            cf = open(csv_f, "w")
            cf.write("epoch,loss\n")

        pfunc = tqdm if (self.rank == 0 and os.isatty(sys.stdout.fileno())) else lambda x: x

        for epoch in pfunc(range(1, num_epochs + 1)):
            optimiser.zero_grad()
            for _ in range(accumulation_steps):
                x_0, params = next(dataloader)
                x_0 = x_0.to(self.device)
                params = params.to(self.device)
                t = torch.randint(low=1, high=self.T + 1, size=(x_0.size(0),)).to(self.device)
                eps = torch.randn_like(x_0).to(self.device)
                x_t = torch.sqrt(self.bar_alphas[t - 1, None, None, None]) * x_0 + torch.sqrt(1 - self.bar_alphas[t - 1, None, None, None]) * eps
                loss = torch.nn.functional.mse_loss(self.model(x_t, t, params), eps) / accumulation_steps

                loss.backward()

            optimiser.step()

            if self.rank == 0 and epoch % 100 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}", flush=True)
                cf.write(f"{epoch},{loss.item()}\n")

        dist.barrier()
        if self.rank == 0:
            cf.close()
            torch.save(self.model.state_dict(), model_path)

    @torch.no_grad
    def sample(self, dataset: DFFRDataset, num_samples=1, params=None):
        if params is None:
            # params = torch.tensor([[0.5, 500e-9, 1.0]] * num_samples).to(self.device)  # Example parameters
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
        log_params = torch.log(params)
        log_norm_params = (log_params - torch.tensor([dataset.extrema['log_aperture_min'], dataset.extrema['log_wavelength_min'], dataset.extrema['log_distance_min']]).to(self.device)) / \
                          (torch.tensor([dataset.extrema['log_aperture_max'], dataset.extrema['log_wavelength_max'], dataset.extrema['log_distance_max']]).to(self.device) - \
                           torch.tensor([dataset.extrema['log_aperture_min'], dataset.extrema['log_wavelength_min'], dataset.extrema['log_distance_min']]).to(self.device))
        x = torch.randn((num_samples, self.num_channels, self.width, self.height)).to(self.device)
        pfunc = tqdm if self.rank == 0 else lambda x: x

        for t in pfunc(range(self.T, 0, -1)):
            z = torch.randn_like(x).to(self.device) if t > 1 else torch.zeros_like(x).to(self.device)
            x = (1 / torch.sqrt(self.alphas[t - 1, None, None, None]) * \
                 (x - ((1 - self.alphas[t - 1, None, None, None]) / torch.sqrt(1 - self.bar_alphas[t - 1, None, None, None])) * self.model(x, torch.Tensor([t] * num_samples).to(self.device), log_norm_params)) + \
                 torch.sqrt(self.betas[t - 1, None, None, None]) * z)
        
        # Undo normalization on intensity
        norm_x = x * (dataset.extrema['log_intensity_max'] - dataset.extrema['log_intensity_min']) + dataset.extrema['log_intensity_min']
        original_x = torch.exp(norm_x)
        
        # Undo normalization on params
        # original_params = torch.exp(log_norm_params * (torch.tensor([dataset.extrema['log_aperture_max'], dataset.extrema['log_wavelength_max'], dataset.extrema['log_distance_max']]).to(self.device) - \
        #                           torch.tensor([dataset.extrema['log_aperture_min'], dataset.extrema['log_wavelength_min'], dataset.extrema['log_distance_min']]).to(self.device)) + \
        #                           torch.tensor([dataset.extrema['log_aperture_min'], dataset.extrema['log_wavelength_min'], dataset.extrema['log_distance_min']]).to(self.device))
        return original_x, params#original_params



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def train(rank, world_size, data_dir, sym_threshold, T, batch_size, lr, epochs):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    dataset = DFFRDataset(data_dir, device, threshold=sym_threshold)
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    infinite_dataloader = cycle(dataloader)

    model = UNet().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    diff_model = DiffusionModel(T, model, width=64, height=64, device=device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    diff_model.train(infinite_dataloader, optim, num_epochs=epochs, accumulation_steps=4)

    cleanup()


def main_multi_gpu():
    print("Multi-GPU execution.")
    data_dir = "data/"
    sym_threshold = 1e-5
    T = 1_000
    batch_size = 16  # Adjust based on GPU memory
    lr = 1e-5
    epochs = 100_000

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, data_dir, sym_threshold, T, batch_size, lr, epochs), nprocs=world_size, join=True)

    # I will soon implement a completely different script for the sampling
    """
    # Sampling part can be done separately after training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load('model.pth'))
    diff_model = DiffusionModel(T, model, width=64, height=64, device=device)

    num_samples = 20
    samples, params = diff_model.sample(num_samples)

    counter = 0
    while counter < num_samples:
        plt.imshow(samples[counter][0].cpu().numpy(), cmap="gray")
        plt.title(f"Aperture: {params[counter][0].item()}, Wavelength: {params[counter][1].item()}, Distance: {params[counter][2].item()}")
        plt.savefig(f"sample{counter}.png")
        plt.clf()
        counter += 1
        """


def main_single_gpu():
    print("Single/zero-GPU execution.")
    data_dir = "data/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sym_threshold = 1e-5
    dataset = DFFRDataset(data_dir, device, threshold=sym_threshold)
    batch_size = 4  # Reduced batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    infinite_dataloader = cycle(dataloader)

    T = 1_000

    # sys.exit(0)

    model = UNet().to(device)

    if len(sys.argv) > 1:
        model.load_state_dict(torch.load(sys.argv[1]))

    diff_model = DiffusionModel(T, model, width=64, height=64, device=device)

    lr = 1e-5
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 100_000

    acc_steps = 4

    diff_model.train(infinite_dataloader, optim, num_epochs=epochs, accumulation_steps=acc_steps)

    num_samples = 20
    samples, params = diff_model.sample(dataset=dataset, num_samples=num_samples)

    counter = 0
    while counter < num_samples:
        plt.imshow(samples[counter][0].cpu().numpy(), cmap="gray")
        plt.title(f"Aperture: {params[counter][0].item()}, Wavelength: {params[counter][1].item()}, Distance: {params[counter][2].item()}")
        plt.savefig(f"sample{counter}.png")
        plt.clf()
        counter += 1

def main():
    if torch.cuda.device_count() > 1:
        main_multi_gpu()
    else:
        main_single_gpu()

if __name__ == "__main__":
    main()
