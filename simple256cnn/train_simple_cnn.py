import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import numpy as np
import struct
import matplotlib.pyplot as plt
from simple_cnn import SimpleCNN

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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        return self.load_dffr(file_path)

    def load_dffr(self, file_path):
        with open(file_path, 'rb') as f:
            f.read(4)  # Skip header
            f.read(4)  # Skip size of T
            f.read(4)  # Skip mantissa digits
            wavelength = struct.unpack('d', f.read(8))[0]
            z_distance = struct.unpack('d', f.read(8))[0]
            f.read(8)  # Skip detector width
            f.read(8)  # Skip detector length
            incident_intensity = struct.unpack('d', f.read(8))[0]
            f.read(8)  # Skip horizontal resolution
            f.read(8)  # Skip vertical resolution
            aperture_id = struct.unpack('i', f.read(4))[0]
            lower_x_limit = struct.unpack('d', f.read(8))[0]
            upper_x_limit = struct.unpack('d', f.read(8))[0]
            lower_y_limit = struct.unpack('d', f.read(8))[0]
            upper_y_limit = struct.unpack('d', f.read(8))[0]
            aperture_size = upper_x_limit - lower_x_limit
            data = np.fromfile(f, dtype=np.float64).reshape((256, 256))

        assert data.shape == (256, 256), "Shape is not 256x256"

        norm_data = data / self.extrema['intensity_max']
        norm_params = np.array([
            aperture_size / self.extrema['aperture_max'],
            wavelength / self.extrema['wavelength_max'],
            z_distance / self.extrema['distance_max']
        ])

        return torch.tensor(norm_data, dtype=torch.float32).unsqueeze(0).to(self.device), torch.tensor(norm_params, dtype=torch.float32).to(self.device)

    def filter_symmetric_files(self):
        valid_files = []
        for file_path in tqdm(self.files, desc='Filtering files for symmetry'):
            with open(file_path, 'rb') as f:
                f.seek(104)
                data = np.fromfile(f, dtype=np.float64).reshape((256, 256))
                if is_symmetric(data, self.threshold):
                    valid_files.append(file_path)
                else:
                    self.discarded_files.append(file_path)
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
                f.read(8)  # Skip detector width
                f.read(8)  # Skip detector length
                incident_intensity = struct.unpack('d', f.read(8))[0]
                f.read(8)  # Skip horizontal resolution
                f.read(8)  # Skip vertical resolution
                aperture_id = struct.unpack('i', f.read(4))[0]
                lower_x_limit = struct.unpack('d', f.read(8))[0]
                upper_x_limit = struct.unpack('d', f.read(8))[0]
                lower_y_limit = struct.unpack('d', f.read(8))[0]
                upper_y_limit = struct.unpack('d', f.read(8))[0]
                aperture_size = upper_x_limit - lower_x_limit
                data = np.fromfile(f, dtype=np.float64).reshape((256, 256))

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

def train_simple_cnn():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "data/"
    dataset = DFFRDataset(data_dir, device)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):  # Set appropriate number of epochs
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            outputs = model(labels)  # Pass parameters to the model
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}], Loss: {running_loss / len(dataloader)}')

    torch.save(model.state_dict(), "simple_cnn.pth")

if __name__ == "__main__":
    train_simple_cnn()
