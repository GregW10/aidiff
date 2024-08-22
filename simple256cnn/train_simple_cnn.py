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
from main32_nolog import DFFRDataset
import signal
import sys


model = None
optimiser = None

def save_model(model, filename="model_simple_cnn.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to \"{filename}\"")

def signal_handler(sig, frame):
    print(f"Received signal {sig}. Saving model and exiting...")
    if model:
        save_model(model)
    sys.exit(0)


def train_simple_cnn():
    global model, optimiser
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "data/"
    nx = 256
    ny = 256
    sym_thresh = 0.05
    dataset = DFFRDataset(data_dir, device, Nx=nx, Ny=ny, threshold=sym_thresh)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleCNN().to(device)
    if len(sys.argv) > 1:
        model.load_state_dict(torch.load(sys.argv[1]))
        print(f"Loaded model \"{sys.argv[1]}\".")
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=1e-5)

    num_epochs = 1_000_000

    prog_bar = tqdm if os.isatty(sys.stdout.fileno()) else lambda x : x

    dlen = len(dataloader)

    cf = open("losses.csv", 'w')
    cf.write("epoch,iteration,loss\n")

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        for i, (inputs, params) in enumerate(prog_bar(dataloader), start=1):
            optimiser.zero_grad()
            outputs = model(params)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            cf.write(f"{epoch},{i},{loss.item()}\n")
            cf.flush()

        print(f"Epoch [{epoch + 1}], Loss: {running_loss/dlen}", flush=True)

    torch.save(model.state_dict(), "simple_cnn.pth")

if __name__ == "__main__":
    train_simple_cnn()
