import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from unet import UNet
from tqdm import tqdm
import os
import signal
import sys
from itertools import cycle
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DataProvider:
    def __init__(self, data):
        self.data = data
        
    def sample_batch(self, batch_size=64):
        return self.data[:batch_size]

class DiffusionModel:
    def __init__(self, T: int, model: nn.Module, width: int = 32, height: int = 32, num_channels: int = 1, device: str = "cpu"):
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
    
    def handle_signal(self, signum, frame):
        print(f"Received signal {signum}. Saving model...")
        self.save_model()
        sys.exit(0)
    
    def save_model(self):
        torch.save(self.model.state_dict(), "interrupted_model.pth")
        print(f"Model saved.")
    
    def train(self, dataloader, optimiser, num_epochs=10_000, csv_f="losses.csv", model_path="model.pth"):
        cf = open(csv_f, "w")
        cf.write("epoch,loss\n")
        
        pfunc = tqdm if os.isatty(sys.stdout.fileno()) else lambda x: x
        
        for epoch in pfunc(range(1, num_epochs + 1)):
            x_0, digits = next(dataloader)
            x_0 = x_0.to(self.device)
            digits = digits.to(self.device)
            t = torch.randint(low=1, high=self.T+1, size=(x_0.size(0),)).to(self.device)
            eps = torch.randn_like(x_0).to(self.device)
            x_t = torch.sqrt(self.bar_alphas[t-1, None, None, None])*x_0 + torch.sqrt(1 - self.bar_alphas[t-1, None, None, None])*eps
            loss = torch.nn.functional.mse_loss(self.model(x_t, t, digits), eps).to(self.device)
            
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}", flush=True)
            
            cf.write(f"{epoch},{loss.item()}\n")
        
        cf.close()
        optimiser.zero_grad()
        torch.save(self.model.state_dict(), model_path)
    
    @torch.no_grad
    def sample(self, num_samples=1, digits=None):
        if digits is None:
            digits = torch.randint(0, 10, (num_samples,)).to(self.device)
        x = torch.randn((num_samples, self.num_channels, self.height, self.width)).to(self.device)
        pfunc = tqdm if os.isatty(sys.stdout.fileno()) else lambda x: x
        
        for t in pfunc(range(self.T, 0, -1)):
            z = torch.randn_like(x).to(self.device) if t > 1 else torch.zeros_like(x).to(self.device)
            x = (1/torch.sqrt(self.alphas[t-1, None, None, None]) * \
                 (x - ((1 - self.alphas[t-1, None, None, None])/torch.sqrt(1 - self.bar_alphas[t-1, None, None, None])) * self.model(x, torch.Tensor([t]*num_samples).to(self.device), digits)) + \
                 torch.sqrt(self.betas[t-1, None, None, None])*z)
        return x, digits

def main():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    mnist = torchvision.datasets.MNIST(root=".", download=True, transform=transform)
    
    batch_size = 64
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
    
    # Create an infinite iterator from the DataLoader
    infinite_dataloader = cycle(dataloader)
    
    T = 1_000
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = "cpu"
    
    model = UNet()
    
    if len(sys.argv) > 1:
        model.load_state_dict(torch.load(sys.argv[1]))
        
    diff_model = DiffusionModel(T, model.to(device))

    lr = 1e-5
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 100_000
    
    diff_model.train(infinite_dataloader, optim, num_epochs=epochs)
    
    num_samples = 20
    samples, digits = diff_model.sample(num_samples)
    
    counter = 0
    while counter < num_samples:
        plt.imshow(samples[counter][0].cpu().numpy(), cmap="gray")
        plt.title(f"Digit: {digits[counter].item()}")
        plt.savefig(f"sample{counter}.png")
        plt.clf()
        counter += 1

if __name__ == "__main__":
    main()
