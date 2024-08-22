import torch

def beta_schedule(T, beta_start, beta_end):
    betas = torch.linspace(beta_start, beta_end, T)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    return betas, alpha_bars

T = 100
beta_start, beta_end = 0.001, 0.2

b, a_bars = beta_schedule(T, beta_start, beta_end)

print("Betas:", b)
print("Alpha Bars:", a_bars)
print(f"Final alpha_bar: {a_bars[-1].item()}")
