import os
import numpy as np
import struct
import matplotlib.pyplot as plt
from tqdm import tqdm

def calc_sym(data_dir, thr=0.001):
    sym_yx = []
    sym_ynegx = []
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dffr')]

    for fp in tqdm(files, desc="Calc sym metrics"):
        with open(fp, 'rb') as f:
            f.seek(104)
            data = np.fromfile(f, dtype=np.float64).reshape((32, 32))
            norm = data / np.max(data)
            lr = np.fliplr(norm)
            ud = np.flipud(norm)
            sym_yx.append(np.sqrt(np.sum(np.abs(norm**2 - lr.T**2))))
            sym_ynegx.append(np.sqrt(np.sum(np.abs(norm**2 - ud.T**2))))

    return np.array(sym_yx), np.array(sym_ynegx)

def plot_hist(data, title, xlabel, ylabel, fname, bins=50):
    counts, edges = np.histogram(data, bins=bins)
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(fname)
    plt.close()

    print(f"\n{title} Histogram\nBins: {bins}\nCounts per bin and ranges:")
    for i in range(len(counts)):
        print(f"Bin {i+1}: Count = {counts[i]}, Range = [{edges[i]}, {edges[i+1]})")
    print("\n")

def calc_stats(data, name):
    mean, med, std = np.mean(data), np.median(data), np.std(data)
    print(f"{name} Stats:\nMean: {mean}\nMedian: {med}\nStd Dev: {std}\n")
    return mean, med, std

def main():
    data_dir = "data/"
    sym_yx, sym_ynegx = calc_sym(data_dir)

    plot_hist(sym_yx, "Sym about y=x", "sym_yx", "Freq", "sym_yx_hist.png")
    plot_hist(sym_ynegx, "Sym about y=-x", "sym_ynegx", "Freq", "sym_ynegx_hist.png")

    calc_stats(sym_yx, "Sym y=x")
    calc_stats(sym_ynegx, "Sym y=-x")

if __name__ == "__main__":
    main()
