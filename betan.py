import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_data(path):
    return np.fromfile(path, dtype=np.float32)


def plot_data(arr1, arr2, prefix, out_dir):
    # Calculate the global min and max
    global_min = min(np.min(arr1), np.min(arr2))
    global_max = max(np.max(arr1), np.max(arr2))
    
    # Calculate the range and add a 10% margin
    y_range = global_max - global_min
    margin = y_range * 0.1
    global_min -= margin
    global_max += margin

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(arr1)
    plt.title(f'{prefix} First Array')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.ylim(global_min, global_max)  # Set the y-limits with margin

    plt.subplot(1, 2, 2)
    plt.plot(arr2)
    plt.title(f'{prefix} Last Array')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.ylim(global_min, global_max)  # Set the y-limits with margin

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_plots.png'))
    plt.close()


def calc_metrics(arr1, arr2):
    dist = np.linalg.norm(arr1 - arr2)
    mean1 = np.mean(arr1)
    mean2 = np.mean(arr2)
    return dist, mean1, mean2


def calc_addl_metrics(arr1, arr2):
    std1 = np.std(arr1)
    std2 = np.std(arr2)
    med1 = np.median(arr1)
    med2 = np.median(arr2)
    return std1, std2, med1, med2


def find_min_max(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    min_idx = np.argmin(arr)
    max_idx = np.argmax(arr)
    return min_val, max_val, min_idx, max_idx


def main():
    parser = argparse.ArgumentParser(description="Analyse Betas and Rawbetas Files")
    parser.add_argument("dir", type=str, help="Directory containing the betas and rawbetas files")
    parser.add_argument("--plot", action="store_true", help="Plot and save the arrays")
    args = parser.parse_args()

    betas = sorted([f for f in os.listdir(args.dir) if f.endswith('.betas')])
    rbetas = sorted([f for f in os.listdir(args.dir) if f.endswith('.rbetas')])

    if not betas or not rbetas:
        print("No .betas or .rbetas files found in the directory.")
        return

    print(f"First Betas File: {betas[0]}")
    print(f"Last Betas File: {betas[-1]}")
    print(f"First Rawbetas File: {rbetas[0]}")
    print(f"Last Rawbetas File: {rbetas[-1]}")

    beta1 = load_data(os.path.join(args.dir, betas[0]))
    beta2 = load_data(os.path.join(args.dir, betas[-1]))
    
    rbeta1 = load_data(os.path.join(args.dir, rbetas[0]))
    rbeta2 = load_data(os.path.join(args.dir, rbetas[-1]))

    beta_dist, beta_mean1, beta_mean2 = calc_metrics(beta1, beta2)
    rbeta_dist, rbeta_mean1, rbeta_mean2 = calc_metrics(rbeta1, rbeta2)

    print(f"Betas Distance: {beta_dist:.6f}")
    print(f"Rawbetas Distance: {rbeta_dist:.6f}")
    print(f"First Betas Mean: {beta_mean1:.6f}")
    print(f"Last Betas Mean: {beta_mean2:.6f}")
    print(f"First Rawbetas Mean: {rbeta_mean1:.6f}")
    print(f"Last Rawbetas Mean: {rbeta_mean2:.6f}")
    
    std1, std2, med1, med2 = calc_addl_metrics(beta1, beta2)
    rstd1, rstd2, rmed1, rmed2 = calc_addl_metrics(rbeta1, rbeta2)

    print(f"First Betas Std Dev: {std1:.6f}")
    print(f"Last Betas Std Dev: {std2:.6f}")
    print(f"First Betas Median: {med1:.6f}")
    print(f"Last Betas Median: {med2:.6f}")
    print(f"First Rawbetas Std Dev: {rstd1:.6f}")
    print(f"Last Rawbetas Std Dev: {rstd2:.6f}")
    print(f"First Rawbetas Median: {rmed1:.6f}")
    print(f"Last Rawbetas Median: {rmed2:.6f}")

    beta_min1, beta_max1, min_idx1, max_idx1 = find_min_max(beta1)
    beta_min2, beta_max2, min_idx2, max_idx2 = find_min_max(beta2)
    rbeta_min1, rbeta_max1, rmin_idx1, rmax_idx1 = find_min_max(rbeta1)
    rbeta_min2, rbeta_max2, rmin_idx2, rmax_idx2 = find_min_max(rbeta2)

    print(f"First Betas Min: {beta_min1:.6f} at index {min_idx1}")
    print(f"First Betas Max: {beta_max1:.6f} at index {max_idx1}")
    print(f"Last Betas Min: {beta_min2:.6f} at index {min_idx2}")
    print(f"Last Betas Max: {beta_max2:.6f} at index {max_idx2}")
    print(f"First Rawbetas Min: {rbeta_min1:.6f} at index {rmin_idx1}")
    print(f"First Rawbetas Max: {rbeta_max1:.6f} at index {rmax_idx1}")
    print(f"Last Rawbetas Min: {rbeta_min2:.6f} at index {rmin_idx2}")
    print(f"Last Rawbetas Max: {rbeta_max2:.6f} at index {rmax_idx2}")

    if args.plot:
        out_dir = os.path.join(args.dir, 'plots')
        os.makedirs(out_dir, exist_ok=True)
        plot_data(beta1, beta2, "Betas", out_dir)
        plot_data(rbeta1, rbeta2, "Rawbetas", out_dir)
        print(f"Plots saved in {out_dir}")


if __name__ == "__main__":
    main()
