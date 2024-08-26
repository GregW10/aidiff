import sys
import numpy as np
import matplotlib.pyplot as plt
from sampleANDcompare32_nolog import l2_true_distance, l2_norm_distance, load_dffr

def plot_comparison(model_dffr_path, dffrcc_dffr_path):
    # Load the patterns from the .dffr files
    model_pattern = load_dffr(model_dffr_path)
    dffrcc_pattern = load_dffr(dffrcc_dffr_path)

    # Calculate L2 distances
    true_distance = l2_true_distance(model_pattern, dffrcc_pattern)
    norm_distance = l2_norm_distance(model_pattern, dffrcc_pattern)

    # Generate the comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(model_pattern, cmap="gray")
    axes[0].set_title("Model-Generated")
    axes[1].imshow(dffrcc_pattern, cmap="gray")
    axes[1].set_title("dffrcc-Generated")

    plt.suptitle(f"L2 True Distance: {true_distance:.3e}, L2 Norm. Distance: {norm_distance:.3e}")
    plt.show()

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_dffr.py <model_dffr_path> <dffrcc_dffr_path>")
        sys.exit(1)

    model_dffr_path = sys.argv[1]
    dffrcc_dffr_path = sys.argv[2]

    plot_comparison(model_dffr_path, dffrcc_dffr_path)

if __name__ == "__main__":
    main()
