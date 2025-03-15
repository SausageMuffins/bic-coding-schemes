"""
main.py

This script replicates Figure 3 from:
"Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems" by Guo et al.

- Top row: Input spike patterns (raster plots) of an example input digit "5" for different coding methods in a 100 ms time window.
- Bottom row: Average input spike counts over time across all the training input images (MNIST) for different coding methods in a 100 ms time window.

Coding methods: Rate, TTFS, Phase, Burst
"""

import numpy as np
import matplotlib.pyplot as plt

# Import coding scheme classes
from rate_coding import RateCoding
from time_to_spike_coding import TTFSCoding
from phase_coding import PhaseCoding
from burst_coding import BurstCoding

# Load MNIST dataset (Keras)
from tensorflow.keras.datasets import mnist

# --- Parameters ---
DURATION = 0.1  # 100 ms
DT = 0.001  # 1 ms time step
NUM_TRAIN_MAX = (
    None  # Optionally limit the number of training samples for speed (e.g., 5000)
)

# Initialize coders
rate_coder = RateCoding(scaling_factor=4, dt=DT, duration=DURATION)
ttfs_coder = TTFSCoding(dt=DT, duration=DURATION, tau_th=0.006, theta0=1.0)
phase_coder = PhaseCoding(num_phases=8)
burst_coder = BurstCoding(dt=DT, duration=DURATION, Nmax=5, Tmin=0.002, Tmax=DURATION)

# Load MNIST
print("Loading MNIST dataset...")
(x_train, y_train), (_, _) = mnist.load_data()

# (Optional) reduce training set size for quicker demonstration
if NUM_TRAIN_MAX is not None and NUM_TRAIN_MAX < x_train.shape[0]:
    x_train = x_train[:NUM_TRAIN_MAX]
    y_train = y_train[:NUM_TRAIN_MAX]

# --- 1) Example digit "5" for top row raster plot ---
example_idx = np.where(y_train == 5)[0][0]
example_image = x_train[example_idx]
print(
    f"Using training image index {example_idx} (digit {y_train[example_idx]}) for example spike pattern."
)

coding_methods = {
    "Rate": rate_coder,
    "TTFS": ttfs_coder,
    "Phase": phase_coder,
    "Burst": burst_coder,
}


# --- Helper function to compute average spike counts per time step across the entire dataset ---
def compute_avg_spike_counts_over_time(coder, images):
    """
    For each image, encode -> sum spikes across all pixels at each time step -> accumulate.
    Then average over the number of images.
    Returns:
        avg_spike_counts (1D np.array): average number of spikes at each time step
        time_axis (1D np.array): corresponding time (in seconds) for each time step
    """
    # We compute the shape from a single encoding
    sample_spike_train = coder.encode(images[0])
    H, W, T = sample_spike_train.shape
    sum_spikes_time = np.zeros(T, dtype=float)

    for img in images:
        spike_train = coder.encode(img)  # shape (H, W, T)
        # sum across spatial dims => shape (T,)
        sum_over_pixels = spike_train.sum(axis=(0, 1))
        sum_spikes_time += sum_over_pixels

    # Average across all images
    sum_spikes_time /= len(images)

    # Build a time axis
    if isinstance(coder, PhaseCoding):
        # Phase coding: T = num_phases; map them evenly to [0, DURATION)
        time_axis = np.linspace(0, DURATION, T, endpoint=False)
    else:
        # Others: T = int(DURATION / DT)
        time_axis = np.arange(T) * coder.dt

    return sum_spikes_time, time_axis


# --- Collect data for plotting ---
fig, axs = plt.subplots(2, len(coding_methods), figsize=(14, 6))

for col_idx, (method_name, coder) in enumerate(coding_methods.items()):
    # 1) Top row: Raster plot of example digit "5"
    spike_train_example = coder.encode(example_image)
    H, W, T_ex = spike_train_example.shape

    # Build a time axis for the example digit
    if isinstance(coder, PhaseCoding):
        time_axis_ex = np.linspace(0, DURATION, T_ex, endpoint=False)
    else:
        time_axis_ex = np.arange(T_ex) * coder.dt

    # Flatten the spatial dimension and record spike times
    neuron_ids = []
    spike_times = []
    for i in range(H):
        for j in range(W):
            idx_spikes = np.where(spike_train_example[i, j, :])[0]
            if idx_spikes.size > 0:
                times = time_axis_ex[idx_spikes]
                neuron_index = i * W + j
                neuron_ids.extend([neuron_index] * len(times))
                spike_times.extend(times)

    axs[0, col_idx].scatter(spike_times, neuron_ids, s=1, c="black")
    axs[0, col_idx].set_title(f"{method_name} coding\nExample digit '5'")
    axs[0, col_idx].set_xlabel("Time (s)")
    axs[0, col_idx].set_ylabel("Input neuron index")
    axs[0, col_idx].set_xlim(0, DURATION)
    axs[0, col_idx].set_ylim(0, H * W)

    # 2) Bottom row: Average spike counts over time (across entire training set)
    avg_spike_counts, time_axis = compute_avg_spike_counts_over_time(coder, x_train)
    # Convert time axis to ms for a more direct match to the figure
    time_axis_ms = time_axis * 1000.0

    axs[1, col_idx].plot(time_axis_ms, avg_spike_counts, color="blue")
    axs[1, col_idx].set_title("Avg spike counts vs time\n(All training images)")
    axs[1, col_idx].set_xlabel("Time (ms)")
    axs[1, col_idx].set_ylabel("Average spike count")
    axs[1, col_idx].set_xlim(0, DURATION * 1000)

plt.tight_layout()
plt.show()
