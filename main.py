"""
This script replicates Figure 3 from:
"Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems" by Guo et al.

- Top row: Input spike patterns (raster plots) of an example input digit "5" for different coding methods in a 100 ms time window.
- Bottom row: Average input spike counts over time across all the training input images (MNIST) for different coding methods in a 100 ms time window.

Coding methods: Rate, TTFS, Phase, Burst

We compare the following with the paper:
    - Total spike count in 100 ms for example digit "5"
    - Average input spike counts vs time (and peak)
    - Latest spike time (where relevant)

Paper references (for example digit '5'):
  Rate:   total spikes ~689,   average spike count ~4,   random pattern
  TTFS:   total spikes ~166,   latest spike ~20 ms
  Phase:  total spikes ~20325, periodic patterns
  Burst:  total spikes ~598,   latest spike ~20 ms

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
NUM_TRAIN_MAX = None  # Optionally limit training samples (e.g., 5000)

# Example parameter initialization:
rate_coder = RateCoding(scaling_factor=4, dt=0.00075, duration=DURATION)
ttfs_coder = TTFSCoding(dt=DT, duration=DURATION, tau_th=0.006, theta0=1.0)
phase_coder = PhaseCoding(num_phases=8)
burst_coder = BurstCoding(dt=DT, duration=DURATION, Nmax=5, Tmin=0.002, Tmax=0.01)

# Load MNIST
print("Loading MNIST dataset...")
(x_train, y_train), (_, _) = mnist.load_data()
if NUM_TRAIN_MAX is not None and NUM_TRAIN_MAX < x_train.shape[0]:
    x_train = x_train[:NUM_TRAIN_MAX]
    y_train = y_train[:NUM_TRAIN_MAX]

# Pick an example digit "5"
example_idx = np.where(y_train == 5)[0][0]
example_image = x_train[example_idx]
print(
    f"Using training image index {example_idx} (digit {y_train[example_idx]}) for example spike pattern.\n"
)

coding_methods = {
    "Rate": rate_coder,
    "TTFS": ttfs_coder,
    "Phase": phase_coder,
    "Burst": burst_coder,
}

# Reference numbers from the paper’s paragraph
paper_reference = {
    "Rate": {
        "total_spikes_example": 689,  # total count in 100 ms
        "latest_spike_ms": None,  # not explicitly stated
        "notes": "Mean ~4, random pattern",
    },
    "TTFS": {
        "total_spikes_example": 166,
        "latest_spike_ms": 20.0,
        "notes": "Each input neuron fires one spike, all before 20 ms",
    },
    "Phase": {
        "total_spikes_example": 20325,
        "latest_spike_ms": None,  # Not explicitly stated, but presumably up to 100 ms
        "notes": "Periodic spike pattern",
    },
    "Burst": {
        "total_spikes_example": 598,
        "latest_spike_ms": 20.0,
        "notes": "Burst frequency depends on pixel intensity",
    },
}


def compute_avg_spike_counts_over_time(coder, images):
    """
    For each image, encode -> sum spikes across all pixels at each time step -> accumulate.
    Then average over the number of images.

    Returns:
        avg_spike_counts (1D np.array): average number of spikes at each time step
        time_axis (1D np.array): corresponding time (in seconds) for each time step
    """
    sample_spike_train = coder.encode(images[0])
    H, W, T = sample_spike_train.shape
    sum_spikes_time = np.zeros(T, dtype=float)

    for img in images:
        spike_train = coder.encode(img)
        sum_over_pixels = spike_train.sum(axis=(0, 1))
        sum_spikes_time += sum_over_pixels

    sum_spikes_time /= len(images)

    if isinstance(coder, PhaseCoding):
        # Phase coding: T = num_phases; map them to [0, DURATION]
        time_axis = np.linspace(0, coder.duration, T, endpoint=True)
    else:
        time_axis = np.arange(T) * coder.dt

    return sum_spikes_time, time_axis


def compute_spike_metrics(coder, images):
    """
    Compute total and average spike counts across all images in the dataset.
    """
    total_spikes = 0
    for img in images:
        spike_train = coder.encode(img)
        total_spikes += spike_train.sum()
    avg_spikes_per_image = total_spikes / len(images)
    return total_spikes, avg_spikes_per_image


fig, axs = plt.subplots(2, len(coding_methods), figsize=(14, 6))

for col_idx, (method_name, coder) in enumerate(coding_methods.items()):
    # Encode the example digit
    spike_train_example = coder.encode(example_image)
    H, W, T_ex = spike_train_example.shape

    # Time axis
    if isinstance(coder, PhaseCoding):
        time_axis_ex = np.linspace(0, coder.duration, T_ex, endpoint=True)
    else:
        time_axis_ex = np.arange(T_ex) * coder.dt
    time_axis_ex_ms = time_axis_ex * 1000.0

    # Flatten for raster
    neuron_ids = []
    spike_times_ms = []
    for i in range(H):
        for j in range(W):
            idx_spikes = np.where(spike_train_example[i, j, :])[0]
            if idx_spikes.size > 0:
                times = time_axis_ex_ms[idx_spikes]
                neuron_index = i * W + j
                neuron_ids.extend([neuron_index] * len(times))
                spike_times_ms.extend(times)

    # Raster plot
    axs[0, col_idx].scatter(spike_times_ms, neuron_ids, s=1, c="black")
    axs[0, col_idx].set_title(f"{method_name} coding\nExample digit '5'")
    axs[0, col_idx].set_xlabel("Time (ms)")
    axs[0, col_idx].set_ylabel("Input neuron index")
    axs[0, col_idx].set_xlim(0, 100)
    axs[0, col_idx].set_ylim(0, H * W)

    # Additional metrics for example digit
    total_spikes_example = spike_train_example.sum()
    latest_spike_time = max(spike_times_ms) if len(spike_times_ms) > 0 else 0

    print(f"[{method_name}] Example digit '5':")
    print(f"  Measured total spikes in 100 ms = {total_spikes_example}")
    print(
        f"  Paper reference total spikes    = {paper_reference[method_name]['total_spikes_example']}"
    )
    diff = total_spikes_example - paper_reference[method_name]["total_spikes_example"]
    print(f"  Difference (measured - paper)   = {diff}\n")

    # Compare latest spike time if the paper mentions it
    if paper_reference[method_name]["latest_spike_ms"] is not None:
        print(f"  Measured latest spike time      = {latest_spike_time:.2f} ms")
        ref_latest = paper_reference[method_name]["latest_spike_ms"]
        print(f"  Paper reference latest spike    = {ref_latest} ms")
        print(
            f"  Difference (measured - paper)   = {latest_spike_time - ref_latest:.2f} ms\n"
        )
    else:
        print(f"  (Paper does not specify a latest spike time for {method_name}.)\n")

    # Plot average spike counts over time for entire training set
    avg_spike_counts, time_axis = compute_avg_spike_counts_over_time(coder, x_train)
    time_axis_ms = time_axis * 1000.0
    axs[1, col_idx].plot(time_axis_ms, avg_spike_counts, color="blue")
    axs[1, col_idx].set_title("Avg spike counts vs time\n(All training images)")
    axs[1, col_idx].set_xlabel("Time (ms)")
    axs[1, col_idx].set_ylabel("Average spike count")
    axs[1, col_idx].set_xlim(0, coder.duration * 1000)

    # Peak average spike count
    peak_idx = np.argmax(avg_spike_counts)
    peak_count = avg_spike_counts[peak_idx]
    peak_time_ms = time_axis_ms[peak_idx]

    # Print peak info
    print(
        f"  Peak avg spike count (training set) = {peak_count:.2f} at {peak_time_ms:.2f} ms"
    )

    # Compute total spikes across entire training set
    total_spikes_all, avg_spikes_per_image = compute_spike_metrics(coder, x_train)
    print(f"  Total spikes (training set)         = {total_spikes_all}")
    print(f"  Avg spikes per image (training set) = {avg_spikes_per_image:.2f}\n")
    print("----------------------------------------------------------------\n")

plt.tight_layout()
plt.show()
