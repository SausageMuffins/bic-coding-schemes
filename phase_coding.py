"""
This script implements the phase coding scheme described in "Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems" by Guo et al.

Phase Coding:
----------------------
1. Each pixel is first converted to its 8-bit binary representation (MSB first).
2. The simulation time is discretized into steps of size dt, over a total duration.
3. Within one cycle of length `num_phases`, each bit corresponds to one phase index.
   - If the bit is '1', a spike is generated at each time step where (t % num_phases) equals that bit's position.
   - If the bit is '0', no spike is generated at that phase.
4. The same cycle is repeated across the entire simulation duration.

Usage Example:
    from phase_coding import PhaseCoding

    phase_coder = PhaseCoding(num_phases=8, dt=0.001, duration=0.1)
    spike_train = phase_coder.encode(image)

    # spike_train has shape (height, width, num_steps)
    # where each (height, width) location has a boolean spike train over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Set global font properties for large fonts
plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    }
)


class PhaseCoding:
    """
    Implements a repeated-cycle phase coding for a 2D image.

    Each pixel is converted into its 8-bit binary representation (MSB first).
    The 8-phase pattern is then repeated across the entire simulation duration,
    resulting in a spike train of shape (height, width, num_steps).
    """

    def __init__(self, num_phases=8, dt=0.001, duration=0.1):
        """
        Initialize the PhaseCoding instance.

        Parameters:
            num_phases (int): Number of phases to use (default 8).
            dt (float): Simulation time step in seconds (default 1 ms).
            duration (float): Total simulation time in seconds (default 0.1 => 100 ms).
        """
        self.num_phases = num_phases
        self.dt = dt
        self.duration = duration
        self.num_steps = int(round(duration / dt))

    def encode(self, image):
        """
        Encode an input image into a repeated-phase spike train over the entire duration.

        The input image is a 2D NumPy array with pixel values in [0, 255].
        For each pixel:
          1) Convert to 8-bit binary (MSB first).
          2) For each time step t, compute phase_idx = t % num_phases.
             If the bit at phase_idx is '1', produce a spike at time step t.

        Returns:
            numpy.ndarray: Boolean 3D array of shape (height, width, num_steps).
        """
        image = image.astype(np.uint8)
        height, width = image.shape
        spike_train = np.zeros((height, width, self.num_steps), dtype=bool)
        for i in range(height):
            for j in range(width):
                binary_str = format(image[i, j], "08b")  # 8-bit binary, MSB first
                bits = np.array([char == "1" for char in binary_str], dtype=bool)
                for t in range(self.num_steps):
                    phase_idx = t % self.num_phases
                    if bits[phase_idx]:
                        spike_train[i, j, t] = True
        return spike_train

    def get_phase_weights(self):
        """
        Compute the weights for each phase in a single cycle:
            w_s(t) = 2^(-phase) for phases 1..8.

        Returns:
            numpy.ndarray: 1D array of length num_phases with the phase weights.
        """
        weights = np.array([2 ** (-(phase)) for phase in range(1, self.num_phases + 1)])
        return weights


# ------------------- Main Script -------------------
if __name__ == "__main__":
    # Load MNIST dataset
    (x_train, y_train), (_, _) = mnist.load_data()
    print("Loaded MNIST training set:", x_train.shape)

    # Choose four digits: 0, 5, 7, and 9.
    target_digits = [0, 5, 7, 9]
    selected_images = []
    for d in target_digits:
        indices = np.where(y_train == d)[0]
        if len(indices) > 0:
            selected_images.append(x_train[indices[0]])
        else:
            print(f"Digit {d} not found in training set.")

    # Initialize PhaseCoding
    phase_coder = PhaseCoding(num_phases=8, dt=0.001, duration=0.1)

    # Prepare figure with 2 rows and 4 columns
    fig, axs = plt.subplots(2, len(target_digits), figsize=(16, 12))

    # Time axis in ms (same for all, since dt and duration are fixed)
    T = phase_coder.num_steps
    time_axis = np.linspace(0, phase_coder.duration, T, endpoint=True) * 1000  # in ms

    # Loop over each selected digit and plot its raster and average spike count
    for idx, image in enumerate(selected_images):
        # Encode the image with phase coding
        spike_train = phase_coder.encode(image)
        H, W, _ = spike_train.shape
        # Raster plot: Flatten the spike train.
        neuron_ids = []
        spike_times = []
        for i in range(H):
            for j in range(W):
                spike_indices = np.where(spike_train[i, j, :])[0]
                if spike_indices.size > 0:
                    # Each pixel is treated as a separate neuron.
                    neuron_index = i * W + j
                    spike_times.extend(time_axis[spike_indices])
                    neuron_ids.extend([neuron_index] * len(spike_indices))

        axs[0, idx].scatter(spike_times, neuron_ids, s=2, color="black")
        axs[0, idx].set_title(f"Digit {target_digits[idx]}")
        axs[0, idx].set_xlabel("Time (ms)")
        axs[0, idx].set_ylabel("Input neuron index")
        axs[0, idx].set_xlim(0, phase_coder.duration * 1000)

        # Compute average spike counts vs. time for this image.
        # For each time step, average over all pixels.
        avg_spikes = spike_train.sum(axis=(0, 1)) / (H * W)
        axs[1, idx].plot(time_axis, avg_spikes, color="blue")
        axs[1, idx].set_title(f"Avg Spike Count vs Time (Digit {target_digits[idx]})")
        axs[1, idx].set_xlabel("Time (ms)")
        axs[1, idx].set_ylabel("Avg spike count per pixel")
        axs[1, idx].set_xlim(0, phase_coder.duration * 1000)

    plt.tight_layout()
    plt.show()
