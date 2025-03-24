"""
Module for Phase Coding with repeated cycles over the entire simulation duration.

This module implements a "repeated" phase coding scheme described in:
"Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems" by Guo et al.
and inspired by Kim et al. (2018c).

Phase Coding (Repeated Cycles) Overview:
----------------------------------------
1. Each input pixel (0–255) is converted into its 8-bit binary representation (MSB first).
2. Each bit in the binary representation indicates whether the pixel spikes in that phase (bit==1).
3. Instead of producing only 8 time steps, we REPEAT these 8 phases across the entire simulation window.
   - For a total simulation of duration T, with time step dt, we have num_steps = T/dt steps.
   - At each time step t, we compute phase_idx = t % 8.
   - If the pixel’s bit in phase_idx is '1', then we produce a spike at time step t.
4. This yields a dense raster over T steps, where the 8-phase pattern repeats.
5. The spike weight changes with time (phase) periodically according to:
       w_s(t) = 2^-[1 + mod(t-1, 8)]
   but typically used in the decoding phase.

Usage Example:
    from phase_coding import PhaseCoding
    coder = PhaseCoding(num_phases=8, dt=0.001, duration=0.1)
    spike_train = coder.encode(image)
    weights = coder.get_phase_weights()
"""

import numpy as np
import matplotlib.pyplot as plt


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
            num_phases (int): Number of phases to use (8 for 8-bit).
            dt (float): Simulation time step in seconds (default 1 ms).
            duration (float): Total simulation time in seconds (default 0.1 => 100 ms).
        """
        self.num_phases = num_phases
        self.dt = dt
        self.duration = duration
        # total time steps across the entire simulation
        self.num_steps = int(round(duration / dt))

    def encode(self, image):
        """
        Encode an input image into a repeated-phase spike train over the entire duration.

        The input image is a 2D NumPy array with pixel values in [0, 255].
        For each pixel:
          1) Convert to 8-bit binary (MSB first).
          2) For each time step t in [0..num_steps-1], compute phase_idx = t % 8.
             If the bit at phase_idx is '1', produce a spike at time step t.

        Returns:
            numpy.ndarray: Boolean 3D array of shape (height, width, num_steps).
        """
        # Ensure image is uint8
        image = image.astype(np.uint8)
        height, width = image.shape

        # Prepare output: shape (height, width, num_steps)
        spike_train = np.zeros((height, width, self.num_steps), dtype=bool)

        # For each pixel, get its 8-bit binary representation
        for i in range(height):
            for j in range(width):
                binary_str = format(image[i, j], "08b")  # MSB first
                bits = np.array([char == "1" for char in binary_str], dtype=bool)
                # Repeated-phase approach
                for t in range(self.num_steps):
                    phase_idx = t % self.num_phases
                    if bits[phase_idx]:
                        spike_train[i, j, t] = True

        return spike_train

    def get_phase_weights(self):
        """
        Compute the weights for each phase in a single cycle:
            w_s(t) = 2^-[1 + mod(t-1, 8)].
        Typically repeated across the entire duration if needed for decoding.

        For phases 1..8, yields [2^-1, 2^-2, ..., 2^-8].

        Returns:
            numpy.ndarray: 1D array of length num_phases with the phase weights.
        """
        weights = np.array([2 ** (-(phase)) for phase in range(1, self.num_phases + 1)])
        return weights


# Example usage / quick test
if __name__ == "__main__":
    # Create a dummy image (28x28) with random values in [0, 255]
    dummy_image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)

    # Initialize PhaseCoding with 8 phases, dt=1 ms, total duration=100 ms
    coder = PhaseCoding(num_phases=8, dt=0.001, duration=0.1)

    # Encode the dummy image
    spike_train = coder.encode(dummy_image)

    # Print shape => (28, 28, 100)
    print("Spike train shape:", spike_train.shape)

    # Retrieve phase weights
    phase_weights = coder.get_phase_weights()
    print("Phase weights:", phase_weights)

    # Visualize the spike train for a single pixel
    row, col = 14, 14
    pixel_spikes = spike_train[row, col, :].astype(int)

    plt.figure()
    plt.stem(np.arange(len(pixel_spikes)), pixel_spikes, basefmt=" ")
    plt.xlabel("Time step")
    plt.ylabel("Spike (1: spike, 0: no spike)")
    plt.title(f"Repeated-Phase Spike Train for Pixel ({row}, {col})")
    plt.show()
