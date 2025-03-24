"""
Module for Burst Coding implementation.

This module implements the burst coding scheme described in
"Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems" by Guo et al.

Burst Coding Overview:
-----------------------
1. Normalize input pixel (0–255) to [0,1].
2. Compute the number of spikes:
       Ns(P) = ceil(Nmax * P)
   where Nmax is the maximum number of spikes (default 5).
3. Compute the inter-spike interval (ISI):
       if Ns > 1:
           ISI(P) = (Tmax - Tmin) * (1 - P) + Tmin
       else:
           ISI(P) = Tmax
   where Tmax is the maximum interval (the processing window) and Tmin is the minimum interval (e.g., 2 ms).
4. For each pixel, generate spikes at times t_i = i * ISI(P) (for i from 0 to Ns(P)-1), as long as t_i is within the simulation duration.
5. Return a binary spike train array of shape (height, width, num_steps).

Usage:
    from burst_coding import BurstCoding
    coder = BurstCoding(dt=0.001, duration=0.01, Nmax=5, Tmin=0.002, Tmax=0.01)
    spike_train = coder.encode(image)
"""

import numpy as np
import matplotlib.pyplot as plt


class BurstCoding:
    """
    Implements burst coding for a 2D image.

    Attributes:
        dt (float): Simulation time step in seconds.
        duration (float): Simulation duration in seconds (set equal to Tmax).
        Nmax (int): Maximum number of spikes in a burst.
        Tmin (float): Minimum inter-spike interval in seconds.
        Tmax (float): Maximum inter-spike interval in seconds.
        num_steps (int): Number of discrete simulation time steps = round(duration / dt).
    """

    def __init__(self, dt=0.001, duration=0.01, Nmax=5, Tmin=0.002, Tmax=0.01):
        """
        Initialize BurstCoding with given parameters.

        Parameters:
            dt (float): Simulation time step (seconds).
            duration (float): Simulation duration (seconds); also used as Tmax.
            Nmax (int): Maximum number of spikes per burst.
            Tmin (float): Minimum inter-spike interval (seconds).
            Tmax (float): Maximum inter-spike interval (seconds).
        """
        self.dt = dt
        self.duration = duration
        self.Nmax = Nmax
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.num_steps = int(np.round(duration / dt))

    def encode(self, image):
        """
        Encode an input image into a burst-coded spike train.

        The input image is expected to be a 2D NumPy array with pixel values in the range [0,255].

        For each pixel:
          1. Normalize to [0,1].
          2. Compute number of spikes: Ns = ceil(Nmax * P).
          3. Compute ISI:
                if Ns > 1: ISI = (Tmax - Tmin) * (1 - P) + Tmin
                else:     ISI = Tmax
          4. For i = 0 to Ns-1, set a spike at time t = i * ISI (if within simulation duration).

        Returns:
            numpy.ndarray: A boolean 3D array of shape (height, width, num_steps),
                           where True indicates a spike occurrence.
        """
        # Normalize image to [0,1]
        norm_image = image.astype(np.float32) / 255.0
        height, width = norm_image.shape
        spike_train = np.zeros((height, width, self.num_steps), dtype=bool)

        for i in range(height):
            for j in range(width):
                P = norm_image[i, j]
                # Compute the number of spikes (if P is 0, then no spike)
                Ns = int(np.ceil(self.Nmax * P))
                if Ns == 0:
                    continue
                # Compute the inter-spike interval (ISI)
                if Ns > 1:
                    # Larger P gives smaller ISI
                    ISI = (self.Tmax - self.Tmin) * (1 - P) + self.Tmin
                else:
                    ISI = self.Tmax
                # Generate spikes at times t = k * ISI for k = 0,..., Ns-1
                for k in range(Ns):
                    spike_time = k * ISI
                    if spike_time < self.duration:
                        time_index = int(np.round(spike_time / self.dt))
                        if time_index < self.num_steps:
                            spike_train[i, j, time_index] = True
        return spike_train


# Example usage / quick test
if __name__ == "__main__":
    # Create a dummy image (e.g., 28x28) with random pixel values between 0 and 255
    dummy_image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)

    # Initialize BurstCoding with Tmax=10ms, Tmin=2ms, Nmax=5
    coder = BurstCoding(dt=0.001, duration=0.01, Nmax=5, Tmin=0.002, Tmax=0.01)

    # Encode the dummy image into a burst spike train
    spike_train = coder.encode(dummy_image)

    # Visualize the burst spike train for a single pixel (e.g., at row 14, column 14)
    pixel_row, pixel_col = 14, 14
    pixel_spikes = spike_train[pixel_row, pixel_col, :].astype(int)

    plt.figure()
    plt.stem(np.arange(len(pixel_spikes)), pixel_spikes, basefmt=" ")
    plt.xlabel("Time step")
    plt.ylabel("Spike (1: spike, 0: no spike)")
    plt.title(f"Burst Spike Train for Pixel ({pixel_row}, {pixel_col})")
    plt.show()
