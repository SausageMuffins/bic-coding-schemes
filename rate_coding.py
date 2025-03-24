"""
Module for Rate Coding implementation.

This module implements the rate coding scheme described in
"Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems" by Guo et al.
The rate coding scheme converts each input pixel into a Poisson spike train.
Input pixels are scaled down by a factor (λ) to produce firing rates confined between 0 and 63.75 Hz.

Usage:
    Import the RateCoding class in your main script:
        from rate_coding import RateCoding
    Create an instance:
        coder = RateCoding(scaling_factor=4, dt=0.001, duration=0.1)
    Encode an image:
        spike_train = coder.encode(image)
"""

import numpy as np


class RateCoding:
    """
    Class for implementing rate coding for input image encoding.

    Attributes:
        scaling_factor (float): Factor to scale down pixel intensities. Default is 4.
        dt (float): Time step in seconds. Default is 0.001 (1 ms).
        duration (float): Duration of the simulation window in seconds. Default is 0.1 (100 ms).
        num_steps (int): Total number of simulation time steps.
    """

    def __init__(self, scaling_factor=4, dt=0.001, duration=0.1):
        """
        Initialize the RateCoding instance.

        Parameters:
            scaling_factor (float): Scaling factor λ for pixel intensities.
            dt (float): Time step in seconds.
            duration (float): Duration of simulation in seconds.
        """
        self.scaling_factor = scaling_factor
        self.dt = dt
        self.duration = duration
        self.num_steps = int(np.round(duration / dt))

    def encode(self, image):
        """
        Encode an input image into a Poisson spike train using rate coding.

        The input image is expected to be a 2D NumPy array with pixel intensities in the range [0, 255].
        Each pixel is scaled down by the scaling factor to obtain a firing rate in Hz.
        A Poisson spike train is generated for each pixel by comparing the firing rate-derived probability with random numbers.

        Parameters:
            image (numpy.ndarray): 2D array representing the input image.

        Returns:
            numpy.ndarray: A binary 3D array of shape (height, width, num_steps) where True indicates a spike occurrence.
        """
        # Ensure the image is in float for computation
        image = image.astype(np.float32)
        # Calculate firing rates in Hz (max firing rate = 255/4 = 63.75 Hz)
        firing_rates = image / self.scaling_factor
        # Compute the probability of spike occurrence per time step:
        # p = firing_rate (Hz) * dt (s)
        spike_prob = firing_rates * self.dt

        # Retrieve the image dimensions
        height, width = image.shape

        # Generate the spike train:
        # For each pixel and each time step, generate a random number and compare with spike probability.
        spike_train = (
            np.random.rand(height, width, self.num_steps) < spike_prob[..., np.newaxis]
        )

        return spike_train


def poisson_spike_train(rate, duration, dt=0.001):
    """
    Generate a Poisson spike train for a given firing rate.

    Parameters:
        rate (float): Firing rate in Hz.
        duration (float): Duration of the spike train in seconds.
        dt (float): Time step in seconds.

    Returns:
        numpy.ndarray: A binary array indicating spike occurrences at each time step.
    """
    num_steps = int(np.round(duration / dt))
    # Compute the per-step spike probability
    p_spike = rate * dt
    # Generate the spike train as a binary array
    spikes = np.random.rand(num_steps) < p_spike
    return spikes


if __name__ == "__main__":
    # Example usage of the RateCoding module.
    import matplotlib.pyplot as plt

    # Create a dummy image (e.g., 28x28) with random pixel intensities between 0 and 255
    dummy_image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)

    # Initialize the RateCoding instance with updated dt=0.0005 for finer detail:
    coder = RateCoding(scaling_factor=4, dt=0.0005, duration=0.1)

    # Encode the dummy image into a spike train
    spike_train = coder.encode(dummy_image)

    # For demonstration, visualize the spike train for a single pixel (e.g., at row 14, column 14)
    pixel_row, pixel_col = 14, 14
    spike_train_pixel = spike_train[pixel_row, pixel_col, :]

    plt.figure()
    plt.stem(np.arange(len(spike_train_pixel)), spike_train_pixel.astype(int))
    plt.xlabel("Time step")
    plt.ylabel("Spike (1: spike, 0: no spike)")
    plt.title(f"Poisson Spike Train for Pixel ({pixel_row}, {pixel_col})")
    plt.show()
