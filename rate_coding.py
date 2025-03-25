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
import math


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
        image = image.astype(np.float32)
        # Calculate firing rates in Hz (max firing rate = 255/4 = 63.75 Hz)
        firing_rates = image / self.scaling_factor
        # Compute the probability of spike occurrence per time step: p = firing_rate * dt
        spike_prob = firing_rates * self.dt
        height, width = image.shape
        # Generate the spike train using a random thresholding method
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
    p_spike = rate * dt
    spikes = np.random.rand(num_steps) < p_spike
    return spikes


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ------------------------------ Part 1: Vary dt and plot average spikes ------------------------------
    # Create a dummy image (e.g., 28x28) with random pixel intensities between 0 and 255
    dummy_image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)

    # Vary dt from 0.0001 to 0.001 in steps of 0.0001
    dt_values = np.arange(0.0001, 0.001 + 1e-6, 0.0001)
    avg_spikes_per_pixel = []
    for dt in dt_values:
        coder = RateCoding(scaling_factor=4, dt=dt, duration=0.1)
        spike_train = coder.encode(dummy_image)
        # Compute the average spike count per pixel
        avg_spikes = spike_train.sum() / (dummy_image.shape[0] * dummy_image.shape[1])
        avg_spikes_per_pixel.append(avg_spikes)

    plt.figure()
    plt.plot(dt_values, avg_spikes_per_pixel, marker="o")
    plt.xlabel("Time step dt (s)")
    plt.ylabel("Average spikes per pixel")
    plt.title("Average Number of Spikes vs. Time Step dt")
    plt.grid(True)
    plt.show()

    # ------------------------------ Part 2: Poisson Distribution Plots ------------------------------
    # Plot the theoretical Poisson distribution of spike counts
    n_values = np.arange(0, 21)  # n = 0 to 20
    plt.figure()
    for avg_n in [1, 4, 10]:
        # Calculate Poisson probability: P(n) = (λ^n * exp(-λ)) / n!
        probabilities = [
            (avg_n**n / math.factorial(n)) * np.exp(-avg_n) for n in n_values
        ]
        plt.plot(n_values, probabilities, marker="o", label=f"<n> = {avg_n}")
    plt.xlabel("Number of spikes (n)")
    plt.ylabel("Probability P(n)")
    plt.title("Poisson Distribution of Spike Counts")
    plt.legend()
    plt.grid(True)
    plt.show()
