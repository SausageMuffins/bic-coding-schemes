"""
Module for Time-to-First-Spike (TTFS) Coding.

This module implements the TTFS coding scheme described in:
"Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems" by Guo et al.

Time-to-First-Spike (TTFS) Coding:
----------------------------------
1. Normalize each pixel by dividing by the maximum (e.g., 255 for 8-bit images).
2. At discrete simulation step i, the continuous time is t = i * dt (seconds).
3. Compute threshold:
       P_th(t) = theta0 * exp(-t / tau_th) - given in the paper
   where tau_th is the time constant in seconds, theta0 ~ 1.0 by default as mentioned in the paper.
4. A pixel fires its first spike at step i if:
       pixel_value_norm > P_th(i)
   After firing once, that pixel is inhibited (no further spikes).

Usage Example:
    from ttfs_coding import TTFSCoding
    coder = TTFSCoding(dt=0.001, duration=0.02, tau_th=0.006, theta0=1.0)
    spike_train = coder.encode(image)
"""

import numpy as np
import matplotlib.pyplot as plt


class TTFSCoding:
    """
    Implements Time-to-First-Spike coding for a 2D image input.

    Attributes:
        dt (float): Time step in seconds (e.g., 0.001 for 1 ms).
        duration (float): Total simulation time in seconds.
        tau_th (float): Time constant in seconds for exponential threshold decay.
        tau_s (float): Optional spike weighting time constant (e.g., 0.015 for 15 ms).
        theta0 (float): Initial threshold constant (often set to 1.0).
        num_steps (int): Number of discrete time steps = round(duration / dt).
    """

    def __init__(self, dt=0.001, duration=0.02, tau_th=0.006, theta0=1.0, tau_s=0.015):
        """
        Initialize TTFSCoding with user-defined parameters.

        Parameters:
            dt (float): Simulation time step in seconds.
            duration (float): Total duration of the encoding in seconds.
            tau_th (float): Threshold decay time constant (in seconds).
            theta0 (float): Initial threshold constant.
            tau_s (float): Spike weighting time constant (in seconds) for reference if needed.
        """
        self.dt = dt
        self.duration = duration
        self.tau_th = tau_th
        self.theta0 = theta0
        self.tau_s = tau_s  # stored if needed for spike weighting
        self.num_steps = int(np.round(duration / dt))

    def encode(self, image):
        """
        Encode a 2D image into a TTFS spike train.

        1. Normalize the image to [0, 1].
        2. For each time step i, compute the threshold P_th(i) = theta0 * exp(-(i*dt)/tau_th).
        3. Any pixel whose normalized intensity > P_th(i) fires a spike (if it has not fired before).

        Parameters:
            image (numpy.ndarray): 2D array of pixel intensities (e.g., shape [H, W], range [0..255]).

        Returns:
            numpy.ndarray of shape (H, W, num_steps), with boolean entries indicating spike occurrences.
            Typically, each pixel will fire at most once (True at exactly one time step).
        """
        # 1. Normalize pixel values to [0,1] (assuming 8-bit 0..255)
        norm_image = image.astype(np.float32) / 255.0

        height, width = norm_image.shape
        spike_train = np.zeros((height, width, self.num_steps), dtype=bool)

        # Keep track of which pixels have already fired
        has_fired = np.zeros((height, width), dtype=bool)

        # 2. Iterate through each discrete time step
        for step in range(self.num_steps):
            t = step * self.dt  # continuous time in seconds
            # P_th(t) = theta0 * exp(- t / tau_th)
            p_th = self.theta0 * np.exp(-t / self.tau_th)

            # Identify pixels that haven't fired yet and exceed the threshold
            can_fire = (~has_fired) & (norm_image > p_th)
            # Mark those as firing at this step
            spike_train[can_fire, step] = True
            # Inhibit further firing from these pixels
            has_fired[can_fire] = True

        return spike_train


# Example usage / quick test
if __name__ == "__main__":
    # Create a dummy image (28x28) with random values between 0 and 255
    dummy_image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)

    # Initialize TTFS coder with example parameters
    # (Try adjusting dt, duration, tau_th, etc. to see the effect.)
    coder = TTFSCoding(dt=0.001, duration=0.02, tau_th=0.006, theta0=1.0, tau_s=0.015)

    # Encode the dummy image
    spike_train = coder.encode(dummy_image)

    # Choose a pixel to visualize
    row, col = 14, 14
    pixel_spikes = spike_train[row, col, :].astype(int)

    plt.figure()
    plt.stem(np.arange(len(pixel_spikes)), pixel_spikes, basefmt=" ")
    plt.xlabel("Time step")
    plt.ylabel("Spike (1: spike, 0: no spike)")
    plt.title(f"TTFS Spike Train for Pixel ({row}, {col})")
    plt.show()
