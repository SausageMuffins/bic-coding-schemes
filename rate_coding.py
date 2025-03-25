import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


class RateCoding:
    """
    Class for implementing rate coding for input image encoding.

    Attributes:
        scaling_factor (float): Factor to scale down pixel intensities.
        dt (float): Time step in seconds.
        duration (float): Duration of the simulation window in seconds.
        num_steps (int): Total number of simulation time steps.
    """

    def __init__(self, scaling_factor=4, dt=0.001, duration=0.1):
        self.scaling_factor = scaling_factor
        self.dt = dt
        self.duration = duration
        self.num_steps = int(np.round(duration / dt))

    def encode(self, image):
        """
        Encode an input image into a Poisson spike train using rate coding.

        Parameters:
            image (numpy.ndarray): 2D array with pixel intensities in [0, 255].

        Returns:
            numpy.ndarray: A binary 3D array of shape (height, width, num_steps)
                           where True indicates a spike occurrence.
        """
        image = image.astype(np.float32)
        firing_rates = (
            image / self.scaling_factor
        )  # e.g., max ~63.75 Hz if pixel=255 and λ=4
        spike_prob = firing_rates * self.dt
        h, w = image.shape
        spike_train = np.random.rand(h, w, self.num_steps) < spike_prob[..., np.newaxis]
        return spike_train


if __name__ == "__main__":
    # 1) Load MNIST dataset
    (x_train, y_train), (_, _) = mnist.load_data()
    print("Loaded MNIST training set:", x_train.shape)

    # 2) Define λ values and corresponding colors for plotting.
    lambda_values = [1, 4, 10]
    colors = {1: "yellow", 4: "red", 10: "blue"}

    plt.figure(figsize=(10, 6))

    # 3) For each λ, compute the empirical distribution of total spikes per image.
    for lam in lambda_values:
        coder = RateCoding(scaling_factor=lam, dt=0.001, duration=0.1)
        total_spikes_per_image = []
        for img in x_train:
            spike_train = coder.encode(img)
            total_spikes_per_image.append(spike_train.sum())
        total_spikes_per_image = np.array(total_spikes_per_image, dtype=int)
        max_spike_count = total_spikes_per_image.max()

        # Build histogram bins from 0 to max_spike_count (inclusive)
        bins = np.arange(0, max_spike_count + 2)
        counts, edges = np.histogram(total_spikes_per_image, bins=bins, density=False)
        empirical_probs = counts / counts.sum()

        # Use edges[:-1] as the x-values (each bin represents a spike count)
        n_values = edges[:-1]
        plt.plot(
            n_values, empirical_probs, color=colors[lam], marker="o", label=f"λ = {lam}"
        )

        print(
            f"λ = {lam}, Mean total spikes per image: {total_spikes_per_image.mean():.2f}"
        )

    # 4) Plot the empirical distributions on the same graph.
    plt.xlabel("Total spikes per image (n)")
    plt.ylabel("Probability P(n)")
    plt.title(
        "Empirical Distribution of Total Spikes per Image\nfor Different λ (RateCoding on MNIST)"
    )
    plt.legend()
    plt.grid(True)
    plt.show()
