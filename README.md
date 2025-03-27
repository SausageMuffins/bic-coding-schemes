# bic-coding-schemes
This repository implements various coding schemes for spiking neural networks (SNNs) based on the paper by Guo et al. (2021). The coding schemes include **Rate Coding**, **Time-to-First-Spike (TTFS) Coding**, **Phase Coding**, and **Burst Coding**.

# Coding Schemes

## Rate Coding
Rate coding encodes information using the spike firing rate. A higher firing rate corresponds to a stronger (brighter) input signal, while a lower firing rate corresponds to a weaker (darker) input signal.

In this implementation:
- Pixel intensities in the range [0–255] are first scaled by a factor (λ).
- The spike probability for each pixel is given by:
  `p_spike = (pixel_value / scaling_factor) * dt`
- The code uses a Poisson process: at each time step, it samples a uniform random number and compares it with the above probability to decide whether to generate a spike. Higher intensities thus lead to more frequent spikes.

Example usage:
    from rate_coding import RateCoding
    
    # Create a rate coder with λ = 4, dt = 0.001, duration = 0.1
    rate_coder = RateCoding(scaling_factor=4, dt=0.001, duration=0.1)
    
    # Encode a 2D image
    spike_train = rate_coder.encode(image)
    
    # spike_train has shape (height, width, num_steps)

---

## Time-to-First-Spike (TTFS) Coding
TTFS coding represents a pixel’s intensity by the timing of its **first spike**. Brighter pixels tend to fire earlier in the simulation; dimmer pixels fire later (or possibly not at all if the threshold has decayed too much).

1. Normalize each pixel to [0, 1].
2. Discretize the simulation time into steps of length `dt`. Let t = i * dt.
3. Define a time-dependent threshold:
   `P_th(t) = theta0 * exp(-t / tau_th)`
4. A pixel fires its first spike if:
   `pixel_value_norm > P_th(t)`
5. After that first spike, the pixel is inhibited (no further spikes).

Example usage:
    from ttfs_coding import TTFSCoding
    
    # Create a TTFS coder
    coder = TTFSCoding(dt=0.001, duration=0.02, tau_th=0.006, theta0=1.0)
    
    # Encode the image
    spike_train = coder.encode(image)
    
    # spike_train has shape (height, width, num_steps)

---

## Phase Coding
Phase coding encodes intensity using an **8-bit binary representation** repeated across time. Each bit corresponds to a specific phase in a cycle:

1. Convert each pixel to an 8-bit binary (MSB first).
2. Define a cycle of length `num_phases` (usually 8), repeated until simulation ends.
3. At each time step t, compute `phase_idx = t % num_phases`.
   - If that bit is "1," the pixel spikes; if it is "0," it does not spike.
4. This cycle repeats over the total duration.

Example usage:
    from phase_coding import PhaseCoding
    
    phase_coder = PhaseCoding(num_phases=8, dt=0.001, duration=0.1)
    spike_train = phase_coder.encode(image)
    
    # spike_train has shape (height, width, num_steps)

---

## Burst Coding
Burst coding assigns a burst of spikes to each pixel, with the burst size and inter-spike interval (ISI) dependent on pixel brightness:

1. Normalize pixel [0–255] to [0,1].
2. Compute the number of spikes:
   Ns = ceil(Nmax * P)
3. Compute the inter-spike interval:
   - If Ns > 1:  ISI(P) = (Tmax - Tmin) * (1 - P) + Tmin
   - Else:       ISI(P) = Tmax
4. Place spikes at times t_i = i * ISI(P) for i in [0, Ns-1], as long as t_i < duration.

Example usage:
    from burst_coding import BurstCoding
    
    coder = BurstCoding(dt=0.001, duration=0.01, Nmax=5, Tmin=0.002, Tmax=0.01)
    spike_train = coder.encode(image)
    
    # spike_train is (height, width, num_steps)

---

# References
Guo, W., Fouda, M. E., Eltawil, A. M., & Salama, K. N. (2021). Neural coding in spiking neural networks: A comparative study for robust neuromorphic systems. *Frontiers in Neuroscience*, 15, 638474.