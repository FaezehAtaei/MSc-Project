# MSc-Project

This project proposes a method capable of both detecting Out-of-Distribution (OOD) data
and generating new in-distribution data samples. To achieve this goal, a Variational Autoencoder
(VAE) model is adopted and augmented with a memory module. Integrating external memory
mechanisms with the VAE provides capacities for identifying OOD data points and synthesizing
new in-distribution samples. During training, the VAE is trained on normal data and the memory
stores prototypical patterns of the normal distribution. At test time, the input is encoded by the
VAE encoder, this encoding is used as a query to retrieve related memory items, which are then
integrated with the input encoding and passed to the decoder for reconstruction. Since the memory
contains underlying normal patterns, normal samples reconstruct well and yield low reconstruction
error while OOD inputs produce high reconstruction error due to distorted output resembling mem-
orized normal data.
