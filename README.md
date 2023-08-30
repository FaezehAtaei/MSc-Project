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

![n](https://github.com/FaezehAtaei/MSc-Project/assets/27311166/23dc3f7f-a404-4d93-b464-248760edb5ed)


# Getting started

You can clone this project using this [link](https://github.com/FaezehAtaei/MSc-Project.git) and install requierments by ```pip install -r requirements.txt```. The requirements.txt would install everything you need. However, before using the requirements.txt, it is suggested to create a virtual environment with python 3.8.16. This code was developed and tested on Ubuntu 20.04.6 using Python 3.8.16 and PyTorch 1.13.1.
