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

<div class="row" align="center">
  <div class="column" align="center">
    <img src="https://github.com/FaezehAtaei/MSc-Project/assets/27311166/23dc3f7f-a404-4d93-b464-248760edb5ed"/>
  </div>
</div>


# Getting started

You can clone this project using this [link](https://github.com/FaezehAtaei/MSc-Project.git) and install requierments by ```pip install -r requirements.txt```. The ```requirements.txt``` would install everything you need. However, before using the ```requirements.txt```, it is suggested to create a virtual environment with python 3.8.16. This code was developed and tested on Ubuntu 20.04.6 using Python 3.8.16 and PyTorch 1.13.1.

## Inputs

In this project MNIST and CIFAR-10 datasets were used to train the model. It is organized to download and prepare the dataset for training in the first run. However, the default dataset is MNIST. To change the default dataset, you just need to change the name of the dataset from ```MINST``` to ```CIFAR-10``` in the ```data_downloader``` funtion which is located in the ```utils.py``` file.

## Training the model

To initiate the model training process, you have a range of options available in the ```models.py``` file. Once you've decided on a particular model architecture, proceed to the ```main.py``` file. In the ```main.py``` file, you'll find a variable named ```model```. This variable needs to be assigned the selected model from the ```models.py``` file. For instance, if you've chosen the ```Mem_VAE_MNIST``` model, which is memory augmented VAE for MNIST dataset, your assignment would be: ```model = Mem_VAE_MNIST()```.

Moreover, prior to commencing the training, remember to set the ```NORMAL_TARGET``` variable in the ```main.py``` file. This variable designates the specific target for normal data in your dataset. For instance, if the target for normal data is label 0, your assignment would be: ```NORMAL_TARGET = 0```.

Following this, you should specify the desired dataset's name in the ```data_downloader``` function within the ```utils.py``` file. Once these steps are complete, you can execute the ```main.py``` file to initiate the training process with your chosen model and dataset.

## Outputs

During training, after each epoch, saved examples of reconstructed input, normal data tests, and anomaly data tests will be saved in the ```results``` folder. At the training's completion, the ```results``` folder will include scatter plots displaying training and validation losses, AUC values, and reconstruction errors.

# Acknowledgments
This code is implemented by getting help from the following sources:
- [Original implementation of active object localization algorithm](https://github.com/jccaicedo/localization-agent)
- [Tutorial for deep reinforcement learning](https://github.com/dennybritz/reinforcement-learning)
- [Tutorial for deep learning from the university of Edinburgh](https://github.com/otoofim/mlpractical)


