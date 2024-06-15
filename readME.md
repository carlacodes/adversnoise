# adversarial noise generation using PGD, pytorch implementation

## Introduction
This is a pytorch implementation of the Projected Gradient Descent (PGD) attack. 
The PGD attack is a white-box attack that aims to generate adversarial examples by adding small perturbations to the input data. 
The perturbations are generated by maximizing the loss function with respect to the input data, while ensuring that the perturbed data 
within a small epsilon-ball around the original data. I used a simple resnet18 model trained on ImageNet dataset (n.b. there is also a copy of deepfool modified slightly for the latest version of pytorch in the helpers folder).

## Installation
To run the code, you need to have the following main packages installed:
- pytorch
- numpy

Clone the repository:
``` git clone https://github.com/carlacodes/adversnoise.git ```

and run the following command to install the required packages using conda:
``` conda env create -f adversnoise.yml ```
