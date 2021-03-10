# Autoencoder Experiments

## Introduction
This repository contains source code of different types of autoencoder and experiments using autoencoders on different datasets.

## Repository structure
```bash
.
├── autoencoders                      // main package, contains autoencoder classes
│   ├── __init__.py                   // 
│   ├── base_autoencoder.py           // base class for autoencoder
│   ├── adversarial_autoencoder.py    // Adversarial autoencoder class
│   ├── variational_autoencoder.py    // Variational autoencoder class
│   ├── denoising_autoencoder.py      // Denoising autoencoder class
│   ├── info_vae.py                   // Info variational autoencoder class
│   ├── ae_callbacks.py               // Callbacks class
│   └── utils.py                      //
├── docs
│   └── README.md                     // documentation, this file
├── experiments                       // script to run experiments with each type of autoencoder
│   ├── adversarial_ae
│   ├── denoising_ae
│   ├── info_vae
│   └── variational_autoencoder
├── setup.py                          // information for package installation
└── tests                             // test files
    └── autoencoders
```

## Installation
Clone the repository
```bash
git clone https://github.com/KienMN/Autoencoder-Experiments.git
```
To install the autoencoder package, run the command
```bash
pip install git+https://github.com/KienMN/Autoencoder-Experiments.git
```
or 
```bash
cd Autoencoder-Experiments
pip install -e .
```

## Requrements
The code uses `tensorflow 2` on GPU to run experiments efficiently.  
See <a href=https://www.tensorflow.org/install/gpu>GPU support</a> to install GPU with Tensorflow.

## Usage
Experiments of each autoencoder are placed in the same folder. Hyperparameters and arguments are specified in file python script in each {autoencoder_folder}.

Run the experiments by the command:
```bash
python experiments/{autoencoder_folder}/{autoencoder_experiment}.py -d {dataset_name}
```

## Citation
This source code is for the paper:

Mai Ngoc K., Hwang M. (2020) Finding the Best k for the Dimension of the Latent Space in Autoencoders. In: Nguyen N.T., Hoang B.H., Huynh C.P., Hwang D., Trawiński B., Vossen G. (eds) Computational Collective Intelligence. ICCCI 2020. Lecture Notes in Computer Science, vol 12496. Springer, Cham. https://doi.org/10.1007/978-3-030-63007-2_35

```
@inproceedings{10.1007/978-3-030-63007-2_35,
author="Mai Ngoc, Kien and Hwang, Myunggwon",
title="Finding the Best k for the Dimension of the Latent Space in Autoencoders",
booktitle="Computational Collective Intelligence",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="453--464",
isbn="978-3-030-63007-2"
}
```

## Reference
1. From Autoencoder to Beta-VAE: https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html  
2. Adversarial autoencoder: https://github.com/Warvito/adversarial-autoencoder  
3. InfoVAE: https://github.com/ShengjiaZhao/InfoVAE
4. Variatioanl autoencoder tutorial: https://www.tensorflow.org/tutorials/generative/cvae
5. Variational autoencoder: https://github.com/cdoersch/vae_tutorial