# Automobile Image Synthesizer: Generative Adversarial Network (GAN)

This repository contains a deep learning implementation of a Generative Adversarial Network (GAN) built using Keras to study adversarial training dynamics and synthesize high-fidelity images.

The model is trained specifically on **Class 1 (Automobiles)** of the classic **CIFAR-10** image dataset over a large-scale execution of **15,000 epochs**.

## 🚀 Features
* **Adversarial Architecture**: Features a competing Generator and Discriminator network adhering to the foundational Goodfellow et al. (2014) framework.
* **Large-Scale Training**: Evaluated over 15,000 training epochs to monitor structural feature convergence from initial random noise to defined object geometry.
* **Performance Analysis**: Logs training loss curves for both sub-networks to map stability and mitigate generative vulnerabilities like mode collapse.

## 🛠️ Tech Stack
* **Python**: Core scripting language.
* **Keras / TensorFlow**: Deep learning framework used to model and compile the neural network layers.
* **Matplotlib / NumPy**: Used for matrix operations and saving visual progressions across epochs.

## 📂 Project Structure
* `gan.py` - The monolithic production script containing the data pipeline, Generator/Discriminator neural layers, custom loss loops, and evaluation plots.
* `README.md` - Project documentation and technical details.
