# CSET419 – Introduction to Generative AI  
## Lab 2: Training a Basic GAN for Image Generation and Noise Simulation

## 1. Overview
This project implements a Basic Generative Adversarial Network (GAN) using PyTorch to generate synthetic images similar to the MNIST or Fashion-MNIST datasets. The trained model is saved and later reused to simulate image corruption using noise and image recovery using the generator.

## 2. Dataset Used
- MNIST (handwritten digits)
- Fashion-MNIST (clothing items)

Datasets are loaded using Torchvision utilities and normalized to the range [-1, 1].

## 3. Project Structure
gan_lab.py  
generator.pth  
discriminator.pth  
generated_samples/  
final_generated_images/  
denoising_results/  

## 4. Model Architecture
Generator: Fully connected neural network converting random noise into 28×28 grayscale images.  
Discriminator: Fully connected neural network classifying images as real or fake.

## 5. Training Procedure
The GAN is trained using adversarial learning where the discriminator and generator are trained alternately. Epoch-wise logs are printed and images are saved periodically.

## 6. Model Saving
After training, the generator and discriminator models are saved using torch.save for later reuse.

## 7. Noise Corruption and Reconstruction
Gaussian noise is added to real images to simulate corruption. The trained generator is then used to generate clean-looking synthetic images from the learned data distribution.

## 8. Explanation of Results
The reconstructed images are not pixel-wise restorations but realistic samples generated from the learned distribution. This is suitable for dataset simulation and pipeline testing.

## 9. Limitations
This is a standard GAN and not a true denoising autoencoder. Exact image restoration would require a denoising autoencoder or conditional GAN.

## 10. Conclusion
The experiment successfully demonstrates GAN-based image generation, model persistence, noise simulation, and generative reconstruction, fulfilling the objectives of CSET419 Lab 2.

## 11. How to Run
pip install torch torchvision matplotlib  
python gan_lab.py  
python denoise_images.py
