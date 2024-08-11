# Image Super Resolution using Autoencoder

This repository contains an implementation of an image super-resolution technique utilizing an autoencoder-based deep neural network. The objective of this project is to reconstruct high-resolution images from low-resolution inputs, showcasing the application of autoencoders in the domain of image enhancement and restoration.

<div align="center"><img src="https://github.com/user-attachments/assets/e75e2497-89a3-4c8f-8992-9e7aacc03b33"></div>

# Overview
Image super-resolution is a pivotal problem in computer vision, involving the enhancement of image quality by increasing its resolution. This project leverages an autoencoder—a class of unsupervised deep neural networks—to learn a compressed representation of the input data and subsequently reconstruct it at a higher resolution. By training the autoencoder on low-resolution and corresponding high-resolution image pairs, the model learns to generate detailed high-resolution outputs from new, unseen low-resolution inputs.

The implementation is primarily demonstrated through a Jupyter Notebook, designed for easy experimentation and modification.

## Key Features

- Autoencoder Architecture: Implements a deep learning autoencoder with convolutional layers to perform high-quality image super-resolution.
- Image Preprocessing: Integrates image preprocessing techniques, including normalization and resizing, essential for effective model training and inference.
- Performance Visualization: Provides comprehensive visualizations to compare low-resolution inputs with the generated high-resolution outputs, enabling qualitative evaluation of the model.

## Installation
To replicate the results or further develop the project, ensure that your environment meets the following prerequisites:

- TensorFlow 
- Numpy
- Matplotlib
- OpenCV

## Usage
The project is structured around a Jupyter Notebook, which facilitates step-by-step execution and modification of the codebase. Follow the instructions below to get started:

### Steps to Run
1. Clone the repository:
```bash
git clone https://github.com/Dipon12/Image-Super-Resolution-using-Autoencoder.git
cd Image-Super-Resolution-using-Autoencoder
```

2. Open the Jupyter Notebook and run the cells to interact with the chatbot:

```bash
jupyter notebook "Image_High_Resolution_using_AutoEncoder.ipynb"
```

3. Model Training and Inference:

- Data Preparation: The notebook guides you through preprocessing the images for training.
- Model Training: Train the autoencoder on the provided dataset or your own data.
- Super-Resolution Generation: Apply the trained model to generate high-resolution images from low-resolution inputs.

## Result
The provided results illustrate the performance of the autoencoder-based image super-resolution model. The top left image displays the low-resolution input image, which exhibit significant pixelation due to downsampling. The bottom-left image shows the model's output, which effectively restores much of the lost detail, evident in the sharper edges and reduced blurring. This qualitative improvement suggests that the model is performing well, achieving higher PSNR values, indicating a high degree of similarity to the original high-resolution image shown on the bottom-right. While there is still a large room for refinement, these results demonstrate the model's capability in enhancing image resolution.

<div align="center"><img src="https://github.com/user-attachments/assets/c2304fb3-a645-46e1-8fb8-2c8fca763f8c"></div>

## Files
- Image_High_Resolution_using_AutoEncoder.ipynb: The core Jupyter Notebook demonstrating the implementation and training process of the autoencoder model.
- requirements.txt: A specification file listing all necessary Python libraries.
- README.md: This technical documentation file, providing an overview and detailed instructions for the project.

