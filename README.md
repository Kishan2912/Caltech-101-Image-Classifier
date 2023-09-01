# Convolutional Neural Network for Image Classification

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Results](#models-and-results)
  - [Model Architectures](#model-architectures)
  - [Results](#results)
- [Image Analysis](#image-analysis)
- [Contributing](#contributing)

## Overview

This project focuses on utilizing Convolutional Neural Networks (CNNs) to perform image classification on a subset of the Caltech-101 dataset. The primary objective is to classify images into five distinct categories using various CNN architectures.

## Dataset

The dataset employed for this project is a subset of the Caltech-101 dataset, consisting of color images with varying sizes. The images have been resized to a standardized size of 224 x 224 pixels. The dataset is divided into training, validation, and test sets.



![train data](https://github.com/Kishan2912/Caltech-101-Image-Classifier/assets/83392319/65acceab-efd5-4636-8995-b2d2e316de32)

## Tech Stack

- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow with Keras
- **Data Preprocessing**: TensorFlow Data API
- **Data Augmentation**: TensorFlow ImageDataGenerator
- **Data Visualization**: Matplotlib
- **Model Evaluation**: Scikit-Learn metrics

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Kishan2912/Caltech-101-Image-Classifier


2. **Go to folder:**
    ```bash
   cd Caltech_Image_classification

3. **Install dependencies:**
    ```bash
   pip install -r requirements.txt


## Usage

### Model Training and Evaluation:
Execute the provided Jupyter Notebook or Python scripts to train and assess different CNN architectures. Modify hyperparameters and model architectures as necessary for experimentation.

### Analyze Model Performance:
Evaluate model performance, including training and testing accuracy. Visualize results using confusion matrices to gain insights into model behavior.


## Models and Results
### Model Architectures
This project explores three different CNN architectures, each with varying numbers of convolutional and fully connected layers. The architectures are designed to extract and learn features from the images effectively.

### Results
The project's best-performing architecture (6-layer) achieved an impressive test accuracy of 98%, showcasing the effectiveness of the chosen CNN architecture.

## Image Analysis
The project includes image analysis, such as feature map visualization and understanding which image patches activate specific neurons in the last convolutional layer. These analyses provide insights into how the CNN interprets and processes image data.

![feature mapp](https://github.com/Kishan2912/Caltech-101-Image-Classifier/assets/83392319/1a1b8255-55b9-4c05-ba19-9d7682c254b9)







## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

