# NYCU CV 2025 Spring - Homework 1

[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Student Information
- **Student ID**: 110550020  
- **Name**: Enfu Liao (廖恩莆)  

## Description

This repo contains the implementation of transfer learning for image classification. The goal is to classify images into 100 categories, using a modified version of the ResNext101_64x4d backbone.

## Setup and Usage

Follow the steps below to set up and run the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/lnfu/nycu-cv-hw1.git
   cd nycu-cv-hw1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple --break-system-packages
   ```

3. Run the following command to aggregate the training and validation datasets into a single dataset:

   ```bash
   # This will merge the 'train' and 'val' datasets into 'all', and future scripts will use 'all'.
   python -m nycu_cv_hw1.utils.aggregate_data
   ```

4. Train the model

   ```bash
   python -m nycu_cv_hw1.train config.yaml
   ```

5. Inference

   ```sh
   python -m nycu_cv_hw1.test config.yaml > prediction.csv
   python -m nycu_cv_hw1.tta_test config.yaml > prediction.csv # use Test-Time Augmentation (TTA)
   ```

## Repository Structure

```
├── data/              # Dataset used for training/testing (if applicable)
│   ├── all            # Aggregated training and validation data
│   ├── train          # Training data
│   ├── val            # Validation data
│   └── test           # Test data
├── models/            # Model
├── logs/              # Training history
├── nycu_cv_hw1/       # Main package containing source code
├── config.yaml        # Configuration file for training (e.g., hyperparameters)
├── README.md          # This file
└── requirements.txt   # Required dependencies
```

## Results

<!-- TODO -->
