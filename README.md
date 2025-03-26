# NYCU CV 2025 Spring - Homework 1

[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Student Information
- **Student ID**: 110550020  
- **Name**: Enfu Liao (廖恩莆)  

## Description

Image classification

## Installation & Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/lnfu/nycu-cv-hw1.git
   cd nycu-cv-hw1
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple --break-system-packages
   ```
3. Run the main script:
   ```sh
   python -m nycu_cv_hw1.train config.yaml
   python -m nycu_cv_hw1.test > prediction.csv
   ```

## Repository Structure

```
├── data/              # Dataset used for training/testing (if applicable)
│   ├── all
│   ├── train
│   ├── val
│   └── test
├── logs/              # Training history
├── nycu_cv_hw1/       # Main package containing source code
├── config.yaml        # Configuration file for training (e.g., hyperparameters)
├── README.md          # This file
└── requirements.txt   # Required dependencies
```

## Results
