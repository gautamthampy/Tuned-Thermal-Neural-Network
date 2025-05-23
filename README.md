# Tuned-Thermal-Neural-Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)

## Overview

This repository demonstrates the application of **Thermal Neural Networks (TNNs)** on electric motor temperature prediction, featuring automatic hyperparameter tuning and explainability analysis.

**Thermal Neural Networks** are physics-informed neural networks that emulate classic lumped-parameter thermal circuits, enabling temperature prediction in thermal systems without requiring detailed knowledge of physical parameters.

### Key Capabilities

- **Thermal System Modeling**: Physics-informed approach to thermal circuit modeling
- **Electric Motor Applications**: Temperature prediction for electric motor components
- **Automated Hyperparameter Tuning**: Keras Tuner integration
- **Explainable AI**: SHAP and tf-explain integration for model interpretability

## Features

### Core Functionality
- **Physics-Informed Neural Networks**: Incorporates thermal circuit principles into neural network architecture
- **Multi-Component Temperature Prediction**: Predicts temperatures for permanent magnet, stator yoke, stator tooth, and stator winding
- **Variable-Length Sequences**: Handles different profile lengths with masking

### Machine Learning Features
- **Automatic Hyperparameter Optimization**: Keras Tuner integration with Bayesian optimization
- **Model Interpretability**: SHAP values and gradient-based explanations
- **Cross-Validation**: Model validation with multiple motor operation profiles

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for training acceleration

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Tuned-Thermal-Neural-Network
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Download the Dataset

Download the electric motor temperature dataset from [Kaggle](https://www.kaggle.com/wkirgsn/electric-motor-temperature) and place the `measures_v2.csv` file in a `data/input/` directory:

```bash
mkdir -p data/input
# Place measures_v2.csv in data/input/
```

### 2. Run the Notebooks

**Main TNN implementation:**
```bash
jupyter notebook TNN_tensorflow.ipynb
```

**Hyperparameter tuning:**
```bash
jupyter notebook TNN_tensorflow_tuner.ipynb
```

## Project Structure

```
Tuned-Thermal-Neural-Network/
├── TNN_tensorflow.ipynb          # Main TNN implementation
├── TNN_tensorflow_tuner.ipynb    # Hyperparameter tuning notebook
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── data/
│   └── input/
│       └── measures_v2.csv       # Dataset (download separately)
├── artifacts/                    # Model outputs
│   ├── tnn_best.keras           # Best trained model
│   └── metrics.json             # Performance metrics
└── tuner_logs/                   # Hyperparameter tuning logs
    └── thermal_nn/              # Keras Tuner results
```

## Model Architecture

### Thermal Neural Network Components

A TNN consists of three main function approximators (MLPs):

1. **Thermal Conductance Network**: Estimates heat transfer coefficients between components
2. **Thermal Capacitance Network**: Models heat storage capacity (inverse capacitances as learnable parameters)
3. **Power Loss Network**: Predicts internal heat generation

### Physics-Informed Design

The model incorporates the **Lumped Parameter Thermal Network (LPTN)** approach:

```
dT/dt = (1/C) * [P_loss + Σ(G_ij * (T_j - T_i))]
```

Where:
- `T`: Temperature vector
- `C`: Thermal capacitances
- `P_loss`: Power losses
- `G_ij`: Thermal conductances between components i and j

### Implementation Details

- **Input Features**: Motor operating conditions (currents, voltages, speeds, ambient conditions)
- **Target Outputs**: Component temperatures (permanent magnet, stator yoke, stator tooth, stator winding)
- **Temporal Processing**: RNN-based architecture for time series prediction
- **Physics Constraints**: Enforced through thermal circuit equations

The TNN's inner cell working is that of [lumped-parameter thermal networks](https://en.wikipedia.org/wiki/Lumped-element_model#Thermal_systems) (LPTNs).
A LPTN is an electrically equivalent circuit whose parameters can be interpreted to be thermal parameters of a system.
A TNN can be interpreted as a hyper network that is parameterizing a LPTN, which in turn is iteratively solved for the current temperature prediction.

In contrast to other neural network architectures, a TNN needs at least to know which input features are temperatures and which are not.
Target features are always temperatures.

In a nutshell, a TNN solves the difficult-to-grasp nonlinearity and scheduling-vector-dependency in [quasi-LPV](https://en.wikipedia.org/wiki/Linear_parameter-varying_control) systems, which a LPTN represents.

## Dataset

### Electric Motor Temperature Dataset

- **Source**: [Kaggle - Electric Motor Temperature](https://www.kaggle.com/wkirgsn/electric-motor-temperature)
- **Description**: Real-world measurements from electric motor operations
- **Features**: 
  - Electrical measurements (currents, voltages)
  - Mechanical measurements (torque, speed)
  - Environmental conditions (ambient temperature, coolant)
- **Targets**: Temperature measurements from 4 motor components
- **Profiles**: Multiple operating scenarios for training/testing

### Data Preprocessing

- **Normalization**: Temperatures scaled by 200°C, other features by maximum absolute values
- **Feature Engineering**: Computed current and voltage magnitudes
- **Temporal Structuring**: Organized into time-major tensors for RNN processing
- **Missing Data Handling**: Masking for variable-length sequences

