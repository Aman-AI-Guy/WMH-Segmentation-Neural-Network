# White Matter Hyperintensity Segmentation Project

## Project Overview

This project focuses on the development of automated systems for the segmentation of White Matter Hyperintensities (WMH) in brain MRI scans. WMHs are regions of abnormal signal intensity visible on brain MRI scans, often associated with small vessel disease, aging, and cognitive impairment. Accurate segmentation of these regions is crucial for clinical assessment, progression monitoring, and research in neurodegenerative disorders.

## Technical Approaches

The project implements two complementary approaches to WMH segmentation:

### 1. Fuzzy Inference System (FIS)
The MATLAB implementation uses a fuzzy logic-based approach that:
- Processes grayscale brain MRI images to identify WMH regions
- Calculates local statistical features using a 3×3 sliding window, including:
  - Mean intensity
  - Standard deviation
  - Kurtosis
  - Skewness
- Applies a fuzzy inference system to classify pixels as WMH or non-WMH
- Implements genetic algorithm-based optimization to tune rule weights for improved accuracy

### 2. Neural Network Approach
The Python implementation uses TensorFlow/Keras to create a neural network that:
- Processes flattened image datasets with extracted statistical features
- Normalizes feature data using min-max scaling
- Trains on labeled WMH mask data
- Likely implements a supervised learning approach for binary classification of brain tissue

## Project Components

The project consists of several key components:
- **Image Processing**: Techniques for feature extraction from brain MRI scans
- **Statistical Analysis**: Calculation of local statistics for tissue characterization
- **Fuzzy Logic System**: Rule-based classification system built in MATLAB
- **Genetic Algorithm**: Optimization method for FIS rule weight tuning
- **Neural Network**: Deep learning approach for WMH classification
- **Evaluation Metrics**: Tools for assessing segmentation performance

## Data Handling

The project works with:
- FLAIR MRI sequences (common for WMH visualization)
- Ground truth WMH masks for training and validation
- Extracted feature datasets (mean, standard deviation, skewness, kurtosis, variance)

## Applications

This WMH segmentation system has potential applications in:
- Clinical diagnosis of cerebrovascular and neurodegenerative diseases
- Quantitative assessment of disease progression
- Research studies on aging and cognitive impairment
- Computer-aided diagnosis systems for neurological disorders

This project demonstrates the complementary use of traditional rule-based systems (FIS) and modern machine learning approaches (neural networks) to address the challenging problem of medical image segmentation.
