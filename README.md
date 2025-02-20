# EEG Sleep Analysis Scripts

Welcome to this repository! This collection of Python scripts is designed to help researchers, particularly PhD students, analyze EEG sleep data efficiently. Whether you're new to EEG research or looking to automate your data processing pipeline, these scripts provide a structured workflow for preprocessing, analyzing, and exploring EEG features.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Feature Computation](#3-feature-computation)
  - [4. Exploration and Visualization](#4-exploration-and-visualization)
  - [5. Slow Wave Analysis](#5-slow-wave-analysis)
  - [6. Sleep Spindle Analysis](#6-sleep-spindle-analysis)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
This repository includes scripts for:
- Demographic data management
- EEG preprocessing
- Computing sleep-related EEG features (e.g., spectral power, aperiodic components, slow waves, sleep spindles)
- Exploring and visualizing the extracted data

These scripts follow a modular approach, making it easy to customize your analysis workflow.

## Usage
Each script serves a specific purpose in the pipeline. Follow these steps:

### 1. Data Preparation
- **`00_demographics.py`**: Loads participant demographic information, ensuring all data is well-structured before processing.

### 2. Preprocessing
- **`01_Preproc.py`**: Cleans and preprocesses raw EEG data (e.g., filtering, artifact rejection).

### 3. Feature Computation
- **`02_01_compute_YASA_HD.py`**: Computes sleep spindles and slow oscillations using YASA.
- **`03_01_compute_globalpower.py`**: Computes global power across different frequency bands.
- **`03_03_compute_aperiodicpower.py`**: Extracts aperiodic components of EEG signals.
- **`03_05_compute_bandpower.py`**: Computes band-specific power (e.g., delta, theta, sigma).

### 4. Exploration and Visualization
- **`02_02_explore_YASA.py`**: Explores and visualizes detected spindles and slow oscillations.
- **`03_02_explore_globalpower.py`**: Examines spectral power distributions.
- **`03_04_explore_aperiodicpower.py`**: Visualizes aperiodic EEG components.
- **`04_01_compute_offset_exponent.py`**: Computes and explores offset and exponent features in EEG power spectra.

### 5. Slow Wave Analysis
- **`05_01_SW_detect.py`**: Detects slow waves in EEG recordings.
- **`05_02_SW_density.py`**: Computes the density of slow waves across different sleep stages.
- **`05_03_SW_plots.py`**: Generates plots to visualize slow wave characteristics.

### 6. Sleep Spindle Analysis
- **`06_01_SS_detect.py`**: Detects sleep spindles in EEG recordings.
- **`06_02_SS_density.py`**: Computes spindle density across different sleep stages.
- **`06_03_SS_plots.py`**: Generates plots to visualize spindle characteristics.

## Dependencies
These scripts require:
- `numpy`
- `pandas`
- `mne`
- `yasa`
- `matplotlib`
- `seaborn`
- `scipy`

(Ensure you install all necessary packages as mentioned in the installation section.)

---

### 🚀 Happy analyzing, and best of luck! 🚀
