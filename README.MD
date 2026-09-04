# Multimodal ECG/SCG Preprocessing Pipeline for predicting RHC at 30s intervals
 
Modular preprocessing pipeline for multimodal biomedical signals  
(ECG + SCG from wearable sensors and WFDB datasets).

This repository contains code for preprocessing and analyzing ECG and SCG signals for heart pressure prediction RHC.  

---

## Overview

This project provides a configurable and optimized preprocessing framework for:

- Electrocardiogram (ECG)
- Seismocardiogram (SCG)


### Data Handling

- Raw MATLAB wearable datasets
- WFDB physiological signal records

### Signal Processing
- Butterworth high-pass filter
- Butterworth band-pass filter
- Motion artifact suppression
- Hampel outlier filter (optimized)
- Stationary wavelet denoising (optional)

##  Installation


### Create environment (recommended)

`python -m venv venv
source venv/bin/activate`


## Commands & Usage

▶ Process WFDB formatted data (.dat)

`python batch_preprocess.py -i path/to/input_folder --wfdb`


▶ Extract the best segments for the signals 

`python extractor_best_signals.py `


▶ Lancer le modele transformer

  `python rhc_predictor.py --segments_folder /path/to/the/segments `
  

The preprocessing script (`preprocessing.py`) is **tested with Python 3.11.14** due to compatibility issues with the `wfdb` library. Using other Python versions may result in errors.

---

## ⚡ Recommended Setup

It is **strongly recommended** to use a **virtual environment** to avoid modifying your global Python installation. This keeps dependencies isolated and prevents conflicts between projects.

### Create a virtual environment

```bash
# Navigate to your project folder
cd /path/to/Mutlimodal-heart-pressure-prediction

# Create a virtual environment
python -m venv venv

# Activate the environment (Linux/macOS)
source venv/bin/activate

# Activate the environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
