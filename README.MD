# Multimodal ECG/SCG Preprocessing Pipeline

High-performance, modular preprocessing pipeline for multimodal biomedical signals  
(ECG + SCG from wearable sensors and WFDB datasets).


---

## Overview

This project provides a configurable and optimized preprocessing framework for:

- Electrocardiogram (ECG)
- Seismocardiogram (SCG)

with support for:

- Raw wearable patch `.mat` files  
- Processed WFDB `.dat` records  

The pipeline is fully driven by configuration files

---

## Key Features

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
▶ Process raw wearable patch data (.mat)

`python batch_preprocess.py -i path/to/input_folder --raw `

▶ Process WFDB formatted data (.dat)

`python batch_preprocess.py -i path/to/input_folder --dat`



