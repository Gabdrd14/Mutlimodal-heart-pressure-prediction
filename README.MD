# Multimodal ECG/SCG Preprocessing Pipeline

High-performance, modular preprocessing pipeline for multimodal biomedical signals  
(ECG + SCG from wearable sensors and WFDB datasets).

Designed for large-scale batch processing, research reproducibility, and AI-based physiological modeling.

---

## 📌 Overview

This project provides a configurable and optimized preprocessing framework for:

- Electrocardiogram (ECG)
- Seismocardiogram (SCG)

with support for:

- Raw wearable patch `.mat` files  
- Processed WFDB `.dat` records  

The pipeline is fully driven by configuration files, allowing rapid experimentation without modifying core logic.

---

## 🎯 Key Features

### Data Handling
- Raw MATLAB wearable datasets
- WFDB physiological signal records

### Signal Processing
- Butterworth high-pass filter
- Butterworth band-pass filter
- Motion artifact suppression
- Hampel outlier filter (optimized)
- Stationary wavelet denoising (optional)


