import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def plot_mat_file(filepath):
    data = sio.loadmat(filepath)

    # 🔹 extract 
    ecg_raw = data["ecg_raw"].squeeze()
    ecg = data["ecg"].squeeze()
    scg = data["scg"].squeeze()
    rhc = data["rhc"].squeeze()
    # rhc = data.get("rhc", np.zeros_like(ecg)).squeeze()

    # ensure same length
    # min_len = min(len(ecg), len(scg), len(rhc))
    # ecg, scg, rhc = ecg[:min_len], scg[:min_len], rhc[:min_len]

    # 🔥 derivatives (optional but useful)
    vel = np.gradient(scg)
    acc = np.gradient(vel)

    # 🔹 plot
    plt.figure(figsize=(12, 8))

    plt.subplot(5, 1, 1)
    plt.plot(ecg)
    plt.title("ECG")
    
    plt.subplot(5, 1, 2)
    plt.plot(ecg_raw)
    plt.title("ECG RAW")

    plt.subplot(5, 1, 3)
    plt.plot(scg)
    plt.title("SCG")

    plt.subplot(5, 1, 4)
    plt.plot(rhc)
    plt.title("RHC")

    plt.subplot(5, 1, 5)
    plt.plot(vel, label="velocity")
    plt.plot(acc, label="acceleration")
    plt.legend()
    plt.title("SCG derivatives")

    plt.tight_layout()
    plt.show()
    
    
plot_mat_file("segments_30s/TRM222_segment.mat")