import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_mat_file(filepath, filename):
    data = sio.loadmat(filepath)

    # 🔹 extract 
    ecg_raw = data["ecg_raw"].squeeze()
    ecg = data["ecg"].squeeze()
    scg = data["scg"].squeeze()
    rhc = data["rhc"].squeeze()

    patch_ACC_lat = data["patch_ACC_lat"].squeeze()
    patch_ACC_hf  = data["patch_ACC_hf"].squeeze()
    patch_ACC_dv  = data["patch_ACC_dv"].squeeze()

    # 🔥 derivatives (optional but useful)
    vel = np.gradient(scg)
    acc = np.gradient(vel)

    # 🔹 plot
    plt.figure(figsize=(12, 8))

    plt.subplot(6, 1, 1)
    plt.plot(ecg)
    plt.title("ECG")
    
    plt.subplot(6, 1, 2)
    plt.plot(patch_ACC_lat)
    plt.title("Patch ACC Lat")
    
    plt.subplot(6, 1, 3)
    plt.plot(patch_ACC_hf)
    plt.title("Patch ACC HF")
    
    
    plt.subplot(6, 1, 4)
    plt.plot(patch_ACC_dv)
    plt.title("Patch AC DV")

    plt.subplot(6, 1, 5)
    plt.plot(scg)
    plt.title("SCG")

    plt.subplot(6, 1, 6)
    plt.plot(rhc)
    plt.title("RHC")

    # plt.subplot(5, 1, 6)
    # plt.plot(vel, label="velocity")
    # plt.plot(acc, label="acceleration")
    # plt.legend()
    # plt.title("SCG derivatives")
    plt.suptitle(f"File: {filename}")

    plt.tight_layout()
    plt.show()
    
    

for fname in os.listdir("segments_30s_strict"):
    try:
        plot_mat_file(f"segments_30s_strict/{fname}", fname)
    except Exception as e:
        print(f"Error plotting {fname} : {e}")