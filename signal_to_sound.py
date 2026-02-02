import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from graph_plot import load_mat_file, resample_signal


def normalize_signal(signal):
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    return signal


def plot_signal_clean(signal_clean, fs, start_time, window_s):
    start_idx = int(start_time * fs)
    end_idx   = start_idx + int(window_s * fs)

    ecg_win = signal_clean[start_idx:end_idx]
    t = np.arange(len(ecg_win)) / fs + start_time

    plt.figure(figsize=(14, 4))
    plt.plot(t, ecg_win, label="Clean ECG", linewidth=1.2)
    plt.title("Clean ECG")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def signal_to_audio(signal_clean, fs, filename):
    sf.write(filename, signal_clean, fs)
    print(f"Saved audio to {filename}")


if __name__ == "__main__":

    INPUT_FOLDER = "processed"
    OUTPUT_FOLDER = "signals_sounds"

    DEFAULT_ECG_FS = 1000
    AUDIO_FS = 44100

    start_time = 820
    window_s = 30

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for fname in os.listdir(INPUT_FOLDER):
        path = os.path.join(INPUT_FOLDER, fname)

        if not os.path.isfile(path) or not fname.lower().endswith(".mat"):
            continue

        try:
            print(f"Processing {fname} ...")

            data = load_mat_file(path)
            ecg_clean = data["ECG_clean"]

            plot_signal_clean(ecg_clean, DEFAULT_ECG_FS, start_time, window_s)

            start_idx = int(start_time * DEFAULT_ECG_FS)
            end_idx = start_idx + int(window_s * DEFAULT_ECG_FS)
            ecg_segment = ecg_clean[start_idx:end_idx]

            ecg_norm = normalize_signal(ecg_segment)
            ecg_audio = ecg_norm / np.max(np.abs(ecg_norm))

            ecg_audio = resample_signal(ecg_audio, DEFAULT_ECG_FS, AUDIO_FS)

            out_name = fname.replace(".mat", "_ecg.wav")
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            signal_to_audio(ecg_audio, AUDIO_FS, out_path)

        except Exception as e:
            print(f"Erreur sur {fname}: {e}")
