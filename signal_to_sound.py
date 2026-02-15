import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from graph_plot import load_mat_file, resample_signal


def normalize_signal(signal):

    ### Définition du Docstring ##

    """
    On normalise le signal : On le centre sur zéro et on le met à l'amplitude maximale de 1.
    
    """

    signal = signal - np.mean(signal) ### On retire la moyenne ### 
    signal = signal / np.max(np.abs(signal)) ### On met le signal à l’échelle [-1, 1] ###
    return signal


def plot_signal_clean(signal_clean, fs, start_time, window_s):

    ### Définition du Docstring ###
    
    """
    On affiche un segment du signal ECG nettoyé.
    
    """

    start_idx = int(start_time * fs)
    end_idx   = start_idx + int(window_s * fs)

    ecg_win = signal_clean[start_idx:end_idx]
    t = np.arange(len(ecg_win)) / fs + start_time ### Vecteur temps pour le segment ###

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

    ### Définition du Docstring ###

    """
    On convertit un signal ECG en fichier audio WAV à l'aide de soundfile.
    
    """

    sf.write(filename, signal_clean, fs)
    print(f"Saved audio to {filename}")


if __name__ == "__main__":

    INPUT_FOLDER = "processed"
    OUTPUT_FOLDER = "signals_sounds"

    DEFAULT_ECG_FS = 1000 ### Fréquence d'échantillonnage de L'ECG ###
    AUDIO_FS = 44100 ### Fréquence d'échantillonnage standard pour un audio ###

    start_time = 820
    window_s = 30  ### Fenêtre de 30 secondes sur L'ECG ###  

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for fname in os.listdir(INPUT_FOLDER):
        path = os.path.join(INPUT_FOLDER, fname)

        if not os.path.isfile(path) or not fname.lower().endswith(".mat"):
            continue

        try:
            print(f"Processing {fname} ...")
            
            ### Récupération des données filtrées de l'ECG ###
            data = load_mat_file(path)
            ecg_clean = data["ECG_clean"]

            plot_signal_clean(ecg_clean, DEFAULT_ECG_FS, start_time, window_s) ### Visualisation ###

            ### Sélection du segment ###
            start_idx = int(start_time * DEFAULT_ECG_FS)
            end_idx = start_idx + int(window_s * DEFAULT_ECG_FS)
            ecg_segment = ecg_clean[start_idx:end_idx]

            ecg_norm = normalize_signal(ecg_segment) ### Normalisation du segment pour amplitude entre -1 et 1 ###

            ecg_audio = resample_signal(ecg_norm, DEFAULT_ECG_FS, AUDIO_FS) ### Conversion du signal normalisé en audio à la fréquence standard AUDIO_FS ###

            out_name = fname.replace(".mat", "_ecg.wav")
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            signal_to_audio(ecg_audio, AUDIO_FS, out_path)

        except Exception as e:
            print(f"Erreur sur {fname}: {e}")
