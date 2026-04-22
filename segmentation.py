import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import neurokit2 as nk
import seaborn as sns
import collections
import time
import logging

from sklearn.cluster import DBSCAN
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import VarianceThreshold

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================
# CONFIG
# ==============================
INPUT_FOLDER = "processed"
FS = 1000

window_s = 30  
window_size = window_s * FS
step = window_size // 4


# ==============================
# FEATURE NAMES
# ==============================
feature_names = [
    "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD",
    "ECG_mean", "ECG_std", "ECG_max", "ECG_min",
    "SCG_mean", "SCG_std", "SCG_max", "SCG_min",
    "SCG_energy", "SCG_fft_mean", "SCG_fft_std", "SCG_fft_peak",
    "ACC_lat_mean", "ACC_lat_std", "ACC_lat_energy",
    "ACC_hf_mean", "ACC_hf_std", "ACC_hf_energy",
    "ACC_dv_mean", "ACC_dv_std", "ACC_dv_energy",
    "ECG_SCG_corr"
]


# # ==============================
# # QUALITY SCORE
# # ==============================
# def signal_quality(ecg, scg):
#     corr = np.corrcoef(ecg, scg)[0, 1]
#     if np.isnan(corr):
#         corr = 0

#     energy_ecg = np.sum(ecg**2)
#     energy_scg = np.sum(scg**2)

#     return abs(corr) + 0.1 * (energy_ecg + energy_scg)


# ==============================
# ECG FEATURES (NeuroKit)
# ==============================
def extract_ecg_features(ecg):
    try:
        # Clean the ECG signal first
        ecg_clean = nk.ecg_clean(ecg, sampling_rate=FS)
        signals, info = nk.ecg_process(ecg_clean, sampling_rate=FS)
        
        # Extract HRV features
        hrv = nk.hrv(info["ECG_R_Peaks"], sampling_rate=FS, show=False)
        
        return [
            hrv["HRV_MeanNN"].values[0],
            hrv["HRV_SDNN"].values[0],
            hrv["HRV_RMSSD"].values[0],
            np.mean(ecg_clean), np.std(ecg_clean), np.max(ecg_clean), np.min(ecg_clean)
        ]
    except Exception as e:
        logger.error(f"Error in extract_ecg_features: {e}")
        return [0]*7


# ==============================
# GENERIC FEATURES
# ==============================
def extract_signal_features(sig):
    fft = np.abs(np.fft.rfft(sig))
    return [
        np.mean(sig), np.std(sig), np.max(sig), np.min(sig),
        np.sum(sig**2),
        np.mean(fft), np.std(fft), np.argmax(fft)
    ]


# # ==============================
# # MAIN DATA EXTRACTION
# # ==============================
# all_features = []
# file_names = []

# for fname in os.listdir(INPUT_FOLDER):
#     if not fname.endswith(".mat"):
#         continue

#     print("Processing", fname)

#     try:
#         data = sio.loadmat(os.path.join(INPUT_FOLDER, fname))

#         ecg = data["ecg_clean"].squeeze()
#         scg = data["scg_clean"].squeeze()
#         acc_lat = data["patch_ACC_lat"].squeeze()
#         acc_hf = data["patch_ACC_hf"].squeeze()
#         acc_dv = data["patch_ACC_dv"].squeeze()

#         min_len = min(len(ecg), len(scg), len(acc_lat), len(acc_hf), len(acc_dv))
#         ecg, scg = ecg[:min_len], scg[:min_len]
#         acc_lat, acc_hf, acc_dv = acc_lat[:min_len], acc_hf[:min_len], acc_dv[:min_len]

#         best_score = -np.inf
#         best_segment = None

#         for start in range(0, min_len - window_size, step):
#             end = start + window_size

#             ecg_seg = ecg[start:end]
#             scg_seg = scg[start:end]

#             score = signal_quality(ecg_seg, scg_seg)

#             if score > best_score:
#                 best_score = score
#                 best_segment = (
#                     ecg_seg, scg_seg,
#                     acc_lat[start:end],
#                     acc_hf[start:end],
#                     acc_dv[start:end]
#                 )

#         if best_segment is None:
#             continue

#         ecg_seg, scg_seg, lat_seg, hf_seg, dv_seg = best_segment

#         features = []
#         features += extract_ecg_features(ecg_seg)
#         features += extract_signal_features(scg_seg)
#         features += extract_signal_features(lat_seg)
#         features += extract_signal_features(hf_seg)
#         features += extract_signal_features(dv_seg)

#         corr = np.corrcoef(ecg_seg, scg_seg)[0, 1]
#         features.append(0 if np.isnan(corr) else corr)

#         all_features.append(features)
#         file_names.append(fname)

#     except Exception as e:
#         print("Error:", e)



















# ==============================
# QUALITY SCORE (FIXED)
# ==============================
def signal_quality(ecg, scg):
    corr = np.corrcoef(ecg, scg)[0, 1]
    if np.isnan(corr):
        corr = 0

    # Energy is still meaningful even if normalized
    energy_ecg = np.sum(ecg ** 2)
    energy_scg = np.sum(scg ** 2)

    return abs(corr) + 0.1 * (energy_ecg + energy_scg)


# ==============================
# MAIN DATA EXTRACTION (FIXED)
# ==============================
all_features = []
file_names = []

start_time = time.time()
logger.info("Starting feature extraction process")

for fname in os.listdir(INPUT_FOLDER):
    if not fname.endswith(".mat"):
        continue

    logger.info(f"Processing {fname}")
    file_start_time = time.time()

    try:
        data = sio.loadmat(os.path.join(INPUT_FOLDER, fname))

        ecg = data["ecg_clean"].squeeze()
        scg = data["scg_clean"].squeeze()
        acc_lat = data["patch_ACC_lat"].squeeze()
        acc_hf = data["patch_ACC_hf"].squeeze()
        acc_dv = data["patch_ACC_dv"].squeeze()

        min_len = min(len(ecg), len(scg), len(acc_lat), len(acc_hf), len(acc_dv))

        ecg = ecg[:min_len]
        scg = scg[:min_len]
        acc_lat = acc_lat[:min_len]
        acc_hf = acc_hf[:min_len]
        acc_dv = acc_dv[:min_len]

        # ==============================
        # COLLECT ALL SEGMENTS
        # ==============================
        segments = []

        for start in range(0, min_len - window_size, step):
            end = start + window_size

            ecg_seg = ecg[start:end]
            scg_seg = scg[start:end]

            score = signal_quality(ecg_seg, scg_seg)

            segments.append((score, start, end))

        if len(segments) == 0:
            continue

        # ==============================
        # SELECT TOP 30 SEGMENTS
        # ==============================
        segments.sort(key=lambda x: x[0], reverse=True)
        top_segments = segments[:30]

        # ==============================
        # FEATURE EXTRACTION
        # ==============================
        for score, start, end in top_segments:

            ecg_seg = ecg[start:end]
            scg_seg = scg[start:end]

            lat_seg = acc_lat[start:end]
            hf_seg = acc_hf[start:end]
            dv_seg = acc_dv[start:end]

            features = []

            # Safe ECG processing (important)
            try:
                ecg_clean = nk.ecg_clean(ecg_seg, sampling_rate=FS)
                signals, info = nk.ecg_process(ecg_clean, sampling_rate=FS)
                hrv = nk.hrv(info["ECG_R_Peaks"], sampling_rate=FS, show=False)

                features += [
                    hrv["HRV_MeanNN"].values[0],
                    hrv["HRV_SDNN"].values[0],
                    hrv["HRV_RMSSD"].values[0],
                    np.mean(ecg_clean), np.std(ecg_clean),
                    np.max(ecg_clean), np.min(ecg_clean)
                ]

            except Exception as e:
                logger.error(f"Error processing ECG features for {fname}: {e}")
                features += [0]*7

            # SCG + ACC features
            features += extract_signal_features(scg_seg)
            features += extract_signal_features(lat_seg)
            features += extract_signal_features(hf_seg)
            features += extract_signal_features(dv_seg)

            corr = np.corrcoef(ecg_seg, scg_seg)[0, 1]
            features.append(0 if np.isnan(corr) else corr)

            all_features.append(features)
            file_names.append(fname)

    except Exception as e:
        logger.error(f"Error processing {fname}: {e}")

    file_end_time = time.time()
    logger.info(f"Finished processing {fname} in {file_end_time - file_start_time:.2f} seconds")

end_time = time.time()
logger.info(f"Feature extraction completed in {end_time - start_time:.2f} seconds")




# ==============================
# MATRIX
# ==============================
X = np.array(all_features)
logger.info(f"Initial shape: {X.shape}")

if X.shape[0] < 2:
    logger.error("Not enough data")
    exit()

# ==============================
# FEATURE SELECTION
# ==============================
start_time = time.time()
logger.info("Starting feature selection")

selector = VarianceThreshold(1e-5)
X = selector.fit_transform(X)

# correlation filter
corr_matrix = np.corrcoef(X, rowvar=False)
upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

to_drop = [i for i in range(len(corr_matrix)) 
           if any(abs(corr_matrix[i][j]) > 0.9 for j in range(i))]

X = np.delete(X, to_drop, axis=1)

logger.info(f"After selection: {X.shape}")

end_time = time.time()
logger.info(f"Feature selection completed in {end_time - start_time:.2f} seconds")

# ==============================
# NORMALIZATION
# ==============================
start_time = time.time()
logger.info("Starting normalization")

X = StandardScaler().fit_transform(X)

end_time = time.time()
logger.info(f"Normalization completed in {end_time - start_time:.2f} seconds")

# ==============================
# EMBEDDING
# ==============================
start_time = time.time()
logger.info("Starting embedding with KernelPCA")

kpca = KernelPCA(n_components=3, kernel='rbf', gamma=0.01)
X_kpca = kpca.fit_transform(X)

end_time = time.time()
logger.info(f"Embedding completed in {end_time - start_time:.2f} seconds")

# ==============================
# FIND EPS
# ==============================
def find_eps(X, k=10):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    plt.plot(k_distances)
    plt.title("K-distance graph")
    plt.show()

find_eps(X_kpca)

# ==============================
# DBSCAN
# ==============================
start_time = time.time()
logger.info("Starting DBSCAN clustering")

eps_value = 0.3

labels = DBSCAN(eps=eps_value, min_samples=4).fit_predict(X_kpca)

end_time = time.time()
logger.info(f"DBSCAN clustering completed in {end_time - start_time:.2f} seconds")  


# ==============================
# PLOTS
# ==============================

# 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_kpca[:,0], X_kpca[:,1], X_kpca[:,2], c=labels)
ax.set_title("3D Clustering")
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.set_zlabel("Dim 3")
plt.show()

# 2D
plt.scatter(X_kpca[:,0], X_kpca[:,1], c=labels)
plt.title("2D projection")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.colorbar()
plt.show()

# distribution
counter = collections.Counter(labels)
plt.bar(counter.keys(), counter.values())
plt.title("Cluster distribution")
plt.show()

# heatmap
sns.heatmap(np.corrcoef(X.T), cmap='coolwarm')
plt.title("Feature correlation")
plt.show()


# ==============================
# PCA INTERPRETATION
# ==============================
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

print("Explained variance:", pca.explained_variance_ratio_)

for i in range(3):
    print(f"\nTop features PC{i+1}:")
    idx = np.argsort(np.abs(pca.components_[i]))[-5:]
    print(idx)


# ==============================
# CLUSTER ANALYSIS
# ==============================
def analyze_clusters(labels, names):
    for u in set(labels):
        idx = np.where(labels == u)[0]
        print(f"\nCluster {u}: {len(idx)} samples")
        for i in idx[:5]:
            print(" ", names[i])

analyze_clusters(labels, file_names)