# ==============================
# Filter parameters
# ==============================

# Example: you can change theses filters
ECG_FILTERS = [
    {"type": "highpass", "cutoff": 0.5},
    {"type": "bandpass", "low": 1, "high": 40},
]

SCG_FILTERS = [
    {"type": "bandpass", "low": 1, "high": 40},
    {"type": "suppress_motion"},
    {"type": "hampel_filter", "window": 15, "n_sigmas": 3},
]
