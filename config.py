# ==============================
# Filter parameters
# ==============================

# Example: you can change theses filters
ECG_FILTERS = [

    {"type": "highpass", "cutoff": 0.5},
    {"type": "lowpass", "cutoff": 40},    
    {"type": "bandpass", "low": 1, "high": 40},
    {"type": "swt_filter","wavelet":"db4","level":2,"method":"soft"},
    {"type": "hampel_filter", "window": 15, "n_sigmas": 4},

]

SCG_FILTERS = [
    {"type": "bandpass", "low": 5, "high": 40},
    {"type": "hampel_filter", "window": 11, "n_sigmas": 3},

    # {"type": "suppress_motion"},
]
