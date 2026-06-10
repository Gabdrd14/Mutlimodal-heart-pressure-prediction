import os
import argparse
import time
import logging
import numpy as np
import scipy.io as sio
from preprocessing import CleanPreprocessingPipeline, ArtifactCleaner, WFDBDataProcessor
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================
# ARGUMENTS
# ==============================

parser = argparse.ArgumentParser(description="Batch preprocessing of medical signal data")
parser.add_argument("-i", "--input", required=False, help="Input folder path") 
parser.add_argument("--raw", action="store_true", help="Process raw .mat files")
parser.add_argument("--wfdb", action="store_true", help="Process WFDB files from dat_signals/")
parser.add_argument("-o", "--output", default="processed", help="Output folder (default: processed)")

args = parser.parse_args()

# Validate arguments
if args.wfdb:
    # WFDB mode: no input needed, processes dat_signals/ folder
    METHOD = "wfdb"
    INPUT_FOLDER = "dat_signals"
    PROCESS_ALL_WFDB = True
elif args.raw:
    if not args.input:
        parser.error("--input is required for --raw mode")
    INPUT_FOLDER = args.input
    METHOD = "raw"
    PROCESS_ALL_WFDB = False
else:
    parser.error("Choose --raw or --wfdb mode")

OUTPUT_DIR = args.output


# ==============================
# OUTPUT
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# FILTER ENGINE 
# ==============================

class FilterEngine:
    """
    Docstring pour FilterEngine

        Class qui va renvoyer les signaux filtrés en fonction du config.py 

    """

    def __init__(self, cleaner):
        self.c = cleaner

        self.map = {
            "highpass": self.hp,
            "lowpass": self.lp, 
            "bandpass": self.bp,
            "swt_filter": self.swt,
            "suppress_motion": self.motion,
            "hampel_filter": self.hampel,
        }

    def hp(self, sig, p):
        return self.c.highpass(sig, p["cutoff"])
    
    def lp(self, sig, p):
        return self.c.lowpass(sig, p["cutoff"])


    def bp(self, sig, p):
        return self.c.bandpass(sig, p["low"], p["high"])

    def swt(self, sig, p):
        return self.c.swt_filter(
            sig,
            wavelet=p.get("wavelet","db4"),
            level=p.get("level",2),
            method=p.get("method","soft")
        )

    def motion(self, sig, p):
        kernel = p.get("kernel_size", 5)
        return self.c.suppress_motion(sig, kernel_size=kernel)

    def hampel(self, sig, p):
        window = p.get("window", 31)
        n_sigmas = p.get("n_sigmas", 3)
        return self.c.hampel_filter(sig, window=window, n_sigmas=n_sigmas)

    def apply(self, sig, filters):
        out = sig
        for f in filters:
            try:
                out = self.map[f["type"]](out, f)
            except Exception as e:
                logger.error(f"Error applying filter {f['type']}: {e}")
                continue
        return out


# ==============================
# MAIN LOOP
# ==============================

DEFAULT_FS = config.FS

cleaner = ArtifactCleaner(fs=DEFAULT_FS)
engine = FilterEngine(cleaner)

start_time = time.time()
logger.info(f"Starting batch processing in {METHOD} mode")

# ========== WFDB MODE ==========
if PROCESS_ALL_WFDB:
    logger.info(f"Processing WFDB records from {INPUT_FOLDER}/")
    
    # Get unique records
    records = set()
    for fname in os.listdir(INPUT_FOLDER):
        if fname.endswith('.hea'):
            base_name = fname.replace('.hea', '')
            records.add(base_name)
    
    if not records:
        logger.error(f"✗ No WFDB records found in {INPUT_FOLDER}/")
    else:
        logger.info(f"Found {len(records)} unique records to process")
        logger.info(f"{'='*70}")
        
        successful = 0
        failed = 0
        
        for idx, record_name in enumerate(sorted(records), 1):
            try:
                logger.info(f"\n[{idx}/{len(records)}] Processing: {record_name}")
                file_start_time = time.time()
                
                # Input and output paths
                input_path = os.path.join(INPUT_FOLDER, record_name)
                output_path = os.path.join(OUTPUT_DIR, f"{record_name}.mat")
                
                # Check if already processed
                if os.path.exists(output_path):
                    logger.info(f"⊘ Already processed: {output_path} (skipping)")
                    continue
                
                # Process WFDB record
                processor = WFDBDataProcessor(input_path, fs=500, scg_bandpass=(1, 40))
                processor.load()
                processor.process()
                

                # Save
                if processor.save_mat(output_path):
                    successful += 1
                    file_end_time = time.time()
                    logger.info(f"SUCCESS in {file_end_time - file_start_time:.2f}s")
                else:
                    failed += 1
                    logger.error(f"FAILED to save")
                    
            except Exception as e:
                failed += 1
                logger.error(f"ERROR processing {record_name}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"WFDB Processing Summary: {successful} successful, {failed} failed")

# ========== RAW MODE ==========
else:
    for fname in os.listdir(INPUT_FOLDER):
        ext = fname.lower().split(".")[-1]
        
        if ext != "mat":
            continue

        logger.info(f"Processing: {fname}")
        file_start_time = time.time()

        try:
            path = os.path.join(INPUT_FOLDER, fname)

            pipe = CleanPreprocessingPipeline(path, METHOD)
            data = pipe.run()

            ecg_raw = data["ecg_raw"]
            scg_raw = data["scg_raw"]
            scg_lat = data["patch_ACC_lat"]
            scg_hf = data["patch_ACC_hf"]
            scg_dv = data["patch_ACC_dv"]

            # Application du filtrage
            ecg_clean = engine.apply(ecg_raw, config.ECG_FILTERS)
            scg_clean = engine.apply(scg_raw, config.SCG_FILTERS)
            patch_ACC_lat = engine.apply(scg_lat, config.SCG_FILTERS)
            patch_ACC_hf = engine.apply(scg_hf, config.SCG_FILTERS)
            patch_ACC_dv = engine.apply(scg_dv, config.SCG_FILTERS)
            
            t = data["time_ECG"]
            if t is None:
                t = np.arange(len(ecg_raw)) / DEFAULT_FS

            out = {
                "ecg_raw": ecg_raw,
                "scg_raw": scg_raw,
                "ecg_clean": ecg_clean,
                "scg_clean": scg_clean,
                "patch_ACC_lat" : patch_ACC_lat,
                "patch_ACC_hf" : patch_ACC_hf ,
                "patch_ACC_dv" : patch_ACC_dv ,
                "time": t
            }

            out_name = fname.replace(".mat", ".mat")
            sio.savemat(os.path.join(OUTPUT_DIR, out_name), out)  

            file_end_time = time.time()
            logger.info(f"Saved {out_name} in {file_end_time - file_start_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error processing {fname}: {e}")

end_time = time.time()
logger.info(f"Batch processing completed in {end_time - start_time:.2f} seconds")
