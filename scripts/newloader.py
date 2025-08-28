import json
from pathlib import Path
import pandas as pd
import scipy.io as sio
import mne

INDEX_FILE = Path("data/interim/data_index.json")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_index():
    """Load the JSON data index."""
    with open(INDEX_FILE, "r") as f:
        return json.load(f)

def load_behavior(sub, index=None, use_cache=True):
    """Load behavioral events from TSV files."""
    cache_file = PROCESSED_DIR / f"{sub}_behavior.csv"
    
    # Try loading from cache first
    if use_cache and cache_file.exists():
        print(f" Loading cached behavior for {sub}")
        return pd.read_csv(cache_file)
    
    # Load from source
    if index is None:
        index = load_index()
    
    func_files = index[sub]["func"]
    event_files = [f for f in func_files if f.endswith("_events.tsv")]
    
    runs = []
    for ef in event_files:
        df = pd.read_csv(ef, sep="\t")
        run_id = ef.split("_run-")[1].split("_")[0]
        df["run"] = run_id
        runs.append(df)
    
    behavior = pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()
    
    # Save to cache
    if use_cache and not behavior.empty:
        behavior.to_csv(cache_file, index=False)
        print(f" Cached behavior to {cache_file}")
    
    return behavior

def load_eeg(sub, index=None, use_cache=True):
    """Load EEG data from MAT files and create MNE Raw object."""
    cache_file = PROCESSED_DIR / f"{sub}_eeg_raw.fif"
    
    # Try loading from cache first
    if use_cache and cache_file.exists():
        print(f" Loading cached EEG for {sub}")
        try:
            return mne.io.read_raw_fif(cache_file, preload=False, verbose=False)  # Don't preload into memory
        except Exception as e:
            print(f" Cache corrupted for {sub}, reloading: {e}")
            cache_file.unlink()  # Delete corrupted cache
    
    # Load from source
    if index is None:
        index = load_index()
    
    eeg_files = index[sub]["eeg"]
    data_files = [f for f in eeg_files if "EEG_data" in f]
    
    if not data_files:
        print(f"  No EEG data files found for {sub}")
        return None
    
    raws = []
    for df in data_files:
        try:
            mat = sio.loadmat(df, squeeze_me=True, struct_as_record=False)
            
            eeg_struct = mat.get("EEGdata")
            if eeg_struct is None:
                print(f" No 'EEGdata' in {df}")
                continue
            
            data = eeg_struct.Y  # shape (n_channels, n_samples)
            fs = float(eeg_struct.fs)
            ch_names = [f"EEG{i+1:03d}" for i in range(data.shape[0])]
            
            info = mne.create_info(ch_names, sfreq=fs, ch_types="eeg", verbose=False)
            raw = mne.io.RawArray(data, info, verbose=False)
            raws.append(raw)
            
        except Exception as e:
            print(f" Failed to load {df}: {e}")
            continue
    
    if not raws:
        return None
    
    eeg = mne.concatenate_raws(raws, verbose=False) if len(raws) > 1 else raws[0]
    
    # Save to cache (but don't keep in memory)
    if use_cache and eeg is not None:
        try:
            eeg.save(cache_file, overwrite=True, verbose=False)
            print(f" Cached EEG to {cache_file}")
            # Reload from cache to free memory
            del eeg, raws  # Free memory
            return mne.io.read_raw_fif(cache_file, preload=False, verbose=False)
        except Exception as e:
            print(f"  Failed to cache {sub}: {e}")
    
    return eeg

def load_subject(sub, index=None, use_cache=True):
    """Load both EEG and behavior data for a subject."""
    if index is None:
        index = load_index()
    
    behavior = load_behavior(sub, index, use_cache)
    eeg = load_eeg(sub, index, use_cache)
    
    return {"subject": sub, "behavior": behavior, "eeg": eeg}

def load_all_subjects(use_cache=True, subjects_only=None):
    """Load all subjects in the dataset with memory management."""
    index = load_index()
    
    # Option to process only specific subjects
    if subjects_only:
        subjects_to_process = [s for s in subjects_only if s in index]
    else:
        subjects_to_process = list(index.keys())
    
    print(f" Processing {len(subjects_to_process)} subjects (cache={'ON' if use_cache else 'OFF'})...")
    print(" EEG data kept on disk to save memory\n")
    
    results = {"successful": [], "failed": []}
    
    for i, sub in enumerate(subjects_to_process, 1):
        print(f" [{i}/{len(subjects_to_process)}] Processing {sub}...")
        
        try:
            # Process one subject at a time to manage memory
            behavior = load_behavior(sub, index, use_cache)
            eeg = load_eeg(sub, index, use_cache)
            
            if eeg is not None:
                results["successful"].append({
                    "subject": sub,
                    "behavior_events": len(behavior) if not behavior.empty else 0,
                    "eeg_channels": len(eeg.ch_names),
                    "eeg_duration": f"{eeg.times[-1]:.1f}s",
                    "sampling_rate": f"{eeg.info['sfreq']:.0f}Hz"
                })
                print(f" {sub}: {len(eeg.ch_names)} channels, {eeg.times[-1]:.1f}s")
            else:
                results["failed"].append({"subject": sub, "reason": "No EEG data"})
                print(f" {sub}: No EEG data")
                
        except Exception as e:
            results["failed"].append({"subject": sub, "reason": str(e)})
            print(f" {sub}: {e}")
        
        print()  # Empty line for readability
    
    # Summary
    n_success = len(results["successful"])
    n_total = len(subjects_to_process)
    print(f" Processing complete: {n_success}/{n_total} subjects successful")
    
    if results["failed"]:
        print(f" Failed subjects: {[f['subject'] for f in results['failed']]}")
    
    return results

# Test usage
if __name__ == "__main__":
    # Process all subjects with memory management
    results = load_all_subjects()
    
    # To load specific subjects for analysis:
    # results = load_all_subjects(subjects_only=["sub-01", "sub-02", "sub-03"])
    
    # To actually load a subject's data into memory for analysis:
    # behavior = load_behavior("sub-01")
    # eeg = load_eeg("sub-01")  # This will load from fast .fif cache
    
    print(f"\n All cached files saved in: {PROCESSED_DIR}")
    print(" To load a subject for analysis: eeg = load_eeg('sub-01')")