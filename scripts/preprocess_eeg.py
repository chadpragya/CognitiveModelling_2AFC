from pathlib import Path
import scipy.io as sio
import mne
import numpy as np 

# Define paths
RAW_DATA = Path("data/raw/ds002739-1.0.0")
INTERIM_EEG = Path("data/interim/eeg")
INTERIM_EEG.mkdir(parents=True, exist_ok=True)  # make sure output folder exists

def preprocess_one_eeg(file_path: Path, save_path: Path):
    mat = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)

    # --- EEG struct ---
    if "EEGdata" not in mat:
        print(f"⚠️ No 'EEGdata' found in {file_path.name}. Keys: {list(mat.keys())}")
        return

    eeg_struct = mat["EEGdata"]

    # Extract fs (prefer from EEGdata, fallback to top-level)
    fs = None
    if hasattr(eeg_struct, "fs"):
        fs = int(eeg_struct.fs)
    elif "fs" in mat:
        fs_val = mat["fs"]
        fs = int(fs_val.item() if hasattr(fs_val, "item") else fs_val)
    else:
        raise ValueError(f"No sampling rate (fs) found in {file_path.name}")

    # Extract EEG data
    if hasattr(eeg_struct, "Y"):
        data = eeg_struct.Y
    else:
        raise ValueError(f"No 'Y' field in EEGdata of {file_path.name}")

    # Ensure numpy array
    data = np.array(data, dtype="float64")

    # Shape correction: (n_channels, n_samples)
    if data.shape[0] > data.shape[1]:
        data = data.T

    n_channels, n_samples = data.shape
    print(f"{file_path.name}: {n_channels} channels, {n_samples} samples, fs={fs}")

    # --- Build MNE Raw object ---
    ch_names = [f"EEG{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    # --- Preprocessing ---
    raw.filter(1., 40., fir_design="firwin")
    raw.notch_filter(50.)

    # --- Save ---
    raw.save(save_path, overwrite=True)
    print(f"✅ Saved preprocessed EEG to {save_path}")

def main():
    # Find all EEG_data .mat files in dataset
    eeg_files = list(RAW_DATA.rglob("EEG_data_sub-*_run-*.mat"))

    for eeg_file in eeg_files:
        subj = eeg_file.parts[-3]   # e.g., sub-01
        run = eeg_file.stem.split("_")[-1]   # e.g., run-01
        save_name = f"{subj}_{run}_eeg_raw.fif"
        save_path = INTERIM_EEG / save_name

        preprocess_one_eeg(eeg_file, save_path)


if __name__ == "__main__":
    main()
