from pathlib import Path
import scipy.io as sio
import mne
import numpy as np
import h5py

# Define paths
RAW_DATA = Path("data/raw/ds002739-1.0.0")
INTERIM_EEG = Path("data/interim/eeg")
INTERIM_EEG.mkdir(parents=True, exist_ok=True)


def load_mat_file(file_path: Path):
    """Try loading .mat with scipy, fallback to h5py if v7.3 (HDF5)."""
    try:
        return sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        with h5py.File(file_path, "r") as f:
            return {k: f[k] for k in f.keys()}


def preprocess_one_eeg(file_path: Path):
    """Load and preprocess one EEG run, return MNE Raw object."""
    mat = load_mat_file(file_path)

    if "EEGdata" not in mat:
        print(f"⚠️ No 'EEGdata' found in {file_path.name}. Keys: {list(mat.keys())}")
        return None

    eeg_struct = mat["EEGdata"]

    # --- Extract sampling rate ---
    fs = None
    if hasattr(eeg_struct, "fs"):
        fs = int(eeg_struct.fs)
    elif "fs" in mat:
        fs_val = mat["fs"]
        fs = int(fs_val.item() if hasattr(fs_val, "item") else fs_val)
    else:
        raise ValueError(f"No sampling rate (fs) found in {file_path.name}")

    # --- Extract EEG data ---
    if hasattr(eeg_struct, "Y"):
        data = eeg_struct.Y
    else:
        raise ValueError(f"No 'Y' field in EEGdata of {file_path.name}")

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

    return raw


def main():
    # Find all EEG_data .mat files
    eeg_files = list(RAW_DATA.rglob("EEG_data_sub-*_run-*.mat"))

    # Group by subject
    subjects = {}
    for eeg_file in eeg_files:
        subj = eeg_file.parts[-3]  # e.g., sub-01
        subjects.setdefault(subj, []).append(eeg_file)

    for subj, files in subjects.items():
        raws = []
        for eeg_file in sorted(files):  # ensure runs in order
            raw = preprocess_one_eeg(eeg_file)
            if raw is not None:
                raws.append(raw)

        if not raws:
            print(f"⚠️ No valid runs for {subj}")
            continue

        # Concatenate all runs
        raw_concat = mne.concatenate_raws(raws)

        # Save one fif per subject
        save_name = f"{subj}_eeg_raw.fif"
        save_path = INTERIM_EEG / save_name
        raw_concat.save(save_path, overwrite=True)
        print(f"✅ Saved concatenated EEG for {subj} -> {save_path}")


if __name__ == "__main__":
    main()