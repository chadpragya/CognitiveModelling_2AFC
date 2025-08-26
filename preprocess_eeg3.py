from pathlib import Path
import scipy.io as sio
import mne
import numpy as np
import h5py
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
RAW_DATA = Path("data/raw/ds002739-1.0.0")
INTERIM_EEG = Path("data/interim/eeg")
INTERIM_EEG.mkdir(parents=True, exist_ok=True)

def load_mat_file(file_path: Path):
    """Try loading .mat with multiple methods and parameters."""
    # Check if file exists and has reasonable size
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    file_size = file_path.stat().st_size
    logger.info(f"Attempting to load {file_path.name} (size: {file_size} bytes)")
    
    if file_size == 0:
        raise ValueError(f"File {file_path.name} is empty")
    
    # Try multiple loading strategies
    loading_strategies = [
        # Strategy 1: Default scipy parameters
        lambda: sio.loadmat(file_path, squeeze_me=True, struct_as_record=False),
        # Strategy 2: Without squeeze_me (sometimes helps with corruption)
        lambda: sio.loadmat(file_path, squeeze_me=False, struct_as_record=False),
        # Strategy 3: With different parameters
        lambda: sio.loadmat(file_path, squeeze_me=True, struct_as_record=True),
        # Strategy 4: Simplest possible loading
        lambda: sio.loadmat(file_path),
    ]
    
    # Try scipy strategies first
    for i, strategy in enumerate(loading_strategies, 1):
        try:
            logger.info(f"Trying scipy strategy {i} for {file_path.name}")
            mat_data = strategy()
            logger.info(f"Successfully loaded {file_path.name} with scipy strategy {i}")
            return mat_data
        except (OSError, IOError) as e:
            logger.warning(f"Scipy strategy {i} failed for {file_path.name} with I/O error: {e}")
            continue
        except NotImplementedError as e:
            logger.info(f"Scipy strategy {i} failed for {file_path.name} (v7.3 format): {e}")
            break  # Try h5py for v7.3 files
        except Exception as e:
            logger.warning(f"Scipy strategy {i} failed for {file_path.name}: {e}")
            continue
    
    # Try h5py for MATLAB v7.3 files
    try:
        logger.info(f"Trying h5py for {file_path.name}")
        # First, verify it's actually an HDF5 file
        if not h5py.is_hdf5(file_path):
            logger.error(f"{file_path.name} is not a valid HDF5 file")
            raise ValueError(f"File {file_path.name} is corrupted or not a valid MATLAB file")
        
        with h5py.File(file_path, "r") as f:
            mat_data = {}
            for key in f.keys():
                try:
                    if isinstance(f[key], h5py.Dataset):
                        # Handle different data types
                        dataset = f[key]
                        if dataset.dtype.kind == 'O':  # Object arrays (references)
                            # Handle MATLAB cell arrays or object references
                            mat_data[key] = dataset[()]
                        else:
                            mat_data[key] = dataset[()]
                    elif isinstance(f[key], h5py.Group):
                        # Handle nested structures
                        mat_data[key] = _load_h5py_group(f[key])
                    else:
                        mat_data[key] = f[key]
                except Exception as e:
                    logger.warning(f"Could not load key '{key}' from {file_path.name}: {e}")
                    continue
            
            if not mat_data:
                raise ValueError(f"No data could be extracted from {file_path.name}")
                
            logger.info(f"Successfully loaded {file_path.name} with h5py")
            return mat_data
            
    except Exception as e:
        logger.error(f"h5py also failed for {file_path.name}: {e}")
    
    # Final attempt: try to read file in binary mode to check for corruption
    try:
        with open(file_path, 'rb') as f:
            header = f.read(128)  # Read first 128 bytes
            if len(header) < 128:
                raise ValueError(f"File {file_path.name} appears to be truncated")
            
            # Check MATLAB file signature
            if header[:4] != b'MATL' and b'MATLAB' not in header[:128]:
                raise ValueError(f"File {file_path.name} does not appear to be a MATLAB file")
                
    except Exception as e:
        logger.error(f"Binary file check failed for {file_path.name}: {e}")
    
    raise ValueError(f"Could not load {file_path.name} with any method. File may be corrupted.")

def _load_h5py_group(group):
    """Recursively load h5py group data."""
    data = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Dataset):
            data[key] = group[key][()]
        elif isinstance(group[key], h5py.Group):
            data[key] = _load_h5py_group(group[key])
    return data

def extract_nested_data(obj, key_path):
    """Helper function to extract data from nested structures."""
    current = obj
    for key in key_path:
        if hasattr(current, key):
            current = getattr(current, key)
        elif isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

def preprocess_one_eeg(file_path: Path):
    """Load and preprocess one EEG run, return MNE Raw object."""
    try:
        logger.info(f"Processing {file_path.name}")
        mat = load_mat_file(file_path)
        
        # Debug: print all keys
        logger.info(f"Available keys in {file_path.name}: {list(mat.keys())}")
        
        if "EEGdata" not in mat:
            logger.warning(f"⚠️ No 'EEGdata' found in {file_path.name}. Keys: {list(mat.keys())}")
            return None

        eeg_struct = mat["EEGdata"]
        logger.info(f"EEGdata type: {type(eeg_struct)}")
        
        # Debug: if it's a structured array, print its fields
        if hasattr(eeg_struct, 'dtype') and eeg_struct.dtype.names:
            logger.info(f"EEGdata fields: {eeg_struct.dtype.names}")

        # --- Extract sampling rate with more robust handling ---
        fs = None
        
        # Try multiple ways to extract fs
        fs_candidates = [
            lambda: int(eeg_struct.fs) if hasattr(eeg_struct, "fs") else None,
            lambda: int(mat["fs"].item()) if "fs" in mat and hasattr(mat["fs"], "item") else None,
            lambda: int(mat["fs"]) if "fs" in mat else None,
            lambda: int(eeg_struct[0].fs) if hasattr(eeg_struct, '__len__') and len(eeg_struct) > 0 and hasattr(eeg_struct[0], 'fs') else None,
        ]
        
        for i, candidate in enumerate(fs_candidates):
            try:
                fs = candidate()
                if fs is not None:
                    logger.info(f"Found fs using method {i+1}: {fs}")
                    break
            except Exception as e:
                logger.debug(f"Method {i+1} failed for fs extraction: {e}")
                continue
        
        if fs is None:
            logger.error(f"No sampling rate (fs) found in {file_path.name}")
            logger.error(f"EEGdata attributes: {dir(eeg_struct) if hasattr(eeg_struct, '__dict__') else 'No attributes'}")
            raise ValueError(f"No sampling rate (fs) found in {file_path.name}")

        # --- Extract EEG data with more robust handling ---
        data = None
        
        # Try multiple ways to extract data
        data_candidates = [
            lambda: eeg_struct.Y if hasattr(eeg_struct, "Y") else None,
            lambda: eeg_struct['Y'] if isinstance(eeg_struct, dict) and 'Y' in eeg_struct else None,
            lambda: eeg_struct[0].Y if hasattr(eeg_struct, '__len__') and len(eeg_struct) > 0 and hasattr(eeg_struct[0], 'Y') else None,
            lambda: eeg_struct.data if hasattr(eeg_struct, "data") else None,
            lambda: eeg_struct['data'] if isinstance(eeg_struct, dict) and 'data' in eeg_struct else None,
        ]
        
        for i, candidate in enumerate(data_candidates):
            try:
                data = candidate()
                if data is not None:
                    logger.info(f"Found data using method {i+1}")
                    break
            except Exception as e:
                logger.debug(f"Method {i+1} failed for data extraction: {e}")
                continue
        
        if data is None:
            logger.error(f"No EEG data found in {file_path.name}")
            logger.error(f"EEGdata attributes: {dir(eeg_struct) if hasattr(eeg_struct, '__dict__') else 'No attributes'}")
            raise ValueError(f"No EEG data found in {file_path.name}")

        # Convert to numpy array and ensure proper dtype
        data = np.array(data, dtype="float64")
        
        # Handle potential issues with data shape
        if data.ndim == 0:
            logger.error(f"Data is scalar in {file_path.name}")
            raise ValueError(f"Data is scalar in {file_path.name}")
        elif data.ndim == 1:
            logger.warning(f"Data is 1D in {file_path.name}, reshaping to (1, n_samples)")
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            logger.warning(f"Data is {data.ndim}D in {file_path.name}, taking first 2 dimensions")
            data = data.reshape(data.shape[0], -1)

        # Shape correction: (n_channels, n_samples)
        if data.shape[0] > data.shape[1]:
            logger.info(f"Transposing data from {data.shape} to {data.T.shape}")
            data = data.T

        n_channels, n_samples = data.shape
        logger.info(f"{file_path.name}: {n_channels} channels, {n_samples} samples, fs={fs}")

        # Validate data
        if n_channels == 0 or n_samples == 0:
            raise ValueError(f"Invalid data dimensions: {n_channels} x {n_samples}")
        
        if not np.isfinite(data).all():
            logger.warning(f"Non-finite values found in {file_path.name}")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Build MNE Raw object ---
        ch_names = [f"EEG{i+1:03d}" for i in range(n_channels)]  # Zero-padded naming
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)

        # --- Preprocessing with error handling ---
        try:
            logger.info(f"Applying bandpass filter (1-40 Hz) to {file_path.name}")
            raw.filter(1., 40., fir_design="firwin", verbose=False)
        except Exception as e:
            logger.error(f"Bandpass filtering failed for {file_path.name}: {e}")
            raise

        try:
            logger.info(f"Applying notch filter (50 Hz) to {file_path.name}")
            raw.notch_filter(50., verbose=False)
        except Exception as e:
            logger.error(f"Notch filtering failed for {file_path.name}: {e}")
            raise

        logger.info(f"Successfully preprocessed {file_path.name}")
        return raw
        
    except Exception as e:
        logger.error(f"Failed to process {file_path.name}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def main():
    try:
        # Find all EEG_data .mat files
        eeg_files = list(RAW_DATA.rglob("EEG_data_sub-*_run-*.mat"))
        logger.info(f"Found {len(eeg_files)} EEG files")
        
        if not eeg_files:
            logger.error(f"No EEG files found in {RAW_DATA}")
            return

        # Group by subject
        subjects = {}
        for eeg_file in eeg_files:
            try:
                subj = eeg_file.parts[-3]  # e.g., sub-01
                subjects.setdefault(subj, []).append(eeg_file)
            except IndexError:
                logger.warning(f"Could not extract subject ID from path: {eeg_file}")
                continue

        logger.info(f"Processing {len(subjects)} subjects: {list(subjects.keys())}")

        successful_subjects = 0
        failed_subjects = []

        for subj_idx, (subj, files) in enumerate(subjects.items(), 1):
            try:
                logger.info(f"Processing subject {subj_idx}/{len(subjects)}: {subj}")
                raws = []
                
                for run_idx, eeg_file in enumerate(sorted(files), 1):
                    logger.info(f"  Processing run {run_idx}/{len(files)}: {eeg_file.name}")
                    
                    # Check file integrity before processing
                    try:
                        file_size = eeg_file.stat().st_size
                        if file_size == 0:
                            logger.error(f"File {eeg_file.name} is empty, skipping")
                            continue
                        elif file_size < 1000:  # Suspiciously small
                            logger.warning(f"File {eeg_file.name} is very small ({file_size} bytes)")
                    except Exception as e:
                        logger.error(f"Cannot access file {eeg_file.name}: {e}")
                        continue
                    
                    raw = preprocess_one_eeg(eeg_file)
                    if raw is not None:
                        raws.append(raw)
                    else:
                        logger.warning(f"Failed to process run {run_idx} for {subj}")
                        # Continue processing other files instead of stopping

                if not raws:
                    logger.warning(f"No valid runs for {subj}")
                    failed_subjects.append(subj)
                    continue

                # Check sampling rates consistency
                sfreqs = [raw.info['sfreq'] for raw in raws]
                if len(set(sfreqs)) > 1:
                    logger.warning(f"Inconsistent sampling rates for {subj}: {sfreqs}")
                    # Resample all to the minimum sampling rate
                    min_sfreq = min(sfreqs)
                    logger.info(f"Resampling all runs to {min_sfreq} Hz")
                    for i, raw in enumerate(raws):
                        if raw.info['sfreq'] != min_sfreq:
                            raws[i] = raw.resample(min_sfreq, verbose=False)

                # Concatenate all runs
                logger.info(f"Concatenating {len(raws)} runs for {subj}")
                raw_concat = mne.concatenate_raws(raws, preload=True, verbose=False)

                # Save one fif per subject
                save_name = f"{subj}_eeg_raw.fif"
                save_path = INTERIM_EEG / save_name
                
                logger.info(f"Saving {subj} to {save_path}")
                raw_concat.save(save_path, overwrite=True, verbose=False)
                
                logger.info(f"Successfully processed {subj} ({subj_idx}/{len(subjects)})")
                successful_subjects += 1
                
                # Clean up memory
                del raw_concat, raws
                
            except Exception as e:
                logger.error(f"Failed to process subject {subj}: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                failed_subjects.append(subj)
                continue

        # Summary
        logger.info(f"\n{'='*50}")
        logger.info(f"PROCESSING SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total subjects: {len(subjects)}")
        logger.info(f"Successfully processed: {successful_subjects}")
        logger.info(f"Failed: {len(failed_subjects)}")
        if failed_subjects:
            logger.info(f"Failed subjects: {failed_subjects}")
        logger.info(f"{'='*50}")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()