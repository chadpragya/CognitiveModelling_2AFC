import json
from pathlib import Path

RAW_DATA = Path("data/raw/ds002739-1.0.0")

index = {}

for sub in sorted(RAW_DATA.glob("sub-*")):
    subj_id = sub.name  # "sub-01"

    # Initialize entry for this subject
    index[subj_id] = {"anat": [], "func": [], "eeg": []}

    # 1. Anatomical MRI (anat)
    anat_dir = sub / "anat"
    if anat_dir.exists():
        anat_files = sorted(anat_dir.glob("*.nii.gz"))
        index[subj_id]["anat"].extend([str(f) for f in anat_files])

    # 2. Functional MRI (func)
    func_dir = sub / "func"
    if func_dir.exists():
        # BOLD fMRI runs
        func_files = sorted(func_dir.glob("*.nii.gz"))
        index[subj_id]["func"].extend([str(f) for f in func_files])

        # Event timing files (behavioral data)
        event_files = sorted(func_dir.glob("*.tsv"))
        index[subj_id]["func"].extend([str(f) for f in event_files])

    # 3. EEG (mat files)
    eeg_dir = sub / "eeg"
    if eeg_dir.exists():
        eeg_files = sorted(eeg_dir.glob("*.mat"))
        index[subj_id]["eeg"].extend([str(f) for f in eeg_files])

# Save the index to JSON
out_path = Path("data/interim/data_index.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    json.dump(index, f, indent=2)

print(f"Indexing complete! Saved to {out_path}")