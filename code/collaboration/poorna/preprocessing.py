#!/usr/bin/env python
# coding: utf-8

# # imports

# In[40]:


#!/usr/bin/env python3
"""
Simple SEED-DV EEG Preprocessing Pipeline
Standard preprocessing steps without unnecessary complexity
"""

import os
import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from mne.preprocessing import ICA
from IPython.display import display


# In[5]:


# Set MNE logging to reduce output
mne.set_log_level('DEBUG')


# In[6]:


# To visualise locally do pip install pytq5 


# In[7]:


get_ipython().run_line_magic('matplotlib', 'qt')


# In[8]:


input_dir = Path("../../../SEED-DV/EEG")
output_dir = Path("preprocessed_eeg")
output_dir.mkdir(exist_ok=True)

# Basic parameters
sfreq = 200  # Original sampling rate
target_sfreq = 200  # Target sampling rate
n_channels = 62

# Standard 62-channel names
ch_names = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]


# # find_subject_files()

# In[9]:


def find_subject_files():
    """Find all subject .npy files."""
    files = list(input_dir.glob("*.npy"))
    subjects = {}

    for file in files:
        subject_id = file.stem
        subjects[subject_id] = file

    return subjects


# In[10]:


subjects = find_subject_files()
print(subjects)


# # load_data()

# In[11]:


def load_data(file_path):
    """Load and reshape SEED-DV data."""
    print(f"  Loading: {file_path.name}")
    data = np.load(file_path)  # Shape: (7_videos, 62_channels, timepoints)

    # Reshape to (channels, timepoints)
    n_videos, n_channels, n_timepoints = data.shape
    reshaped_data = data.transpose(1, 0, 2).reshape(n_channels, -1)

    print(f"  Original shape: {data.shape}")
    print(f"  Reshaped to: {reshaped_data.shape}")

    return reshaped_data


# In[12]:


reshaped_data = load_data(subjects["sub1"])


# In[13]:


import numpy as np

# Paths
locs_path = "../../../SEED-DV/channel_62_pos.locs"
sfp_path = "../../../SEED-DV/channel_62_pos.sfp"

with open(locs_path, 'r') as infile, open(sfp_path, 'w') as outfile:
    for line in infile:
        if line.strip() == "":
            continue
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        _, angle_str, radius_str, label = parts
        angle_deg = float(angle_str)
        radius = float(radius_str)

        # EEGLAB radius (0–0.5) → scale to 0–1
        scaled_radius = radius * 2.0

        # Rotate 90° (EEGLAB 0° = right, MNE 0° = front)
        theta = np.deg2rad(angle_deg + 90)

        # Convert to 2D Cartesian
        x = scaled_radius * np.cos(theta)
        y = scaled_radius * np.sin(theta)

        # Project to upper hemisphere
        z = np.sqrt(max(0, 1 - x**2 - y**2))

        # Flip X-axis to match MNE coordinate system
        x = -x

        outfile.write(f"{label} {x:.6f} {y:.6f} {z:.6f}\n")

print(f"✅ Converted and corrected .locs → .sfp: {sfp_path}")


# In[14]:


import numpy as np

# Paths
locs_path = "../../../SEED-DV/channel_62_pos.locs"
sfp_path = "../../../SEED-DV/channel_62_pos.sfp"

with open(locs_path, 'r') as infile, open(sfp_path, 'w') as outfile:
    for line in infile:
        if line.strip() == "":
            continue
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        _, angle_str, radius_str, label = parts
        angle_deg = float(angle_str)
        radius = float(radius_str)

        # Convert to radians, rotate EEGLAB to MNE (EEGLAB top = 0°, MNE front = 0°)
        theta = np.deg2rad(angle_deg + 90)

        # 2D Cartesian (still on unit disc)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        # Project onto hemisphere (z up)
        z = np.sqrt(max(0, 1 - x**2 - y**2))

        outfile.write(f"{label} {x:.6f} {y:.6f} {z:.6f}\n")

print(f"✅ Converted .locs → .sfp for MNE: {sfp_path}")


# In[15]:


import numpy as np

locs_path = "../../../SEED-DV/channel_62_pos.locs"
sfp_path = "../../../SEED-DV/channel_62_pos.sfp"

radius_scale = 2.0  # <-- Try 1.5 or 2.0, adjust as needed

with open(locs_path, 'r') as infile, open(sfp_path, 'w') as outfile:
    for line in infile:
        if line.strip() == "":
            continue
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        _, angle_str, radius_str, label = parts
        angle_deg = float(angle_str)
        radius = float(radius_str) * radius_scale  # ⬅️ Scaling radius here

        # Rotate EEGLAB → MNE: EEGLAB top = 0°, MNE front = 0°
        theta = np.deg2rad(angle_deg + 90)

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.sqrt(max(0, 1 - x**2 - y**2))  # Hemisphere projection

        outfile.write(f"{label} {x:.6f} {y:.6f} {z:.6f}\n")
print(f"✅ Converted .locs → .sfp for MNE: {sfp_path}")


# In[16]:


with open(sfp_path,"r") as file:
    content = file.read()
    print(content)


# # create_raw_object()

# In[17]:


def create_raw_object(data):
    """Create MNE Raw object."""
    # Create info
    info = mne.create_info(
        ch_names=ch_names[:n_channels],
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels
    )

    # Create Raw object
    raw = mne.io.RawArray(data, info)

    # Set standard montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    return raw


# ## custom using read_custom_montage() file

# In[18]:


def create_raw_object(data):
    """Create MNE Raw object."""
    # Create info
    info = mne.create_info(
        ch_names=ch_names[:n_channels],
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels
    )

    # Create Raw object
    raw = mne.io.RawArray(data, info)

    # Set standard montage
    montage = mne.channels.read_custom_montage("../../../SEED-DV/channel_62_pos.locs")
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    return raw


# In[19]:


raw_data = create_raw_object(reshaped_data)


# In[20]:


raw_data.plot(
    n_channels=30,        # number of channels to display at once
    scalings='auto',      # or {'eeg': 20e-6} to manually set
    duration=10.0,        # time (in seconds) to show per window
    start=0.0,            # start time (in seconds)
    show=True,            # display immediately
    block=True,           # block execution until closed
    title='EEG Raw Data'  # plot window title
)


# In[21]:


raw_data.info["bads"]


# In[22]:


raw_data.plot_sensors(kind="topomap",show_names= True)


# # apply_filters()

# In[23]:


raw_data.compute_psd().plot()


# In[24]:


# EEG Filtering Justification:
# - We apply a bandpass filter from 0.5 to 40 Hz:
#   This range preserves the main EEG bands (theta, alpha, beta) relevant for cognitive and motor tasks.
#   It removes slow drifts (<0.5 Hz) and high-frequency muscle artifacts and noise (>40 Hz, including gamma).
# - A 50 Hz notch filter is used to remove power line interference, common in many countries.
# - These settings are appropriate for training models that decode text or cognitive intent from EEG signals.

def apply_filters(raw):
    """
    Apply preprocessing filters to EEG data.
    """
    print("  Applying filters...")

    # Bandpass filter between 0.5 and 40 Hz to retain useful EEG bands
    raw.filter(
        l_freq=0.5,             # Remove slow drifts
        h_freq=40.0,            # Exclude high-frequency noise (gamma, EMG)
        fir_design='firwin',    # Linear-phase FIR filter
        skip_by_annotation='edge'  # Avoid filtering edges marked as bad
    )

    # Notch filter at 50 Hz to eliminate power line interference
    raw.notch_filter(
        freqs=50.0,
        fir_design='firwin'
    )

    return raw


# In[25]:


filtered_data = apply_filters(raw_data)


# In[26]:


filtered_data.compute_psd().plot()


# # apply_reference()

# In[28]:


def apply_reference(raw):
    """Apply common average reference."""
    print("  Applying common average reference...")
    raw.set_eeg_reference(ref_channels='average', projection=True)
    raw.apply_proj()
    return raw


# In[29]:


rereferenced_data = apply_reference(filtered_data)


# In[30]:


rereferenced_data.compute_psd().plot()


# # detect_bad_channels()

# In[31]:


def detect_bad_channels(raw):
    """Detects bad EEG channels based on signal variance and visualizes before and after interpolation."""
    print("  Detecting bad channels...")

    # Pick only EEG channels (ignore EOG, ECG, etc.)
    eeg_data = raw.copy().pick(picks="eeg")

    # Compute standard deviation across time for each EEG channel
    data = eeg_data.get_data()
    channel_stds = np.std(data, axis=1)
    mean_std = np.mean(channel_stds)
    std_threshold = 3 * np.std(channel_stds)

    bad_channels = []
    for ch_name, ch_std in zip(eeg_data.ch_names, channel_stds):
        if ch_std > mean_std + std_threshold or ch_std < mean_std - std_threshold:
            bad_channels.append(ch_name)

    # Mark bads in raw.info['bads']
    if bad_channels:
        print(f"    Found {len(bad_channels)} bad channels: {bad_channels}")
        eeg_data.info['bads'] = bad_channels

        # Visualize before and after interpolation
        eeg_data_interp = eeg_data.copy().interpolate_bads(reset_bads=False)

        for title, data in zip(["Original (with bads)", "Interpolated"], [eeg_data, eeg_data_interp]):
            with mne.viz.use_browser_backend("matplotlib"):
                fig = data.plot(butterfly=True, color="#00000022", bad_color="r")
            fig.subplots_adjust(top=0.9)
            fig.suptitle(title, size="xx-large", weight="bold")

        # Check for NaNs or Infs before proceeding
        if np.isnan(eeg_data_interp.get_data()).any() or np.isinf(eeg_data_interp.get_data()).any():
            print("❗ Interpolated data contains NaNs or Infs — skipping this subject")
        else:
            print("→ Interpolation complete and verified.")
            raw.info['bads'] = bad_channels
            raw.interpolate_bads(reset_bads=True)
    else:
        print("    No bad channels detected")

    return raw


# In[32]:


bad_channels_detected = detect_bad_channels(rereferenced_data)


# # apply_ica()

# In[39]:


def apply_ica(raw, manual=True):
    """
    Apply ICA to remove artifacts from EEG data, with optional manual component rejection.
    """
    print("  Applying ICA...")

    ica = ICA(n_components=20, random_state=42, method='fastica')
    ica.fit(raw)

    try:
        eog_inds, eog_scores = ica.find_bads_eog(raw, threshold=2.0)
        if eog_inds:
            print(f"    Auto-detected EOG components: {eog_inds}")
            ica.exclude = eog_inds
    except Exception as e:
        print(f"    EOG detection failed: {e}")

    if manual:
        # Display ICA components to the user
        ica.plot_components()
        # ica.plot_sources(raw)

        # Ask user to enter indices they want to exclude
        exclude_input = input("Enter component indices to exclude (comma-separated): ")
        try:
            exclude = [int(x.strip()) for x in exclude_input.split(',') if x.strip() != '']
            ica.exclude = exclude
            print("    Manually excluded components:", ica.exclude)
        except:
            print("    Invalid input. No components were excluded.")

    raw = ica.apply(raw)
    return raw


# In[40]:


ica_data = apply_ica(bad_channels_detected)


# In[46]:


def apply_ica(raw):
    """
    Apply ICA with optional manual exclusion. Use in Jupyter by visually inspecting components.
    """
    print("  Applying ICA...")

    ica = ICA(n_components=20, random_state=42, method='fastica')
    ica.fit(raw)

    try:
        eog_inds, _ = ica.find_bads_eog(raw, threshold=2.0)
        if eog_inds:
            print(f"    Auto-detected EOG components: {eog_inds}")
            ica.exclude = eog_inds
    except Exception as e:
        print(f"    EOG detection failed: {e}")

    # Let user visually inspect and decide
    fig = ica.plot_components(inst=raw)
    display(fig)

    # Ask user to enter indices (in notebook)
    exclude_input = input("Enter component indices to exclude (comma-separated): ")

    try:
        exclude = [int(x.strip()) for x in exclude_input.split(',') if x.strip()]
        ica.exclude = exclude
        print("Manually excluded components:", ica.exclude)
    except Exception as e:
        print("Invalid input. No components excluded:", e)

    raw_clean = ica.apply(raw.copy())
    return raw_clean, ica


# In[47]:


raw_clean, ica = apply_ica(bad_channels_detected)


# In[49]:


fig = ica.plot_components(inst=raw_clean)
display(fig)


# In[51]:


ica.plot_sources(raw_clean)


# # create_epochs()

# In[52]:


def create_epochs( raw):
    """Create 13s epochs per block and mark end of 3s hint within each epoch."""
    print("  Creating block-level epochs with hint-end markers...")

    sfreq = raw.info['sfreq']
    block_duration = 13.0
    hint_duration = 3.0
    n_blocks = 40

    events = []

    for i in range(n_blocks):
        block_start = i * block_duration
        block_start_sample = int(block_start * sfreq)
        hint_end_sample = block_start_sample + int(hint_duration * sfreq)

        # Event 1: Start of epoch (we'll use this to define the epoch)
        events.append([block_start_sample, 0, 1])   # event_id=1 means 'block_start'

        # Event 2: Hint ends at 3s after block start (within epoch)
        events.append([hint_end_sample, 0, 2])      # event_id=2 means 'hint_end'

    events = np.array(events)
    print(f"    Created {n_blocks} epochs and {n_blocks} hint-end markers")

    # Select only 'block_start' events to create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id={'block_start': 1, 'hint_end': 2},
        tmin=0.0,
        tmax=block_duration,
        baseline=None,
        preload=True
    )

    print(f"    Created {len(epochs)} epochs (13s each)")
    return epochs


# In[53]:


epochs = create_epochs(raw_clean)


# In[54]:


epochs.plot()


# In[48]:


for ev in epochs.events:
    print(ev)  # Should show 2 events per block: [sample, 0, event_id]


# In[55]:


epochs


# # normalize_data()

# In[56]:


def normalize_data( epochs):
    """Simple z-score normalization."""
    print("  Normalizing data...")

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)

    # Z-score normalization per channel
    for ch_idx in range(data.shape[1]):
        channel_data = data[:, ch_idx, :].flatten()
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)

        if std_val > 0:
            data[:, ch_idx, :] = (data[:, ch_idx, :] - mean_val) / std_val

    epochs._data = data
    return epochs


# In[57]:


normalized_epochs = normalize_data(epochs)


# In[58]:


normalized_epochs.plot(scalings="auto")


# # save_data()

# In[58]:


def save_data( epochs, subject_id):
    """Save preprocessed data."""
    print("  Saving data...")

    data = epochs.get_data()

    # Save numpy array
    output_file = output_dir / f"{subject_id}_preprocessed.npy"
    np.save(output_file, data)

    # Save metadata
    metadata = {
        'subject_id': subject_id,
        'shape': data.shape,
        'sfreq': epochs.info['sfreq'],
        'ch_names': epochs.ch_names,
        'n_epochs': len(epochs),
        'preprocessing_steps': [
            'bandpass_filter_0.5_40Hz',
            'notch_filter_50Hz',
            'common_average_reference',
            'bad_channel_detection',
            'ICA_artifact_removal',
            'epoching_13s',
            'z_score_normalization'
        ]
    }

    metadata_file = output_dir / f"{subject_id}_metadata.npy"
    np.save(metadata_file, metadata)

    print(f"    Saved: {output_file}")
    print(f"    Shape: {data.shape}")

    return output_file


# In[59]:


output_file = save_data(normalized_epochs,"sub1")


# # plot_comparision()

# In[77]:


def plot_comparison(raw_before, raw_after, subject_id):
    """
    Plot comparison of EEG data before and after preprocessing.

    - Left subplot: Power Spectral Density (PSD) before vs. after
    - Right subplot: Time-series of a selected channel (e.g., Cz) before vs. after

    Saves the resulting plot as a PNG.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import mne

    print("  Creating comparison plot...")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PSD plot
        psd_before, freqs_before = mne.time_frequency.psd_array_welch(
            raw_before.get_data(), sfreq=raw_before.info['sfreq'], fmin=0.5, fmax=80
        )
        psd_after, freqs_after = mne.time_frequency.psd_array_welch(
            raw_after.get_data(), sfreq=raw_after.info['sfreq'], fmin=0.5, fmax=80
        )

        axes[0].semilogy(freqs_before, np.mean(psd_before, axis=0), 'r-', label='Before', alpha=0.7)
        axes[0].semilogy(freqs_after, np.mean(psd_after, axis=0), 'b-', label='After', alpha=0.7)
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Power Spectral Density (V²/Hz)')
        axes[0].set_title('Power Spectral Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Time series plot (first 10 seconds)
        duration = 10  # seconds
        n_samples = int(duration * raw_before.info['sfreq'])
        time_vec = np.arange(n_samples) / raw_before.info['sfreq']

        # Choose 'Cz' channel if available, else use the first channel
        ch_idx = raw_before.ch_names.index('F3') if 'F3' in raw_before.ch_names else 0
        ch_name = raw_before.ch_names[ch_idx]

        before_data = raw_before.get_data()[ch_idx, :n_samples] * 1e6  # in µV
        after_data = raw_after.get_data()[ch_idx, :n_samples] * 1e6    # in µV

        axes[1].plot(time_vec, before_data, 'r-', label='Before', alpha=0.7)
        axes[1].plot(time_vec, after_data, 'b-', label='After', alpha=0.7)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude (µV)')
        axes[1].set_title(f'Time Series - Channel: {ch_name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save comparison figure
        plot_file = output_dir / f"{subject_id}_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    Saved plot: {plot_file}")

    except Exception as e:
        print(f"    Error creating plot: {e}")


# In[78]:


plot_comparison(raw_data,ica_data,"sub1")


# In[ ]:


def process_subject( subject_id, file_path):
    """Process a single subject."""
    print(f"\nProcessing subject: {subject_id}")

    try:
        # Load data
        data = load_data(file_path)

        # Create Raw object
        raw = create_raw_object(data)
        raw_original = raw.copy()

        # Apply preprocessing steps
        raw = apply_filters(raw)
        raw = apply_reference(raw)
        raw = detect_bad_channels(raw)
        raw = apply_ica(raw)

        # Create epochs
        epochs = create_epochs(raw)

        if len(epochs) == 0:
            print(f"  No valid epochs for {subject_id}")
            return None

        # Normalise
        epochs = normalize_data(epochs)

        # Save data
        output_file = save_data(epochs, subject_id)

        # Create comparison plot
        plot_comparison(raw_original, raw, subject_id)

        print(f"✓ Successfully processed {subject_id}")
        print(f"  Final shape: {epochs.get_data().shape}")

        return output_file

    except Exception as e:
        print(f"✗ Error processing {subject_id}: {e}")
        return None


# In[ ]:


def run_pipeline():
    """Run the complete preprocessing pipeline."""
    print("Simple SEED-DV EEG Preprocessing Pipeline")
    print("=" * 50)

    # Find subjects
    subjects = find_subject_files()
    print(f"Found {len(subjects)} subjects")

    # Process each subject
    successful = 0
    failed = 0

    for subject_id, file_path in subjects.items():
        result = process_subject(subject_id, file_path)
        if result:
            successful += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"Total subjects: {len(subjects)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)


# In[ ]:


# Usage
if __name__ == "__main__":
    input_dir=Path("../../../SEED-DV/EEG")
    output_dir=Path("preprocessed_eeg_simple")
    run_pipeline()

