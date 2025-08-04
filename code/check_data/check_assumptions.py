import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# --- CONFIGURATION ---
directory_path = '../../SEED-DV/EEG'  # <-- Update this to your actual path
sfreq = 1000  # Hz, assumed sampling frequency; change if needed

# --- Locate Session Files ---
session_files = sorted(glob.glob(os.path.join(directory_path, 'sub1_session*.npy')))
if not session_files:
    raise FileNotFoundError("❌ No .npy session files found. Please check the directory path.")

# --- Inspect EEG File Structure ---
def inspect_session(filepath, sfreq):
    try:
        eeg_data = np.load(filepath)
        print(f"\n--- Inspecting File: {os.path.basename(filepath)} ---")
        print(f"Data Shape: {eeg_data.shape}")

        if eeg_data.ndim == 2:
            num_channels, num_samples = eeg_data.shape
            duration_seconds = num_samples / sfreq
            print(f"→ Channels: {num_channels}")
            print(f"→ Samples: {num_samples}")
            print(f"→ Duration: {duration_seconds:.2f} seconds")

        elif eeg_data.ndim == 3:
            num_trials, num_channels, num_samples = eeg_data.shape
            duration_seconds = num_samples / sfreq
            print(f"→ Trials: {num_trials}")
            print(f"→ Channels: {num_channels}")
            print(f"→ Samples per Trial: {num_samples}")
            print(f"→ Duration per Trial: {duration_seconds:.2f} seconds")

        else:
            print(f"[ERROR] Unexpected data shape: {eeg_data.shape}")

        print("-" * 50)

    except Exception as e:
        print(f"[ERROR] Failed to inspect {filepath}: {e}")

# --- Analyze Signal Scale to Infer Units ---
def analyze_signal_scale(filepath, sfreq):
    try:
        eeg_data = np.load(filepath)

        if eeg_data.ndim == 2:
            sample_data = eeg_data[0, :]  # Channel 0
        elif eeg_data.ndim == 3:
            sample_data = eeg_data[0, 0, :]  # Trial 0, Channel 0
        else:
            raise ValueError("Unsupported data shape for signal analysis.")

        mean_val = np.mean(sample_data)
        std_val = np.std(sample_data)
        min_val = np.min(sample_data)
        max_val = np.max(sample_data)
        peak_amplitude = max(abs(min_val), abs(max_val))

        print(f"\n--- Signal Scale Analysis (Channel 0 of {os.path.basename(filepath)}) ---")
        print(f"Mean: {mean_val:.3e}")
        print(f"Std Dev: {std_val:.3e}")
        print(f"Min: {min_val:.3e}")
        print(f"Max: {max_val:.3e}")
        print(f"Peak Amplitude: {peak_amplitude:.3e}")

        print("\n--- Interpretation ---")
        if peak_amplitude < 0.01:
            print("🔍 Likely Units: VOLTS (V)")
            print("⚠️ Suggest using rejection threshold ~100e-6 to 150e-6 V (100–150 µV equivalent).")
        else:
            print("🔍 Likely Units: MICROVOLTS (µV)")
            print("⚠️ Suggest using rejection threshold ~100 to 150 µV.")

        # Save plot instead of showing
        plt.figure(figsize=(12, 4))
        plt.plot(sample_data[:sfreq], color='royalblue')
        plt.title("First Second of Channel 0")
        plt.xlabel("Samples (1 sec)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()

        plot_file = f"{os.path.basename(filepath).replace('.npy', '')}_channel0_preview.png"
        plt.savefig(plot_file)
        print(f"📊 Plot saved as: {plot_file}")

    except Exception as e:
        print(f"[ERROR] Failed to analyze signal scale: {e}")

# --- Run Analysis ---
for session_file in session_files:
    inspect_session(session_file, sfreq)

# --- Run Signal Scale Analysis on First File ---
analyze_signal_scale(session_files[0], sfreq)

# --- Unverifiable Assumptions ---
print("\n--- List of Unverifiable Assumptions Due to Missing Event Markers ---")
print("1. ⛔ **Unknown Onset Timing:** Cannot confirm the start time of the first cue.")
print("2. ⛔ **Missing Event Boundaries:** No info on when specific stimuli (video/cue) transitions occur.")
print("3. ⛔ **Block Timing:** Durations of inter-trial or inter-block intervals are not available.")
print("4. ⛔ **Synchronization:** EEG may not be time-locked to stimulus presentation without trigger info.")
print("5. ⛔ **Stimulus Labeling:** No way to match specific EEG segments to labeled events (e.g., emotions).")
