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

# Set MNE logging to reduce output
mne.set_log_level('WARNING')

class SimpleEEGPreprocessor:
    """Simple EEG preprocessing pipeline for SEED-DV dataset."""
    
    def __init__(self, input_dir="../SEED-DV/EEG", output_dir="preprocessed_eeg"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Basic parameters
        self.sfreq = 200  # Original sampling rate
        self.target_sfreq = 200  # Target sampling rate
        self.n_channels = 62
        
        # Standard 62-channel names
        self.ch_names = [
            'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
            'CB1', 'O1', 'OZ', 'O2', 'CB2'
        ]

    def find_subject_files(self):
        """Find all subject .npy files."""
        files = list(self.input_dir.glob("*.npy"))
        subjects = {}
        
        for file in files:
            subject_id = file.stem
            subjects[subject_id] = file
        
        return subjects

    def load_data(self, file_path):
        """Load and reshape SEED-DV data."""
        print(f"  Loading: {file_path.name}")
        data = np.load(file_path)  # Shape: (7_videos, 62_channels, timepoints)
        
        # Reshape to (channels, timepoints)
        n_videos, n_channels, n_timepoints = data.shape
        reshaped_data = data.transpose(1, 0, 2).reshape(n_channels, -1)
        
        print(f"  Original shape: {data.shape}")
        print(f"  Reshaped to: {reshaped_data.shape}")
        
        return reshaped_data

    def create_raw_object(self, data):
        """Create MNE Raw object."""
        # Create info
        info = mne.create_info(
            ch_names=self.ch_names[:self.n_channels],
            sfreq=self.sfreq,
            ch_types=['eeg'] * self.n_channels
        )
        
        # Create Raw object
        raw = mne.io.RawArray(data, info)
        
        # Set standard montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        return raw

    def apply_filters(self, raw):
        """Apply standard filtering."""
        print("  Applying filters...")
        
        # Bandpass filter: 0.5-80 Hz
        raw.filter(l_freq=0.5, h_freq=80, fir_design='firwin')
        
        # Notch filter: 50 Hz (power line)
        raw.notch_filter(freqs=50, fir_design='firwin')
        
        return raw

    def apply_reference(self, raw):
        """Apply common average reference."""
        print("  Applying common average reference...")
        raw.set_eeg_reference(ref_channels='average', projection=True)
        raw.apply_proj()
        return raw

    def detect_bad_channels(self, raw):
        """Simple bad channel detection."""
        print("  Detecting bad channels...")
        data = raw.get_data()
        
        # Find channels with extreme values
        channel_stds = np.std(data, axis=1)
        mean_std = np.mean(channel_stds)
        std_threshold = 3 * np.std(channel_stds)
        
        bad_channels = []
        for i, (ch_name, ch_std) in enumerate(zip(raw.ch_names, channel_stds)):
            if ch_std > mean_std + std_threshold or ch_std < mean_std - std_threshold:
                bad_channels.append(ch_name)
        
        if bad_channels:
            print(f"    Found {len(bad_channels)} bad channels: {bad_channels}")
            raw.info['bads'] = bad_channels
            raw.interpolate_bads(reset_bads=True)
        else:
            print("    No bad channels detected")
        
        return raw

    def apply_ica(self, raw):
        """Apply ICA for artifact removal."""
        print("  Applying ICA...")
        
        # Fit ICA
        ica = ICA(n_components=20, random_state=42, method='fastica')
        ica.fit(raw)
        
        # Find and remove EOG artifacts
        try:
            eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=2.0)
            if eog_indices:
                print(f"    Found {len(eog_indices)} EOG components")
                ica.exclude = eog_indices
        except:
            print("    No EOG components found")
        
        # Apply ICA
        raw = ica.apply(raw)
        
        return raw

    def create_epochs(self, raw):
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


    def downsample(self, epochs):
        """Downsample to target frequency."""
        print(f"  Downsampling to {self.target_sfreq} Hz...")
        epochs.resample(self.target_sfreq)
        return epochs

    def normalize_data(self, epochs):
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

    def save_data(self, epochs, subject_id):
        """Save preprocessed data."""
        print("  Saving data...")
        
        data = epochs.get_data()
        
        # Save numpy array
        output_file = self.output_dir / f"{subject_id}_preprocessed.npy"
        np.save(output_file, data)
        
        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'shape': data.shape,
            'sfreq': epochs.info['sfreq'],
            'ch_names': epochs.ch_names,
            'n_epochs': len(epochs),
            'preprocessing_steps': [
                'bandpass_filter_0.5_80Hz',
                'notch_filter_50Hz',
                'common_average_reference',
                'bad_channel_detection',
                'ICA_artifact_removal',
                'epoching_10s',
                'downsampling_250Hz',
                'z_score_normalization'
            ]
        }
        
        metadata_file = self.output_dir / f"{subject_id}_metadata.npy"
        np.save(metadata_file, metadata)
        
        print(f"    Saved: {output_file}")
        print(f"    Shape: {data.shape}")
        
        return output_file

    def plot_comparison(self, raw_before, raw_after, subject_id):
        """Plot before/after comparison."""
        print("  Creating comparison plot...")
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot PSD comparison
            psd_before, freqs_before = mne.time_frequency.psd_array_welch(
                raw_before.get_data(), sfreq=raw_before.info['sfreq'],
                fmin=0.5, fmax=80)
            
            psd_after, freqs_after = mne.time_frequency.psd_array_welch(
                raw_after.get_data(), sfreq=raw_after.info['sfreq'],
                fmin=0.5, fmax=80)
            
            axes[0].semilogy(freqs_before, np.mean(psd_before, axis=0), 
                            'r-', label='Before', alpha=0.7)
            axes[0].semilogy(freqs_after, np.mean(psd_after, axis=0), 
                            'b-', label='After', alpha=0.7)
            axes[0].set_xlabel('Frequency (Hz)')
            axes[0].set_ylabel('Power (V²/Hz)')
            axes[0].set_title('Power Spectral Density')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot time series comparison (first 10 seconds)
            duration = 10
            n_samples = int(duration * raw_before.info['sfreq'])
            time_vec = np.arange(n_samples) / raw_before.info['sfreq']
            
            # Use Cz channel if available
            ch_idx = raw_before.ch_names.index('CZ') if 'CZ' in raw_before.ch_names else 0
            
            axes[1].plot(time_vec, raw_before.get_data()[ch_idx, :n_samples] * 1e6,
                        'r-', label='Before', alpha=0.7)
            axes[1].plot(time_vec, raw_after.get_data()[ch_idx, :n_samples] * 1e6,
                        'b-', label='After', alpha=0.7)
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Amplitude (µV)')
            axes[1].set_title(f'Time Series - {raw_before.ch_names[ch_idx]}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"{subject_id}_comparison.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved plot: {plot_file}")
            
        except Exception as e:
            print(f"    Error creating plot: {e}")

    def process_subject(self, subject_id, file_path):
        """Process a single subject."""
        print(f"\nProcessing subject: {subject_id}")
        
        try:
            # Load data
            data = self.load_data(file_path)
            
            # Create Raw object
            raw = self.create_raw_object(data)
            raw_original = raw.copy()
            
            # Apply preprocessing steps
            raw = self.apply_filters(raw)
            raw = self.apply_reference(raw)
            raw = self.detect_bad_channels(raw)
            raw = self.apply_ica(raw)
            
            # Create epochs
            epochs = self.create_epochs(raw)
            
            if len(epochs) == 0:
                print(f"  No valid epochs for {subject_id}")
                return None
            
            # Downsample and normalize
            epochs = self.downsample(epochs)
            epochs = self.normalize_data(epochs)
            
            # Save data
            output_file = self.save_data(epochs, subject_id)
            
            # Create comparison plot
            self.plot_comparison(raw_original, raw, subject_id)
            
            print(f"✓ Successfully processed {subject_id}")
            print(f"  Final shape: {epochs.get_data().shape}")
            
            return output_file
            
        except Exception as e:
            print(f"✗ Error processing {subject_id}: {e}")
            return None

    def run_pipeline(self):
        """Run the complete preprocessing pipeline."""
        print("Simple SEED-DV EEG Preprocessing Pipeline")
        print("=" * 50)
        
        # Find subjects
        subjects = self.find_subject_files()
        print(f"Found {len(subjects)} subjects")
        
        # Process each subject
        successful = 0
        failed = 0
        
        for subject_id, file_path in subjects.items():
            result = self.process_subject(subject_id, file_path)
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
        print(f"Output directory: {self.output_dir}")
        print("=" * 50)

# Usage
if __name__ == "__main__":
    # Initialize and run
    preprocessor = SimpleEEGPreprocessor(
        input_dir="../SEED-DV/EEG",
        output_dir="preprocessed_eeg_simple"
    )
    
    preprocessor.run_pipeline()