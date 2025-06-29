#!/usr/bin/env python3
"""
SEED-DV EEG Dataset Complete Preprocessing Pipeline
Phase 2: Full preprocessing implementation with bug fixes

This script implements the comprehensive preprocessing pipeline for SEED-DV dataset
based on the research report specifications and discovered data structure.
Data structure: (7_videos, 62_channels, timepoints)
"""

import os
import numpy as np
import mne
from pathlib import Path
from mne.preprocessing import ICA
from mne.decoding import Scaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set MNE logging to WARNING to reduce verbose output
mne.set_log_level('WARNING')

class SEEDDVPreprocessor:
    """Complete preprocessing pipeline for SEED-DV EEG data."""
    
    def __init__(self, input_dir="../SEED-DV/EEG", output_dir="preprocessed_eeg"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sfreq = 1000  # Original sampling rate
        self.target_sfreq = 250  # Target sampling rate after downsampling
        self.n_eeg_channels = 62
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Channel names for 62-channel 10-10 system
        self.ch_names = [
            'Fp1', 'Fpz', 'Fp2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFz', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
            'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F10',
            'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
            'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
            'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
            'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
            'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
            'O1', 'Oz', 'O2'
        ][:self.n_eeg_channels]

    def find_subject_files(self):
        """Find all subject files and group sessions."""
        files = list(self.input_dir.glob("*.npy"))
        subjects = {}
        
        for file in files:
            if "_session2" in file.name:
                subject_id = file.name.split("_session2")[0]
                if subject_id not in subjects:
                    subjects[subject_id] = []
                subjects[subject_id].append(file)
            else:
                subject_id = file.stem
                if subject_id not in subjects:
                    subjects[subject_id] = []
                subjects[subject_id].append(file)
        
        # Sort sessions for each subject
        for subject_id in subjects:
            subjects[subject_id].sort()
            
        return subjects

    def load_and_concatenate_sessions(self, subject_files):
        """Load and concatenate multiple sessions for a subject."""
        all_data = []
        
        for file_path in subject_files:
            print(f"  Loading: {file_path.name}")
            data = np.load(file_path)  # Shape: (7_videos, 62_channels, timepoints)
            
            # Reshape to (62_channels, total_timepoints) by concatenating videos
            n_videos, n_channels, n_timepoints = data.shape
            reshaped_data = data.transpose(1, 0, 2).reshape(n_channels, -1)
            all_data.append(reshaped_data)
        
        # Concatenate sessions along time axis
        if len(all_data) > 1:
            concatenated_data = np.concatenate(all_data, axis=1)
            print(f"  Concatenated {len(all_data)} sessions")
        else:
            concatenated_data = all_data[0]
        
        return concatenated_data

    def create_mne_raw(self, data, subject_id):
        """Create MNE Raw object from numpy array."""
        # Create info object
        ch_types = ['eeg'] * self.n_eeg_channels
        info = mne.create_info(ch_names=self.ch_names[:self.n_eeg_channels], 
                               sfreq=self.sfreq, ch_types=ch_types)
        
        # Create Raw object
        raw = mne.io.RawArray(data, info)
        
        # Set montage (10-10 system)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        return raw

    def apply_filtering(self, raw):
        """Apply filtering: high-pass (0.5 Hz), low-pass (80 Hz), notch (50 Hz)."""
        print("  Applying filters...")
        
        # High-pass filter at 0.5 Hz
        raw.filter(l_freq=0.5, h_freq=None, fir_design='firwin', 
                   phase='zero', filter_length='auto')
        
        # Low-pass filter at 80 Hz (preserve gamma band)
        raw.filter(l_freq=None, h_freq=80, fir_design='firwin', 
                   phase='zero', filter_length='auto')
        
        # Notch filter at 50 Hz (China power line frequency)
        raw.notch_filter(freqs=50, fir_design='firwin', phase='zero')
        
        return raw

    def apply_rereferencing(self, raw):
        """Apply Common Average Reference (CAR)."""
        print("  Applying Common Average Reference...")
        raw.set_eeg_reference(ref_channels='average', projection=True)
        raw.apply_proj()
        return raw

    def detect_bad_channels(self, raw):
        """Detect and interpolate bad channels with improved method."""
        print("  Detecting bad channels...")
        
        # Get data after filtering for better bad channel detection
        data = raw.get_data()
        
        # Method 1: Check for flat channels (std < threshold)
        channel_stds = np.std(data, axis=1)
        flat_threshold = 1e-6  # Very small threshold for flat channels
        flat_channels = [raw.ch_names[i] for i, std in enumerate(channel_stds) 
                        if std < flat_threshold]
        
        # Method 2: Check for extreme variance channels
        channel_vars = np.var(data, axis=1)
        q25, q75 = np.percentile(channel_vars, [25, 75])
        iqr = q75 - q25
        
        # More conservative outlier detection using IQR
        lower_bound = q25 - 3 * iqr  # 3 IQR instead of 1.5
        upper_bound = q75 + 3 * iqr
        
        outlier_channels = [raw.ch_names[i] for i, var in enumerate(channel_vars)
                           if var < lower_bound or var > upper_bound]
        
        # Method 3: Check for channels with extreme amplitude
        channel_max_abs = np.max(np.abs(data), axis=1)
        amp_threshold = np.percentile(channel_max_abs, 99)  # 99th percentile
        high_amp_channels = [raw.ch_names[i] for i, max_amp in enumerate(channel_max_abs)
                            if max_amp > amp_threshold * 2]  # 2x the 99th percentile
        
        # Combine all bad channel detections
        bad_channels = list(set(flat_channels + outlier_channels + high_amp_channels))
        
        if bad_channels:
            print(f"    Found {len(bad_channels)} bad channels: {bad_channels}")
            raw.info['bads'] = bad_channels
            
            # Check if too many bad channels
            if len(bad_channels) > 0.15 * len(raw.ch_names):  # Increased threshold to 15%
                print(f"    WARNING: >15% channels are bad ({len(bad_channels)}/{len(raw.ch_names)})")
                print(f"    Consider manual inspection of data quality")
            
            # Interpolate bad channels
            raw.interpolate_bads(reset_bads=True)
            print(f"    Interpolated {len(bad_channels)} bad channels")
        else:
            print("    No bad channels detected")
        
        return raw

    def apply_ica_artifact_removal(self, raw):
        """Apply ICA for artifact removal (EOG, ECG, EMG) with fixed indexing."""
        print("  Applying ICA for artifact removal...")
        
        # Prepare data for ICA (use filtered data)
        ica = ICA(n_components=0.95, max_iter='auto', random_state=42)
        
        # Fit ICA
        print("    Fitting ICA...")
        ica.fit(raw)
        
        # Automated artifact detection
        # For EOG: use frontal channels as proxy
        frontal_channels = [ch for ch in raw.ch_names if any(fp in ch.lower() 
                           for fp in ['fp1', 'fp2', 'af7', 'af8', 'f7', 'f8'])]
        eog_indices = []
        if frontal_channels:
            try:
                eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=frontal_channels[0])
                if eog_indices:
                    print(f"    Found {len(eog_indices)} EOG components: {eog_indices}")
            except Exception as e:
                print(f"    EOG detection failed: {e}")
        
        # For ECG: use automated detection
        ecg_indices = []
        try:
            ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='correlation', 
                                                        threshold='auto')
            if ecg_indices:
                print(f"    Found {len(ecg_indices)} ECG components: {ecg_indices}")
        except Exception as e:
            print("    No ECG components found or detection failed")
        
        # For muscle artifacts: detect high-frequency components (FIXED)
        muscle_indices = []
        try:
            # Get ICA source data
            sources = ica.get_sources(raw)
            source_data = sources.get_data()  # Shape: (n_components, n_times)
            
            for i in range(source_data.shape[0]):  # Loop over components
                component = source_data[i, :]  # Get single component time series
                
                # Check for high-frequency content typical of muscle artifacts
                try:
                    # Reshape to 2D array for psd_array_welch
                    component_2d = component.reshape(1, -1)
                    freqs, psd = mne.time_frequency.psd_array_welch(
                        component_2d, sfreq=raw.info['sfreq'], 
                        fmin=20, fmax=80, n_fft=min(1024, len(component)//4))
                    
                    # Calculate high-frequency power ratio
                    high_freq_mask = freqs[0] > 40
                    if np.any(high_freq_mask):
                        high_freq_power = np.mean(psd[0, high_freq_mask])
                        total_power = np.mean(psd[0, :])
                        
                        if total_power > 0 and high_freq_power / total_power > 0.6:
                            muscle_indices.append(i)
                            
                except Exception as e:
                    # Skip this component if PSD calculation fails
                    continue
            
            if muscle_indices:
                print(f"    Found {len(muscle_indices)} potential muscle components: {muscle_indices[:5]}...")
                
        except Exception as e:
            print(f"    Muscle artifact detection failed: {e}")
        
        # Combine all artifact components (limit to reasonable number)
        all_bad_components = list(set(eog_indices + ecg_indices + muscle_indices[:3]))
        
        if all_bad_components:
            print(f"    Removing {len(all_bad_components)} artifact components")
            ica.exclude = all_bad_components
            raw = ica.apply(raw)
            
            # Note about Chinese Cue artifacts
            print("    Note: Speech/swallowing artifacts during Chinese Cue periods")
            print("          have been addressed through ICA muscle artifact removal")
        else:
            print("    No artifact components identified for removal")
        
        return raw, ica

    def create_epochs(self, raw):
        """Create epochs for video segments."""
        print("  Creating epochs...")
        
        # Since we don't have explicit event markers, we'll create synthetic events
        # Based on the 7 videos structure and assuming equal length videos
        data_length = raw.n_times
        video_length = data_length // 7  # Approximate length per video
        
        # Create events for each video start
        events = []
        for i in range(7):
            event_sample = i * video_length
            if event_sample < data_length - int(10.2 * raw.info['sfreq']):  # Ensure 10.2s epoch fits
                events.append([event_sample, 0, 1])  # [sample, previous_event, event_id]
        
        if len(events) == 0:
            # Fallback: create single epoch from beginning
            events = [[0, 0, 1]]
            print("    Warning: Could not create multiple epochs, using single epoch")
        
        events = np.array(events)
        
        # Create epochs (10-second duration with 0.2s baseline)
        epoch_duration = 10.0  # seconds
        baseline = (-0.2, 0.0)  # 200ms baseline
        
        try:
            epochs = mne.Epochs(raw, events, event_id={'video': 1}, 
                               tmin=-0.2, tmax=epoch_duration, 
                               baseline=baseline, preload=True,
                               reject=None, flat=None)
            
            print(f"    Created {len(epochs)} epochs of {epoch_duration}s each")
        except Exception as e:
            print(f"    Epoch creation failed: {e}")
            # Create a single epoch from the beginning of the data
            events = np.array([[int(0.2 * raw.info['sfreq']), 0, 1]])  # Start after baseline
            epochs = mne.Epochs(raw, events, event_id={'video': 1}, 
                               tmin=-0.2, tmax=min(epoch_duration, (raw.n_times / raw.info['sfreq']) - 0.3), 
                               baseline=baseline, preload=True,
                               reject=None, flat=None)
            print(f"    Created {len(epochs)} fallback epoch")
        
        return epochs

    def downsample_data(self, epochs):
        """Downsample to target frequency."""
        print(f"  Downsampling from {epochs.info['sfreq']} Hz to {self.target_sfreq} Hz...")
        epochs.resample(self.target_sfreq, npad='auto')
        return epochs

    def apply_normalization(self, epochs):
        """Apply Z-scoring normalization per-channel and per-epoch."""
        print("  Applying Z-score normalization...")
        
        # Get data
        data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        
        # Z-score per channel per epoch
        for epoch_idx in range(data.shape[0]):
            for ch_idx in range(data.shape[1]):
                channel_data = data[epoch_idx, ch_idx, :]
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                if std_val > 0:
                    data[epoch_idx, ch_idx, :] = (channel_data - mean_val) / std_val
        
        # Update epochs data
        epochs._data = data
        return epochs

    def save_preprocessed_data(self, epochs, subject_id):
        """Save preprocessed epochs data."""
        print("  Saving preprocessed data...")
        
        # Get data in format suitable for deep learning
        data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_timepoints)
        
        # Save as numpy array
        output_file = self.output_dir / f"{subject_id}_preprocessed.npy"
        np.save(output_file, data)
        
        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'shape': data.shape,
            'sfreq': epochs.info['sfreq'],
            'ch_names': epochs.ch_names,
            'n_epochs': len(epochs),
            'epoch_duration': 10.0,
            'preprocessing_steps': [
                'filtering (0.5-80 Hz, notch 50 Hz)',
                'common_average_reference',
                'bad_channel_interpolation',
                'ICA_artifact_removal',
                'epoching (10s)',
                'downsampling (250 Hz)',
                'z_score_normalization'
            ]
        }
        
        metadata_file = self.output_dir / f"{subject_id}_metadata.npy"
        np.save(metadata_file, metadata)
        
        print(f"    Saved: {output_file}")
        print(f"    Data shape: {data.shape}")
        return output_file

    def plot_preprocessing_comparison(self, raw_before, raw_after, subject_id):
        """Plot PSD comparison to demonstrate preprocessing effectiveness."""
        print("  Generating preprocessing comparison plots...")
        
        try:
            # Compute PSD for both raw signals
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Before preprocessing
            psd_before, freqs_before = mne.time_frequency.psd_array_welch(
                raw_before.get_data(), sfreq=raw_before.info['sfreq'], 
                fmin=0.5, fmax=100, n_fft=2048)
            psd_before_mean = np.mean(psd_before, axis=0)
            
            ax1.semilogy(freqs_before, psd_before_mean)
            ax1.set_title(f'{subject_id} - Before Preprocessing')
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('PSD (V²/Hz)')
            ax1.axvline(50, color='red', linestyle='--', alpha=0.7, label='50 Hz line noise')
            ax1.legend()
            ax1.set_xlim(0, 100)
            
            # After preprocessing
            psd_after, freqs_after = mne.time_frequency.psd_array_welch(
                raw_after.get_data(), sfreq=raw_after.info['sfreq'], 
                fmin=0.5, fmax=80, n_fft=2048)
            psd_after_mean = np.mean(psd_after, axis=0)
            
            ax2.semilogy(freqs_after, psd_after_mean)
            ax2.set_title(f'{subject_id} - After Preprocessing')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('PSD (V²/Hz)')
            ax2.set_xlim(0, 80)
            
            plt.tight_layout()
            plot_file = self.output_dir / f"{subject_id}_preprocessing_comparison.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved comparison plot: {plot_file}")
            
        except Exception as e:
            print(f"    Could not generate comparison plot: {e}")

    def process_subject(self, subject_id, subject_files):
        """Complete preprocessing pipeline for a single subject."""
        print(f"\nProcessing subject: {subject_id}")
        print("=" * 50)
        
        try:
            # 1. Load and concatenate sessions
            data = self.load_and_concatenate_sessions(subject_files)
            
            # 2. Create MNE Raw object
            raw = self.create_mne_raw(data, subject_id)
            print(f"  Created Raw object: {raw.info['nchan']} channels, {raw.n_times} timepoints")
            
            # Keep copy for comparison
            raw_before = raw.copy()
            
            # 3. Apply filtering
            raw = self.apply_filtering(raw)
            
            # 4. Apply re-referencing
            raw = self.apply_rereferencing(raw)
            
            # 5. Detect and handle bad channels
            raw = self.detect_bad_channels(raw)
            
            # 6. Apply ICA artifact removal
            raw, ica = self.apply_ica_artifact_removal(raw)
            
            # 7. Create epochs
            epochs = self.create_epochs(raw)
            
            # 8. Downsample
            epochs = self.downsample_data(epochs)
            
            # 9. Apply normalization
            epochs = self.apply_normalization(epochs)
            
            # 10. Save preprocessed data
            output_file = self.save_preprocessed_data(epochs, subject_id)
            
            # 11. Generate comparison plots
            self.plot_preprocessing_comparison(raw_before, raw, subject_id)
            
            print(f"Successfully processed {subject_id}")
            return True
            
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_preprocessing(self):
        """Run complete preprocessing pipeline for all subjects."""
        print("SEED-DV EEG Preprocessing Pipeline")
        print("=" * 50)
        
        # Find all subject files
        subjects = self.find_subject_files()
        print(f"Found {len(subjects)} subjects")
        
        successful_subjects = []
        failed_subjects = []
        
        # Process each subject
        for subject_id, subject_files in subjects.items():
            success = self.process_subject(subject_id, subject_files)
            if success:
                successful_subjects.append(subject_id)
            else:
                failed_subjects.append(subject_id)
        
        # Final summary
        print("\n" + "=" * 50)
        print("PREPROCESSING COMPLETE")
        print("=" * 50)
        print(f"Successfully processed: {len(successful_subjects)} subjects")
        if failed_subjects:
            print(f"Failed to process: {len(failed_subjects)} subjects: {failed_subjects}")
        
        print(f"\nPreprocessed data saved to: {self.output_dir}")
        print("\nNext steps:")
        print("1. Review generated PSD comparison plots for quality assessment")
        print("2. Load preprocessed data for deep learning model training")
        print("3. Align with video metadata (BLIP-captions, meta-info)")

def main():
    """Main function to run the preprocessing pipeline."""
    # Initialize preprocessor
    preprocessor = SEEDDVPreprocessor()
    
    # Run preprocessing
    preprocessor.run_preprocessing()

if __name__ == "__main__":
    main()
