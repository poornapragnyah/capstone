#!/usr/bin/env python3
"""
SEED-DV EEG Dataset Enhanced Preprocessing Pipeline
Improved version with better epoching, ASR, and artifact rejection

This script implements the comprehensive preprocessing pipeline for SEED-DV dataset
with enhanced artifact removal and proper epoch extraction.
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
    """Enhanced preprocessing pipeline for SEED-DV EEG data."""
    
    def __init__(self, input_dir="../SEED-DV/EEG", output_dir="preprocessed_eeg"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sfreq = 200  # CORRECTED: Original sampling rate is 200 Hz (already downsampled)
        self.target_sfreq = 200  # CORRECTED: Keep at 200 Hz (no further downsampling needed)
        self.n_eeg_channels = 62
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # SEED-DV specific parameters
        self.n_videos = 7
        self.concepts_per_video = 40
        self.concept_duration = 10.0  # seconds per concept
        self.cue_duration = 3.0  # Chinese cue duration
        self.block_duration = 520.0  # seconds per block (40 × 13s = 520s)
        
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
        """UPDATED: Apply only notch filter since data is already band-pass filtered."""
        print("  Applying notch filter (data already band-pass filtered 0.1-100 Hz)...")
        
        # Only apply notch filter at 50 Hz (China power line frequency)
        # Data is already band-pass filtered between 0.1-100 Hz
        raw.notch_filter(freqs=50, fir_design='firwin', phase='zero')
        
        return raw
    
    def apply_rereferencing(self, raw):
        """Apply Common Average Reference (CAR)."""
        print("  Applying Common Average Reference...")
        raw.set_eeg_reference(ref_channels='average', projection=True)
        raw.apply_proj()
        return raw

    def detect_bad_channels(self, raw):
        """Fixed bad channel detection with more conservative thresholds."""
        print("  Detecting bad channels...")
        
        # Get data after filtering for better bad channel detection
        data = raw.get_data()
        
        bad_channels = []
        
        # Method 1: Check for flat channels (std < threshold)
        channel_stds = np.std(data, axis=1)
        flat_threshold = 1e-7  # More conservative threshold
        flat_channels = [raw.ch_names[i] for i, std in enumerate(channel_stds) 
                        if std < flat_threshold]
        bad_channels.extend(flat_channels)
        
        # Method 2: Check for extreme variance channels (more conservative)
        channel_vars = np.var(data, axis=1)
        q25, q75 = np.percentile(channel_vars, [25, 75])
        iqr = q75 - q25
        
        # Much more conservative outlier detection
        lower_bound = q25 - 5 * iqr  # Changed from 2.5 to 5
        upper_bound = q75 + 5 * iqr  # Changed from 2.5 to 5
        
        outlier_channels = [raw.ch_names[i] for i, var in enumerate(channel_vars)
                        if var < lower_bound or var > upper_bound]
        bad_channels.extend(outlier_channels)
        
        # Method 3: Check for channels with extreme amplitude (more conservative)
        channel_max_abs = np.max(np.abs(data), axis=1)
        amp_q99 = np.percentile(channel_max_abs, 99)  # Changed from 95 to 99
        high_amp_channels = [raw.ch_names[i] for i, max_amp in enumerate(channel_max_abs)
                            if max_amp > amp_q99 * 5]  # Changed from 3x to 5x
        bad_channels.extend(high_amp_channels)
        
        # Method 4: Correlation-based bad channel detection (more conservative)
        corr_matrix = np.corrcoef(data)
        # Remove diagonal and compute mean correlation for each channel
        np.fill_diagonal(corr_matrix, np.nan)
        mean_corr = np.nanmean(corr_matrix, axis=1)
        
        # Only mark channels with extremely low correlation
        low_corr_threshold = np.percentile(mean_corr, 2)  # Changed from 10% to 2%
        low_corr_channels = [raw.ch_names[i] for i, corr in enumerate(mean_corr)
                            if corr < low_corr_threshold and corr < 0.1]  # Additional threshold
        bad_channels.extend(low_corr_channels)
        
        # Remove duplicates
        bad_channels = list(set(bad_channels))
        
        # Limit the number of bad channels to prevent over-removal
        max_bad_channels = min(6, int(0.10 * len(raw.ch_names)))  # Max 6 channels or 10%
        if len(bad_channels) > max_bad_channels:
            print(f"    Limiting bad channels from {len(bad_channels)} to {max_bad_channels}")
            # Keep only the most problematic ones based on variance
            channel_problem_scores = []
            for ch_name in bad_channels:
                ch_idx = raw.ch_names.index(ch_name)
                var_score = abs(channel_vars[ch_idx] - np.median(channel_vars)) / np.std(channel_vars)
                corr_score = abs(mean_corr[ch_idx] - np.median(mean_corr)) / np.std(mean_corr)
                total_score = var_score + corr_score
                channel_problem_scores.append((ch_name, total_score))
            
            # Sort by problem score and keep worst ones
            channel_problem_scores.sort(key=lambda x: x[1], reverse=True)
            bad_channels = [ch for ch, score in channel_problem_scores[:max_bad_channels]]
        
        if bad_channels:
            print(f"    Found {len(bad_channels)} bad channels: {bad_channels}")
            raw.info['bads'] = bad_channels
            
            # Interpolate bad channels
            raw.interpolate_bads(reset_bads=True)
            print(f"    Interpolated {len(bad_channels)} bad channels")
        else:
            print("    No bad channels detected")
        
        return raw

    def apply_asr(self, raw):
        """Simplified ASR with less aggressive artifact marking."""
        print("  Applying Artifact Subspace Reconstruction (ASR)...")
        
        try:
            # Import ASR from mne-icalabel or implement basic version
            from mne.preprocessing import annotate_muscle_zscore
            
            # More conservative muscle artifact detection
            threshold_muscle = 6  # Increased from 4 to 6 (less sensitive)
            
            # Annotate muscle artifacts
            muscle_annot, muscle_scores = annotate_muscle_zscore(
                raw, ch_type='eeg', threshold=threshold_muscle, 
                min_length_good=0.5, filter_freq=[80,90])  # Increased min_length_good
            
            if len(muscle_annot) > 0:
                # Only keep the most severe artifacts (top 50%)
                if len(muscle_annot) > 10:  # If too many artifacts detected
                    # Sort by duration and keep only longer ones
                    durations = muscle_annot.duration
                    duration_threshold = np.percentile(durations, 70)  # Keep top 30%
                    keep_mask = durations >= duration_threshold
                    
                    # Create new annotations with filtered artifacts
                    filtered_annot = mne.Annotations(
                        onset=muscle_annot.onset[keep_mask],
                        duration=muscle_annot.duration[keep_mask],
                        description=np.array(muscle_annot.description)[keep_mask]
                    )
                    raw.set_annotations(filtered_annot)
                    print(f"    Marked {len(filtered_annot)} severe muscle artifact segments (filtered from {len(muscle_annot)})")
                else:
                    raw.set_annotations(muscle_annot)
                    print(f"    Marked {len(muscle_annot)} muscle artifact segments")
            else:
                print("    No muscle artifacts detected by ASR")
                    
        except ImportError:
            print("    ASR not available, using alternative muscle artifact detection")
            # Simplified alternative that's less aggressive
            data = raw.get_data()
            
            # Only process a few representative channels to avoid over-detection
            representative_channels = [0, len(data)//2, len(data)-1]  # First, middle, last
            
            for ch_idx in representative_channels:
                ch_data = data[ch_idx, :]
                
                # High-pass filter for muscle artifacts (>40 Hz instead of >30 Hz)
                from scipy import signal
                sos = signal.butter(4, 40, btype='high', fs=raw.info['sfreq'], output='sos')
                high_freq_data = signal.sosfilt(sos, ch_data)
                
                # Find high-amplitude segments with more conservative threshold
                envelope = np.abs(signal.hilbert(high_freq_data))
                threshold = np.percentile(envelope, 99)  # Changed from 95 to 99
                
                artifact_mask = envelope > threshold * 3  # Increased multiplier
                
                # Create annotations for artifact segments
                if np.any(artifact_mask):
                    artifact_starts = np.where(np.diff(artifact_mask.astype(int)) == 1)[0]
                    artifact_ends = np.where(np.diff(artifact_mask.astype(int)) == -1)[0]
                    
                    if len(artifact_starts) > 0 and len(artifact_ends) > 0:
                        # Ensure matching starts and ends
                        if len(artifact_starts) > 0 and len(artifact_ends) > 0:
                            if artifact_starts[0] > artifact_ends[0]:
                                artifact_ends = artifact_ends[1:]
                            if len(artifact_starts) > len(artifact_ends):
                                artifact_starts = artifact_starts[:-1]
                                
                            if len(artifact_starts) > 0:
                                onset_times = artifact_starts / raw.info['sfreq']
                                durations = (artifact_ends - artifact_starts) / raw.info['sfreq']
                                
                                # Only keep longer artifacts (>0.1 seconds)
                                long_artifacts = durations > 0.1
                                if np.any(long_artifacts):
                                    annotations = mne.Annotations(
                                        onset=onset_times[long_artifacts], 
                                        duration=durations[long_artifacts], 
                                        description=['muscle_artifact'] * np.sum(long_artifacts))
                                    raw.set_annotations(annotations)
                                    print(f"    Marked {np.sum(long_artifacts)} muscle artifacts")
                                break  # Only annotate once for efficiency
        
        return raw

    def apply_ica_artifact_removal(self, raw):
        """Enhanced ICA for artifact removal with better component selection."""
        print("  Applying ICA for artifact removal...")
        
        # Prepare data for ICA (use filtered data)
        ica = ICA(n_components=0.95, max_iter='auto', random_state=42, method='fastica')
        
        # Fit ICA
        print("    Fitting ICA...")
        ica.fit(raw)
        
        # Enhanced EOG detection
        frontal_channels = [ch for ch in raw.ch_names if any(fp in ch.lower() 
                           for fp in ['fp1', 'fp2', 'fpz', 'af7', 'af8', 'f7', 'f8'])]
        eog_indices = []
        if frontal_channels:
            try:
                # Try multiple frontal channels for better EOG detection
                for ch_name in frontal_channels[:3]:  # Try first 3 frontal channels
                    try:
                        eog_idx, eog_scores = ica.find_bads_eog(raw, ch_name=ch_name, threshold=2.5)
                        eog_indices.extend(eog_idx)
                    except:
                        continue
                
                eog_indices = list(set(eog_indices))  # Remove duplicates
                if eog_indices:
                    print(f"    Found {len(eog_indices)} EOG components: {eog_indices}")
            except Exception as e:
                print(f"    EOG detection failed: {e}")
        
        # Enhanced ECG detection
        ecg_indices = []
        try:
            ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='correlation', 
                                                        threshold=0.8)  # More stringent
            if ecg_indices:
                print(f"    Found {len(ecg_indices)} ECG components: {ecg_indices}")
        except Exception as e:
            print("    No ECG components found or detection failed")
        
        # Enhanced muscle artifact detection
        muscle_indices = []
        try:
            sources = ica.get_sources(raw)
            source_data = sources.get_data()
            
            for i in range(min(source_data.shape[0], 20)):  # Check first 20 components
                component = source_data[i, :]
                
                try:
                    # More robust PSD calculation
                    component_2d = component.reshape(1, -1)
                    freqs, psd = mne.time_frequency.psd_array_welch(
                        component_2d, sfreq=raw.info['sfreq'], 
                        fmin=1, fmax=80, n_fft=min(2048, len(component)//4))
                    
                    # Calculate multiple frequency band ratios
                    delta_mask = (freqs[0] >= 1) & (freqs[0] <= 4)
                    theta_mask = (freqs[0] >= 4) & (freqs[0] <= 8)
                    alpha_mask = (freqs[0] >= 8) & (freqs[0] <= 13)
                    beta_mask = (freqs[0] >= 13) & (freqs[0] <= 30)
                    gamma_mask = (freqs[0] >= 30) & (freqs[0] <= 80)
                    
                    if np.any(gamma_mask) and np.any(delta_mask):
                        gamma_power = np.mean(psd[0, gamma_mask])
                        delta_power = np.mean(psd[0, delta_mask])
                        total_power = np.mean(psd[0, :])
                        
                        # Multiple criteria for muscle artifacts
                        gamma_ratio = gamma_power / total_power if total_power > 0 else 0
                        gamma_delta_ratio = gamma_power / delta_power if delta_power > 0 else 0
                        
                        # Check for high-frequency dominance
                        high_freq_mask = freqs[0] > 40
                        if np.any(high_freq_mask):
                            high_freq_power = np.mean(psd[0, high_freq_mask])
                            high_freq_ratio = high_freq_power / total_power if total_power > 0 else 0
                            
                            # More stringent criteria
                            if (gamma_ratio > 0.4 or 
                                gamma_delta_ratio > 10 or 
                                high_freq_ratio > 0.3):
                                muscle_indices.append(i)
                                
                except Exception as e:
                    continue
            
            # Limit muscle components to avoid over-removal
            muscle_indices = muscle_indices[:5]  # Max 5 muscle components
            
            if muscle_indices:
                print(f"    Found {len(muscle_indices)} potential muscle components: {muscle_indices}")
                
        except Exception as e:
            print(f"    Muscle artifact detection failed: {e}")
        
        # Combine all artifact components with safety limits
        all_bad_components = list(set(eog_indices + ecg_indices + muscle_indices))
        
        # Safety check: don't remove too many components
        max_components_to_remove = min(10, int(0.3 * ica.n_components_))
        if len(all_bad_components) > max_components_to_remove:
            print(f"    Limiting component removal from {len(all_bad_components)} to {max_components_to_remove}")
            all_bad_components = all_bad_components[:max_components_to_remove]
        
        if all_bad_components:
            print(f"    Removing {len(all_bad_components)} artifact components")
            ica.exclude = all_bad_components
            raw = ica.apply(raw.copy())  # Apply to copy to preserve original
        else:
            print("    No artifact components identified for removal")
        
        return raw, ica

    def create_epochs_enhanced(self, raw, subject_files):
        """CORRECTED: Epoch creation with proper 200 Hz sampling rate calculations."""
        print("  Creating enhanced epochs with corrected timing...")
        
        # Load original data to understand structure
        original_data = []
        for file_path in subject_files:
            data = np.load(file_path)  # Shape: (7_videos, 62_channels, timepoints)
            original_data.append(data)
        
        if len(original_data) > 1:
            # Concatenate sessions (only sub1 has multiple sessions)
            full_data = np.concatenate(original_data, axis=0)  # (total_videos, 62, timepoints)
        else:
            full_data = original_data[0]
        
        n_total_videos = full_data.shape[0]  # 7 for single session, 14 for sub1
        timepoints_per_video = full_data.shape[2]  # Should be 104000 for 520s at 200Hz
        
        # Verify timing calculations
        expected_duration_per_block = timepoints_per_video / self.sfreq
        print(f"    Expected duration per block: {expected_duration_per_block:.1f}s")
        print(f"    Total blocks: {n_total_videos}")
        
        # Create events based on SEED-DV protocol
        events = []
        epoch_id = 1
        current_sample = 0
        
        for block_idx in range(n_total_videos):
            print(f"    Processing block {block_idx + 1}/{n_total_videos}...")
            
            block_start_sample = current_sample
            block_duration_samples = int(self.block_duration * self.sfreq)  # 520s * 200Hz = 104000 samples
            
            # Each block contains 40 concept presentations
            for concept_idx in range(self.concepts_per_video):
                # Time within the block for this concept
                concept_time_in_block = concept_idx * (self.cue_duration + self.concept_duration)  # concept_idx * 13s
                
                # Add cue duration to get to video start
                video_start_time_in_block = concept_time_in_block + self.cue_duration  # +3s for cue
                
                # Convert to absolute sample index
                video_start_sample = block_start_sample + int(video_start_time_in_block * self.sfreq)
                
                # Ensure we don't exceed the available data
                if video_start_sample + int((self.concept_duration + 0.2) * self.sfreq) <= raw.n_times:
                    events.append([video_start_sample, 0, epoch_id])
                    epoch_id += 1
                else:
                    print(f"      Warning: Concept {concept_idx + 1} in block {block_idx + 1} exceeds data length")
                    break
            
            # Move to next block
            current_sample += block_duration_samples
        
        if len(events) == 0:
            print("    ERROR: No valid events created!")
            return None
        
        events = np.array(events)
        print(f"    Created {len(events)} epoch events")
        
        # Expected number of events
        expected_events = n_total_videos * self.concepts_per_video
        print(f"    Expected events: {expected_events}, Created: {len(events)}")
        
        # Create epochs with appropriate rejection criteria
        epoch_duration = self.concept_duration  # 10.0 seconds
        baseline = (-0.2, 0.0) # was (-0.2, 0.0) but changed to (0.0, 0.0) for no baseline correction
        
        # Appropriate rejection criteria for 200 Hz data
        reject_criteria = {
            'eeg': 300e-6,  # 300µV threshold (reasonable for preprocessed data)
        }
        
        flat_criteria = {
            'eeg': 0.5e-6,  # 0.5µV flat threshold
        }
        
        try:
            # Create event_id dictionary
            event_id = {f'concept_{i}': i for i in range(1, len(events) + 1)}
            
            epochs = mne.Epochs(raw, events, event_id=event_id, 
                            tmin=-0.2, tmax=epoch_duration, 
                            baseline=baseline, preload=True,
                            reject=reject_criteria, flat=flat_criteria,
                            reject_by_annotation=True)
            
            print(f"    Created {len(epochs)} epochs of {epoch_duration}s each")
            print(f"    Rejected {len(events) - len(epochs)} epochs due to artifacts")
            
            # If too few epochs, try with more lenient criteria
            if len(epochs) < len(events) * 0.5:  # If more than 50% rejected
                print("    High rejection rate, trying with more lenient criteria...")
                
                lenient_reject = {
                    'eeg': 400e-6,  # More lenient: 400µV
                }
                
                epochs = mne.Epochs(raw, events, event_id=event_id, 
                                tmin=-0.2, tmax=epoch_duration, 
                                baseline=baseline, preload=True,
                                reject=lenient_reject, flat=flat_criteria,
                                reject_by_annotation=False)
                
                print(f"    Created {len(epochs)} epochs with lenient criteria")
                
        except Exception as e:
            print(f"    Epoch creation with rejection failed: {e}")
            # Try without rejection
            try:
                event_id = {f'concept_{i}': i for i in range(1, len(events) + 1)}
                epochs = mne.Epochs(raw, events, event_id=event_id, 
                                tmin=-0.2, tmax=epoch_duration, 
                                baseline=baseline, preload=True,
                                reject=None, flat=None,
                                reject_by_annotation=False)
                print(f"    Created {len(epochs)} epochs without rejection")
            except Exception as e2:
                print(f"    Complete epoch creation failure: {e2}")
                return None
        
        return epochs

    def downsample_data(self, epochs):
        """UPDATED: No downsampling needed - data is already at optimal 200 Hz."""
        print(f"  Data already at optimal sampling rate: {epochs.info['sfreq']} Hz")
        print("  Skipping downsampling step...")
        return epochs

    def apply_advanced_normalization(self, epochs):
        """Fixed normalization with more conservative outlier removal."""
        print("  Applying advanced Z-score normalization...")
        
        # Get data
        data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        original_n_epochs = data.shape[0]
        
        # Step 1: More conservative outlier epoch removal
        epoch_max_vals = np.max(np.abs(data), axis=(1, 2))  # Max absolute value per epoch
        epoch_threshold = np.percentile(epoch_max_vals, 98)  # Changed from 95 to 98
        
        good_epochs = epoch_max_vals <= epoch_threshold * 3  # Changed from 2x to 3x
        
        if np.sum(~good_epochs) > 0:
            print(f"    Removing {np.sum(~good_epochs)} outlier epochs")
            data = data[good_epochs]
            # Update epochs object
            epochs = epochs[good_epochs]
        
        # Only proceed if we have enough epochs
        if data.shape[0] < 2:
            print("    Warning: Too few epochs for normalization, skipping outlier removal")
            data = epochs.get_data()  # Use all epochs
        
        # Step 2: Z-score normalization per channel across all epochs
        data = epochs.get_data()  # Get updated data
        
        for ch_idx in range(data.shape[1]):
            channel_data = data[:, ch_idx, :].flatten()  # Flatten across epochs and time
            
            # Use regular statistics instead of robust for better stability
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            
            if std_val > 1e-10:  # Avoid division by very small numbers
                data[:, ch_idx, :] = (data[:, ch_idx, :] - mean_val) / std_val
            else:
                # If std is too small, just center the data
                data[:, ch_idx, :] = data[:, ch_idx, :] - mean_val
        
        # Update epochs data
        epochs._data = data
        
        print(f"    Normalized data shape: {data.shape}")
        if original_n_epochs != data.shape[0]:
            print(f"    Kept {data.shape[0]}/{original_n_epochs} epochs after outlier removal")
        
        return epochs

    def save_preprocessed_data(self, epochs, subject_id):
        """Save preprocessed epochs data with enhanced metadata."""
        print("  Saving preprocessed data...")
        
        # Get data in format suitable for deep learning
        data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_timepoints)
        
        # Save as numpy array
        output_file = self.output_dir / f"{subject_id}_preprocessed.npy"
        np.save(output_file, data)
        
        # Enhanced metadata
        metadata = {
            'subject_id': subject_id,
            'shape': data.shape,
            'sfreq': epochs.info['sfreq'],
            'ch_names': epochs.ch_names,
            'n_epochs': len(epochs),
            'epoch_duration': 10.0,
            'baseline': (-0.2, 0.0),
            'preprocessing_steps': [
                'filtering (0.5-80 Hz, notch 50 Hz)',
                'common_average_reference',
                'enhanced_bad_channel_detection',
                'ASR_artifact_subspace_reconstruction',
                'enhanced_ICA_artifact_removal',
                'enhanced_epoching_with_rejection',
                'downsampling (250 Hz)',
                'robust_z_score_normalization',
                'outlier_epoch_removal'
            ],
            'quality_metrics': {
                'mean_amplitude': float(np.mean(np.abs(data))),
                'std_amplitude': float(np.std(data)),
                'snr_estimate': float(np.mean(data**2) / np.var(data)),
                'n_interpolated_channels': len(getattr(epochs.info, 'bads', [])),
            },
            'preprocessing_parameters': {
                'highpass_freq': 0.5,
                'lowpass_freq': 80.0,
                'notch_freq': 50.0,
                'target_sfreq': self.target_sfreq,
                'reject_threshold_eeg': 150e-6,
                'flat_threshold_eeg': 1e-6,
            }
        }
        
        metadata_file = self.output_dir / f"{subject_id}_metadata.npy"
        np.save(metadata_file, metadata)
        
        print(f"    Saved: {output_file}")
        print(f"    Data shape: {data.shape}")
        print(f"    Quality: mean_amp={metadata['quality_metrics']['mean_amplitude']:.4f}, "
              f"std_amp={metadata['quality_metrics']['std_amplitude']:.4f}")
        
        return output_file

    def plot_preprocessing_comparison(self, raw_before, raw_after, subject_id):
        """Enhanced preprocessing comparison plots."""
        print("  Generating enhanced preprocessing comparison plots...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Before preprocessing - PSD
            psd_before, freqs_before = mne.time_frequency.psd_array_welch(
                raw_before.get_data(), sfreq=raw_before.info['sfreq'], 
                fmin=0.5, fmax=100, n_fft=2048)
            
            # After preprocessing - PSD
            psd_after, freqs_after = mne.time_frequency.psd_array_welch(
                raw_after.get_data(), sfreq=raw_after.info['sfreq'], 
                fmin=0.5, fmax=100, n_fft=2048)
            
            # Plot 1: Power Spectral Density comparison
            axes[0, 0].semilogy(freqs_before, np.mean(psd_before, axis=0), 
                            label='Before', alpha=0.8, color='red')
            axes[0, 0].semilogy(freqs_after, np.mean(psd_after, axis=0), 
                            label='After', alpha=0.8, color='blue')
            axes[0, 0].set_xlabel('Frequency (Hz)')
            axes[0, 0].set_ylabel('Power Spectral Density (V²/Hz)')
            axes[0, 0].set_title('PSD Comparison')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Time domain comparison (first 10 seconds)
            duration = min(10, raw_before.n_times / raw_before.info['sfreq'])
            n_samples = int(duration * raw_before.info['sfreq'])
            
            time_vec_before = np.arange(n_samples) / raw_before.info['sfreq']
            time_vec_after = np.arange(min(n_samples, raw_after.n_times)) / raw_after.info['sfreq']
            
            # Select a representative channel (e.g., Cz)
            ch_idx = raw_before.ch_names.index('Cz') if 'Cz' in raw_before.ch_names else 0
            
            axes[0, 1].plot(time_vec_before, 
                        raw_before.get_data()[ch_idx, :n_samples] * 1e6, 
                        label='Before', alpha=0.7, color='red')
            axes[0, 1].plot(time_vec_after, 
                        raw_after.get_data()[ch_idx, :len(time_vec_after)] * 1e6, 
                        label='After', alpha=0.7, color='blue')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Amplitude (µV)')
            axes[0, 1].set_title(f'Time Domain - Channel {raw_before.ch_names[ch_idx]}')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Channel variance comparison
            var_before = np.var(raw_before.get_data(), axis=1)
            var_after = np.var(raw_after.get_data(), axis=1)
            
            x_pos = np.arange(len(var_before))
            width = 0.35
            
            axes[1, 0].bar(x_pos - width/2, var_before * 1e12, width, 
                        label='Before', alpha=0.7, color='red')
            axes[1, 0].bar(x_pos + width/2, var_after * 1e12, width, 
                        label='After', alpha=0.7, color='blue')
            axes[1, 0].set_xlabel('Channel Index')
            axes[1, 0].set_ylabel('Variance (µV²)')
            axes[1, 0].set_title('Channel Variance Comparison')
            axes[1, 0].legend()
            axes[1, 0].set_xticks(x_pos[::10])  # Show every 10th channel
            axes[1, 0].set_xticklabels([raw_before.ch_names[i] for i in x_pos[::10]], 
                                    rotation=45)
            
            # Plot 4: Frequency band power comparison
            # Define frequency bands
            bands = {
                'Delta (0.5-4 Hz)': (0.5, 4),
                'Theta (4-8 Hz)': (4, 8),
                'Alpha (8-13 Hz)': (8, 13),
                'Beta (13-30 Hz)': (13, 30),
                'Gamma (30-80 Hz)': (30, 80)
            }
            
            band_names = list(bands.keys())
            power_before = []
            power_after = []
            
            for band_name, (low_freq, high_freq) in bands.items():
                # Calculate power in frequency band
                freq_mask_before = (freqs_before >= low_freq) & (freqs_before <= high_freq)
                freq_mask_after = (freqs_after >= low_freq) & (freqs_after <= high_freq)
                
                if np.any(freq_mask_before) and np.any(freq_mask_after):
                    power_before.append(np.mean(np.mean(psd_before[:, freq_mask_before], axis=1)))
                    power_after.append(np.mean(np.mean(psd_after[:, freq_mask_after], axis=1)))
                else:
                    power_before.append(0)
                    power_after.append(0)
            
            x_pos_bands = np.arange(len(band_names))
            
            axes[1, 1].bar(x_pos_bands - width/2, power_before, width, 
                        label='Before', alpha=0.7, color='red')
            axes[1, 1].bar(x_pos_bands + width/2, power_after, width, 
                        label='After', alpha=0.7, color='blue')
            axes[1, 1].set_xlabel('Frequency Bands')
            axes[1, 1].set_ylabel('Average Power (V²/Hz)')
            axes[1, 1].set_title('Frequency Band Power Comparison')
            axes[1, 1].legend()
            axes[1, 1].set_xticks(x_pos_bands)
            axes[1, 1].set_xticklabels(band_names, rotation=45)
            axes[1, 1].set_yscale('log')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"{subject_id}_preprocessing_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved comparison plot: {plot_file}")
            
        except Exception as e:
            print(f"    Error generating plots: {e}")

    def process_subject(self, subject_id, subject_files):
        """Updated processing pipeline with corrected parameters."""
        print(f"\nProcessing subject: {subject_id}")
        print(f"Files: {[f.name for f in subject_files]}")
        
        try:
            # Step 1: Load and concatenate sessions
            print("Step 1: Loading data...")
            concatenated_data = self.load_and_concatenate_sessions(subject_files)
            print(f"  Final data shape: {concatenated_data.shape}")
            
            # Step 2: Create MNE Raw object with CORRECT sampling frequency
            print("Step 2: Creating MNE Raw object...")
            raw = self.create_mne_raw(concatenated_data, subject_id)
            raw_original = raw.copy()
            
            # Step 3: Apply filtering (ONLY notch filter - data already band-pass filtered)
            print("Step 3: Filtering...")
            raw = self.apply_filtering(raw)
            
            # Step 4: Apply rereferencing
            print("Step 4: Rereferencing...")
            raw = self.apply_rereferencing(raw)
            
            # Step 5: Detect and interpolate bad channels
            print("Step 5: Bad channel detection...")
            raw = self.detect_bad_channels(raw)
            
            # Step 6: Apply ASR
            print("Step 6: Artifact Subspace Reconstruction...")
            raw = self.apply_asr(raw)
            
            # Step 7: Apply ICA
            print("Step 7: ICA artifact removal...")
            raw, ica = self.apply_ica_artifact_removal(raw)
            
            # Step 8: Create epochs with CORRECTED timing
            print("Step 8: Creating epochs...")
            epochs = self.create_epochs_enhanced(raw, subject_files)
            
            if epochs is None or len(epochs) == 0:
                print(f"  ERROR: No valid epochs created for {subject_id}")
                return None
            
            # Step 9: Skip downsampling (data already at optimal 200 Hz)
            print("Step 9: Sampling rate check...")
            epochs = self.downsample_data(epochs)
            
            # Step 10: Normalization
            print("Step 10: Normalization...")
            epochs = self.apply_advanced_normalization(epochs)
            
            # Step 11: Save data
            print("Step 11: Saving data...")
            output_file = self.save_preprocessed_data(epochs, subject_id)
            
            # Step 12: Generate comparison plots
            print("Step 12: Generating plots...")
            self.plot_preprocessing_comparison(raw_original, raw, subject_id)
            
            print(f"✓ Successfully processed {subject_id}")
            print(f"  Output: {output_file}")
            print(f"  Final epochs: {len(epochs)}")
            print(f"  Data shape: {epochs.get_data().shape}")
            
            return output_file
            
        except Exception as e:
            print(f"✗ Error processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def run_preprocessing_pipeline(self):
        """Run the complete preprocessing pipeline for all subjects."""
        print("="*60)
        print("SEED-DV EEG Dataset Enhanced Preprocessing Pipeline")
        print("="*60)
        
        # Find all subject files
        subjects = self.find_subject_files()
        print(f"\nFound {len(subjects)} subjects:")
        for subject_id, files in subjects.items():
            print(f"  {subject_id}: {len(files)} session(s)")
        
        # Process each subject
        successful_subjects = []
        failed_subjects = []
        
        for subject_id, subject_files in subjects.items():
            result = self.process_subject(subject_id, subject_files)
            if result:
                successful_subjects.append(subject_id)
            else:
                failed_subjects.append(subject_id)
        
        # Summary
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Total subjects: {len(subjects)}")
        print(f"Successfully processed: {len(successful_subjects)}")
        print(f"Failed: {len(failed_subjects)}")
        
        if failed_subjects:
            print(f"Failed subjects: {failed_subjects}")
        
        print(f"\nPreprocessed data saved in: {self.output_dir}")
        print("="*60)

    def generate_seeddv_events(self, raw_data_shape, sfreq=200):
        """
        Generate precise event timings for SEED-DV dataset based on data structure.
        
        Parameters:
        -----------
        raw_data_shape : tuple
            Shape of the original data (n_blocks, n_channels, n_timepoints)
        sfreq : int, default=200
            EEG sampling frequency in Hz
            
        Returns:
        --------
        events : numpy.ndarray
            MNE-compatible events array with columns [sample, 0, event_id]
        """
        n_blocks, n_channels, n_timepoints_per_block = raw_data_shape
        
        print(f"  Generating events for {n_blocks} blocks...")
        print(f"  Each block: {n_timepoints_per_block} timepoints ({n_timepoints_per_block/sfreq:.1f}s)")
        
        events = []
        event_id = 1
        current_sample = 0
        
        for block_idx in range(n_blocks):
            # Each block contains 40 concept presentations
            block_start_sample = current_sample
            
            for concept_idx in range(40):  # 40 concepts per block
                # Time within block: concept_idx * 13s (3s cue + 10s video)
                concept_start_time = concept_idx * 13.0
                
                # Video starts 3 seconds after concept start (after cue)
                video_start_time = concept_start_time + 3.0
                
                # Convert to absolute sample index
                video_start_sample = block_start_sample + int(video_start_time * sfreq)
                
                events.append([video_start_sample, 0, event_id])
                event_id += 1
            
            # Move to next block
            current_sample += n_timepoints_per_block
        
        events = np.array(events, dtype=int)
        
        print(f"  Generated {len(events)} events")
        print(f"  Expected total events: {n_blocks * 40}")
        print(f"  First event at sample: {events[0, 0]} (time: {events[0, 0]/sfreq:.2f}s)")
        print(f"  Last event at sample: {events[-1, 0]} (time: {events[-1, 0]/sfreq:.2f}s)")
        
        return events


    # Alternative function with more detailed timing validation
    def generate_seeddv_events_with_validation(self, sfreq=1000, onset_time_first_cue_sec=5.0, 
                                            raw_duration_sec=None):
        """
        Generate SEED-DV events with additional validation against raw data duration.
        
        Parameters:
        -----------
        sfreq : int, default=1000
            EEG sampling frequency in Hz
        onset_time_first_cue_sec : float, default=5.0
            Time in seconds from EEG recording start to first hint/cue onset
        raw_duration_sec : float, optional
            Total duration of raw EEG data in seconds for validation
            
        Returns:
        --------
        events : numpy.ndarray
            MNE-compatible events array of shape (280, 3)
        timing_info : dict
            Detailed timing information for debugging
        """
        import numpy as np
        
        # Generate basic events
        events = self.generate_seeddv_events(sfreq, onset_time_first_cue_sec)
        
        # Calculate detailed timing information
        timing_info = {
            'n_events': len(events),
            'first_event_time_sec': events[0, 0] / sfreq,
            'last_event_time_sec': events[-1, 0] / sfreq,
            'expected_experiment_end_sec': events[-1, 0] / sfreq + 10.0,  # +10s for last video
            'average_inter_event_interval_sec': np.mean(np.diff(events[:, 0])) / sfreq,
            'event_timing_assumptions': {
                'onset_first_cue_sec': onset_time_first_cue_sec,
                'hint_duration_sec': 3.0,
                'video_duration_sec': 10.0,
                'rest_between_blocks_sec': 30.0,
                'sampling_freq_hz': sfreq
            }
        }
        
        # Validation against raw data duration if provided
        if raw_duration_sec is not None:
            timing_info['raw_duration_sec'] = raw_duration_sec
            timing_info['duration_check'] = {
                'sufficient_duration': raw_duration_sec >= timing_info['expected_experiment_end_sec'],
                'margin_sec': raw_duration_sec - timing_info['expected_experiment_end_sec']
            }
            
            if not timing_info['duration_check']['sufficient_duration']:
                print(f"WARNING: Raw data duration ({raw_duration_sec:.1f}s) may be insufficient")
                print(f"         Expected experiment end: {timing_info['expected_experiment_end_sec']:.1f}s")
            else:
                print(f"✓ Raw data duration validation passed (margin: {timing_info['duration_check']['margin_sec']:.1f}s)")
        
        return events, timing_info


    # Integration function to replace your current epoch creation
    def create_epochs_with_inferred_events(self, raw, sfreq=None, onset_time_first_cue_sec=0.0,
                                        tmin=-0.2, tmax=10.0, baseline=(-0.2, 0.0),
                                        reject=None, flat=None):
        """
        Drop-in replacement for epoch creation using inferred SEED-DV event timings.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw EEG data object
        sfreq : int, optional
            Sampling frequency (will use raw.info['sfreq'] if not provided)
        onset_time_first_cue_sec : float, default=5.0
            Time offset for first concept
        tmin, tmax : float
            Epoch time bounds relative to event onset
        baseline : tuple
            Baseline correction period
        reject, flat : dict or None
            Artifact rejection criteria
            
        Returns:
        --------
        epochs : mne.Epochs
            Epochs object with 280 concept events (or fewer after rejection)
        events : numpy.ndarray
            Generated events array
        """
        import mne
        
        if sfreq is None:
            sfreq = raw.info['sfreq']
        
        # Generate events using the inferred timing
        print("Generating SEED-DV events using protocol-based inference...")
        events = self.generate_seeddv_events(sfreq=sfreq, 
                                    onset_time_first_cue_sec=onset_time_first_cue_sec)
        
        # Validate events against raw data duration
        raw_duration_sec = raw.n_times / raw.info['sfreq']
        max_event_time = (events[-1, 0] + int(tmax * sfreq)) / sfreq
        
        if max_event_time > raw_duration_sec:
            print(f"WARNING: Some events extend beyond raw data duration")
            print(f"         Raw duration: {raw_duration_sec:.1f}s")
            print(f"         Latest event end: {max_event_time:.1f}s")
            
            # Filter events that fit within the raw data
            valid_events = []
            for event in events:
                event_end_sample = event[0] + int(tmax * sfreq)
                if event_end_sample < raw.n_times:
                    valid_events.append(event)
            
            events = np.array(valid_events)
            print(f"         Using {len(events)} valid events")
        
        # Create event_id dictionary
        event_id = {f'concept_{i+1}': i+1 for i in range(len(events))}
        
        # Create epochs
        epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=tmin, tmax=tmax, baseline=baseline,
                        reject=reject, flat=flat, preload=True)
        
        print(f"Created {len(epochs)} epochs from {len(events)} events")
        
        return epochs, events

    def create_epochs_from_structure(self, raw, original_data_shape):
        """
        Alternative epoch creation method using data structure.
        
        Parameters:
            -----------
            raw : mne.io.Raw
                MNE Raw object
            original_data_shape : tuple
                Shape of original numpy data (n_blocks, n_channels, n_timepoints)
                
            Returns:
            --------
            epochs : mne.Epochs
                Epochs object
            """
        print("  Creating epochs from data structure...")
        
        # Generate events based on data structure
        events = self.generate_seeddv_events(original_data_shape, self.sfreq)
        
        # Validate events against raw data length
        max_event_sample = events[-1, 0] + int(10.2 * self.sfreq)  # Last event + epoch duration
        if max_event_sample > raw.n_times:
            print(f"    Warning: Some events exceed raw data length")
            print(f"    Raw length: {raw.n_times} samples, Max event needs: {max_event_sample}")
            
            # Filter valid events
            valid_events = []
            for event in events:
                if event[0] + int(10.2 * self.sfreq) <= raw.n_times:
                    valid_events.append(event)
            
            events = np.array(valid_events)
            print(f"    Using {len(events)} valid events")
        
        # Create epochs
        event_id = {f'concept_{i}': i for i in range(1, len(events) + 1)}
        
        epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=-0.2, tmax=10.0, baseline=(-0.2, 0.0),
                        reject={'eeg': 200e-6}, flat={'eeg': 0.5e-6},
                        preload=True)
        
        print(f"    Created {len(epochs)} epochs")
        return epochs
    

# Main execution
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = SEEDDVPreprocessor(
        input_dir="../SEED-DV/EEG",  # Adjust path as needed
        output_dir="preprocessed_eeg_5"
    )
    
    # Run preprocessing pipeline
    preprocessor.run_preprocessing_pipeline()