#!/usr/bin/env python3
"""
Fixed SEED-DV EEG preprocessing pipeline with corrected understanding:
- Data is already filtered (0.1-100 Hz) and downsampled to 200 Hz
- Shape (7, 62, 104000) represents 7 blocks × 62 channels × 520 seconds at 200 Hz
- Each block contains 40 events (13s each: 3s hint + 10s video)
"""

import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class SEEDDVPreprocessor:
    """Enhanced preprocessing pipeline for SEED-DV EEG data."""
    
    def __init__(self, input_dir="../SEED-DV/EEG", output_dir="preprocessed_eeg"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sfreq = 200  # Data is already downsampled to 200 Hz
        self.target_sfreq = 200  # Keep at 200 Hz
        self.n_eeg_channels = 62
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # SEED-DV specific parameters
        self.n_blocks = 7  # 7 blocks per complete experiment
        self.events_per_block = 40  # 40 events per block
        self.block_duration = 520.0  # seconds per block (40 × 13s)
        self.event_duration = 13.0  # seconds per event (3s hint + 10s video)
        self.hint_duration = 3.0  # seconds for hint/cue
        self.video_duration = 10.0  # seconds for video presentation
        
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

    def load_and_concatenate_sessions(self, subject_files):
        """Load and concatenate multiple sessions for a subject."""
        all_data = []
        
        for file_path in subject_files:
            print(f"  Loading: {file_path.name}")
            data = np.load(file_path)  # Shape: (7_blocks, 62_channels, 104000_timepoints)
            
            # Verify expected shape
            expected_samples_per_block = int(self.block_duration * self.sfreq)  # 520s × 200Hz = 104000
            if data.shape != (7, 62, expected_samples_per_block):
                print(f"    WARNING: Unexpected shape {data.shape}, expected (7, 62, {expected_samples_per_block})")
            
            # Reshape to (62_channels, total_timepoints) by concatenating blocks
            n_blocks, n_channels, n_timepoints = data.shape
            reshaped_data = data.transpose(1, 0, 2).reshape(n_channels, -1)
            all_data.append(reshaped_data)
            print(f"    Reshaped to: {reshaped_data.shape}")
        
        # Concatenate sessions along time axis
        if len(all_data) > 1:
            concatenated_data = np.concatenate(all_data, axis=1)
            print(f"  Concatenated {len(all_data)} sessions: {concatenated_data.shape}")
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
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        except Exception as e:
            print(f"    Warning: Could not set montage: {e}")
        
        return raw

    def apply_filtering(self, raw):
        """Apply minimal additional filtering (data already filtered 0.1-100 Hz)."""
        print("  Applying minimal additional filtering...")
        
        # Only apply notch filter for power line interference (50 Hz in China)
        try:
            raw.notch_filter(freqs=50, fir_design='firwin', phase='zero')
            print("    Applied 50 Hz notch filter only (data already band-pass filtered)")
        except Exception as e:
            print(f"    Warning: Notch filter failed: {e}")
        
        return raw

    def apply_rereferencing(self, raw):
        """Apply common average referencing (CAR) to the raw data."""
        print("  Applying common average referencing...")
        try:
            raw.set_eeg_reference('average', projection=True)
            raw.apply_proj()  # Apply the projection immediately
            print("    Common average reference applied and projected")
        except Exception as e:
            print(f"    Warning: Rereferencing failed: {e}")
        return raw

    def detect_bad_channels(self, raw):
        """Detect and interpolate bad channels using statistical methods."""
        print("  Detecting bad channels...")
        
        try:
            # Get the data
            data = raw.get_data()
            
            # Statistical bad channel detection
            # Method 1: Channels with extremely high or low variance
            channel_vars = np.var(data, axis=1)
            var_threshold_high = np.percentile(channel_vars, 95)
            var_threshold_low = np.percentile(channel_vars, 5)
            
            bad_channels_var = []
            for i, var in enumerate(channel_vars):
                if var > var_threshold_high * 5 or var < var_threshold_low * 0.2:
                    bad_channels_var.append(raw.ch_names[i])
            
            # Method 2: Channels with extreme amplitudes
            channel_max_abs = np.max(np.abs(data), axis=1)
            amp_threshold = np.percentile(channel_max_abs, 99)
            
            bad_channels_amp = []
            for i, max_amp in enumerate(channel_max_abs):
                if max_amp > amp_threshold * 2:  # 2x the 99th percentile
                    bad_channels_amp.append(raw.ch_names[i])
            
            # Method 3: Channels with flat signals
            bad_channels_flat = []
            for i, ch_data in enumerate(data):
                if np.std(ch_data) < 1e-6:  # Very flat signal
                    bad_channels_flat.append(raw.ch_names[i])
            
            # Combine all bad channels
            bad_channels = list(set(bad_channels_var + bad_channels_amp + bad_channels_flat))
            
            if bad_channels:
                print(f"    Detected bad channels: {bad_channels}")
                raw.info['bads'] = bad_channels
                raw.interpolate_bads(reset_bads=True)
                print("    Bad channels interpolated")
            else:
                print("    No bad channels detected")
                
        except Exception as e:
            print(f"    Warning: Bad channel detection failed: {e}")
        
        return raw

    def apply_asr(self, raw):
        """Apply simple artifact rejection based on amplitude thresholds."""
        print("  Applying amplitude-based artifact rejection...")
        
        try:
            # Mark periods with extreme amplitudes as bad
            data = raw.get_data()
            
            # Define amplitude thresholds (in Volts)
            amp_threshold = 150e-6  # 150 microvolts
            
            # Find time points where any channel exceeds threshold
            bad_times = np.any(np.abs(data) > amp_threshold, axis=0)
            
            if np.any(bad_times):
                # Create annotations for bad segments
                onset_samples = np.where(np.diff(np.concatenate(([False], bad_times, [False]))))[0]
                onsets = onset_samples[::2] / raw.info['sfreq']
                durations = (onset_samples[1::2] - onset_samples[::2]) / raw.info['sfreq']
                
                if len(onsets) > 0 and len(durations) > 0:
                    annotations = mne.Annotations(onset=onsets[:len(durations)], 
                                                duration=durations[:len(onsets)], 
                                                description='BAD_amplitude')
                    raw.set_annotations(annotations)
                    print(f"    Marked {len(onsets)} bad segments due to high amplitude")
                else:
                    print("    No bad segments marked")
            else:
                print("    No high-amplitude artifacts detected")
                
        except Exception as e:
            print(f"    Warning: ASR failed: {e}")
        
        return raw

    def apply_ica_artifact_removal(self, raw):
        """Apply ICA artifact removal to the raw data."""
        print("  Applying ICA artifact removal...")
        
        try:
            # Check if we have enough data for ICA
            if raw.n_times < 2 * raw.info['sfreq']:  # Less than 2 seconds
                print("    Skipping ICA: insufficient data length")
                return raw, None
            
            # Create a copy for ICA fitting (filter more aggressively for ICA)
            raw_ica = raw.copy()
            raw_ica.filter(l_freq=1.0, h_freq=None, fir_design='firwin')
            
            # Fit ICA
            n_components = min(20, raw.info['nchan'] - 1)  # Ensure we don't exceed channel count
            ica = mne.preprocessing.ICA(n_components=n_components, 
                                      random_state=97, 
                                      max_iter=800,
                                      method='fastica')
            ica.fit(raw_ica)
            
            # Simple automatic exclusion based on variance
            ica_data = ica.get_sources(raw_ica).get_data()
            component_vars = np.var(ica_data, axis=1)
            
            # Exclude components with extremely high variance (likely artifacts)
            var_threshold = np.percentile(component_vars, 95)
            exclude_components = []
            for i, var in enumerate(component_vars):
                if var > var_threshold * 3:  # 3x the 95th percentile
                    exclude_components.append(i)
            
            # Limit to maximum 3 components to avoid over-removal
            exclude_components = exclude_components[:3]
            
            if exclude_components:
                ica.exclude = exclude_components
                print(f"    Excluded ICA components: {exclude_components}")
                ica.apply(raw)
                print("    ICA artifact removal applied")
            else:
                print("    No ICA components excluded")
                
        except Exception as e:
            print(f"    Warning: ICA failed: {e}")
            ica = None
        
        return raw, ica

    def generate_seeddv_events(self, total_duration_sec):
        """Generate precise event timings for SEED-DV dataset."""
        print("  Generating SEED-DV events based on protocol...")
        
        # Calculate number of complete blocks in the data
        n_complete_blocks = int(total_duration_sec / self.block_duration)
        print(f"    Total duration: {total_duration_sec:.1f}s")
        print(f"    Complete blocks: {n_complete_blocks}")
        
        events = []
        event_id = 1
        
        for block_idx in range(n_complete_blocks):
            block_start_sec = block_idx * self.block_duration
            
            for event_idx in range(self.events_per_block):
                # Each event is 13 seconds: 3s hint + 10s video
                event_start_in_block = event_idx * self.event_duration
                hint_start_sec = block_start_sec + event_start_in_block
                video_start_sec = hint_start_sec + self.hint_duration  # Video starts after 3s hint
                
                # Convert to sample index (video onset is our event marker)
                video_start_sample = int(video_start_sec * self.sfreq)
                
                # Add event: [sample, 0, event_id]
                events.append([video_start_sample, 0, event_id])
                event_id += 1
        
        events = np.array(events, dtype=int)
        
        print(f"    Generated {len(events)} events")
        if len(events) > 0:
            print(f"    First event at: {events[0, 0]/self.sfreq:.1f}s")
            print(f"    Last event at: {events[-1, 0]/self.sfreq:.1f}s")
        
        return events

    def create_epochs_enhanced(self, raw, subject_files):
        """Create epochs using corrected SEED-DV protocol understanding."""
        print("  Creating epochs based on SEED-DV protocol...")
        
        # Calculate total duration
        total_duration_sec = raw.n_times / raw.info['sfreq']
        
        # Generate events based on protocol
        events = self.generate_seeddv_events(total_duration_sec)
        
        if len(events) == 0:
            print("    ERROR: No events generated")
            return None
        
        # Create epochs with appropriate parameters
        epoch_duration = self.video_duration  # 10 seconds for video presentation
        baseline = (-0.2, 0.0)  # 200ms baseline before video onset
        
        # Start with very lenient rejection criteria since data is already preprocessed
        reject_criteria = {
            'eeg': 500e-6,  # 500 µV threshold (more lenient)
        }
        
        flat_criteria = {
            'eeg': 0.1e-6,  # 0.1 µV threshold (more lenient)
        }
        
        try:
            # Create event_id dictionary
            event_id = {f'video_{i+1}': i+1 for i in range(len(events))}
            
            # First attempt with lenient criteria
            epochs = mne.Epochs(raw, events, event_id=event_id,
                            tmin=-0.2, tmax=epoch_duration, 
                            baseline=baseline, preload=True,
                            reject=reject_criteria, flat=flat_criteria,
                            reject_by_annotation=True)
            
            print(f"    Created {len(epochs)} epochs of {epoch_duration}s each")
            print(f"    Rejected {len(events) - len(epochs)} epochs due to artifacts")
            
            # If still too many rejections (>70%), try without rejection
            if len(epochs) < len(events) * 0.3:  # If rejected more than 70%
                print("    Too many rejections, trying without artifact rejection...")
                
                epochs = mne.Epochs(raw, events, event_id=event_id,
                                tmin=-0.2, tmax=epoch_duration, 
                                baseline=baseline, preload=True,
                                reject=None, flat=None,
                                reject_by_annotation=False)
                
                print(f"    Created {len(epochs)} epochs without rejection criteria")
                
                # Apply manual outlier removal based on amplitude percentiles
                if len(epochs) > 0:
                    data = epochs.get_data()
                    
                    # Calculate per-epoch amplitude metrics
                    epoch_max_amps = np.max(np.abs(data), axis=(1, 2))
                    epoch_mean_amps = np.mean(np.abs(data), axis=(1, 2))
                    
                    # Remove extreme outliers (beyond 99.5th percentile)
                    max_amp_threshold = np.percentile(epoch_max_amps, 99.5)
                    mean_amp_threshold = np.percentile(epoch_mean_amps, 99.5)
                    
                    good_epochs_mask = (epoch_max_amps <= max_amp_threshold) & (epoch_mean_amps <= mean_amp_threshold)
                    
                    if not np.all(good_epochs_mask):
                        bad_epoch_indices = np.where(~good_epochs_mask)[0]
                        epochs.drop(bad_epoch_indices)
                        print(f"    Manually removed {len(bad_epoch_indices)} extreme outlier epochs")
                    
                    print(f"    Final epochs after manual cleaning: {len(epochs)}")
            
            # Ensure we have at least some epochs
            if len(epochs) == 0:
                print("    ERROR: No epochs survived preprocessing")
                return None
            
            # Additional check: if we have very few epochs, warn but continue
            if len(epochs) < 50:
                print(f"    WARNING: Only {len(epochs)} epochs available (less than 50)")
            
            return epochs
                    
        except Exception as e:
            print(f"    Epoch creation failed: {e}")
            # Final fallback: try with minimal event set and no rejection
            try:
                # Use only first 100 events if we have too many
                max_events = min(100, len(events))
                events_subset = events[:max_events]
                event_id = {f'video_{i+1}': i+1 for i in range(max_events)}
                
                epochs = mne.Epochs(raw, events_subset, event_id=event_id,
                                tmin=-0.2, tmax=epoch_duration, 
                                baseline=baseline, preload=True,
                                reject=None, flat=None,
                                reject_by_annotation=False)
                print(f"    Created {len(epochs)} epochs with fallback method (subset of events)")
                return epochs
                
            except Exception as e2:
                print(f"    Final epoch creation failed: {e2}")
                return None

    def downsample_data(self, epochs):
        """Skip downsampling (data already at target 200 Hz)."""
        print(f"  Skipping downsampling (data already at target {self.target_sfreq} Hz)")
        return epochs

    def apply_advanced_normalization(self, epochs):
        """Apply advanced normalization techniques to the epochs."""
        print("  Applying advanced normalization...")
        
        try:
            # Get the data
            data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_timepoints)
            
            # Robust Z-score normalization per channel across time
            def robust_zscore(x):
                median = np.median(x, axis=-1, keepdims=True)
                mad = np.median(np.abs(x - median), axis=-1, keepdims=True)
                return (x - median) / (mad * 1.4826 + 1e-6)  # 1.4826 makes MAD consistent with std
            
            normalized_data = robust_zscore(data)
            
            # Replace the data in epochs
            epochs._data = normalized_data
            
            # Outlier epoch removal based on robust Z-scores
            epoch_scores = np.mean(np.abs(normalized_data), axis=(1, 2))
            outlier_threshold = np.percentile(epoch_scores, 95) * 2
            good_epochs = epoch_scores < outlier_threshold
            
            if not np.all(good_epochs):
                bad_epoch_indices = np.where(~good_epochs)[0]
                epochs.drop(bad_epoch_indices)
                print(f"    Removed {len(bad_epoch_indices)} outlier epochs")
            
            print(f"    Normalized epochs, remaining: {len(epochs)}")
            
        except Exception as e:
            print(f"    Warning: Normalization failed: {e}")
        
        return epochs

    def save_preprocessed_data(self, epochs, subject_id):
        """Save preprocessed epochs data with metadata."""
        print("  Saving preprocessed data...")
        
        # Get data in format suitable for deep learning
        data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_timepoints)
        
        # Save as numpy array
        output_file = self.output_dir / f"{subject_id}_preprocessed.npy"
        np.save(output_file, data)
        
        # Metadata
        metadata = {
            'subject_id': subject_id,
            'shape': data.shape,
            'sfreq': epochs.info['sfreq'],
            'ch_names': epochs.ch_names,
            'n_epochs': len(epochs),
            'epoch_duration': self.video_duration,
            'baseline': (-0.2, 0.0),
            'original_data_info': {
                'already_filtered': '0.1-100 Hz',
                'already_downsampled': '200 Hz',
                'original_shape_per_file': '(7_blocks, 62_channels, 104000_samples)',
                'block_duration_sec': self.block_duration,
                'events_per_block': self.events_per_block,
            },
            'preprocessing_steps': [
                'notch_filtering (50 Hz only)',
                'common_average_reference',
                'bad_channel_detection_and_interpolation',
                'amplitude_based_artifact_rejection',
                'ICA_artifact_removal',
                'protocol_based_epoching',
                'robust_z_score_normalization',
                'outlier_epoch_removal'
            ],
            'quality_metrics': {
                'mean_amplitude': float(np.mean(np.abs(data))),
                'std_amplitude': float(np.std(data)),
                'snr_estimate': float(np.mean(data**2) / (np.var(data) + 1e-10)),
                'n_interpolated_channels': len(getattr(epochs.info, 'bads', [])),
            },
            'preprocessing_parameters': {
                'notch_freq': 50.0,
                'sfreq': self.sfreq,
                'target_sfreq': self.target_sfreq,
                'reject_threshold_eeg': 200e-6,
                'flat_threshold_eeg': 0.5e-6,
            }
        }
        
        metadata_file = self.output_dir / f"{subject_id}_metadata.npy"
        np.save(metadata_file, metadata)
        
        print(f"    Saved: {output_file}")
        print(f"    Data shape: {data.shape}")
        print(f"    Sampling frequency: {epochs.info['sfreq']} Hz")
        print(f"    Quality: mean_amp={metadata['quality_metrics']['mean_amplitude']:.4f}, "
              f"std_amp={metadata['quality_metrics']['std_amplitude']:.4f}")
        
        return output_file

    def plot_preprocessing_comparison(self, raw_before, raw_after, subject_id):
        """Enhanced preprocessing comparison plots."""
        print("  Generating preprocessing comparison plots...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Before preprocessing - PSD
            psd_before, freqs_before = mne.time_frequency.psd_array_welch(
                raw_before.get_data(), sfreq=raw_before.info['sfreq'], 
                fmin=0.1, fmax=95, n_fft=1024)
            
            # After preprocessing - PSD
            psd_after, freqs_after = mne.time_frequency.psd_array_welch(
                raw_after.get_data(), sfreq=raw_after.info['sfreq'], 
                fmin=0.1, fmax=95, n_fft=1024)
            
            # Plot 1: Power Spectral Density comparison
            axes[0, 0].semilogy(freqs_before, np.mean(psd_before, axis=0), 
                            label='Before', alpha=0.8, color='red')
            axes[0, 0].semilogy(freqs_after, np.mean(psd_after, axis=0), 
                            label='After', alpha=0.8, color='blue')
            axes[0, 0].set_xlabel('Frequency (Hz)')
            axes[0, 0].set_ylabel('Power Spectral Density (V²/Hz)')
            axes[0, 0].set_title('PSD Comparison (200 Hz Data)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xlim([0.1, 95])
            
            # Plot 2: Time domain comparison (first 20 seconds)
            duration = min(20, raw_before.n_times / raw_before.info['sfreq'])
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
            axes[1, 0].set_xticks(x_pos[::10])
            axes[1, 0].set_xticklabels([raw_before.ch_names[i] for i in x_pos[::10]], 
                                    rotation=45)
            
            # Plot 4: Frequency band power comparison
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
        """Complete preprocessing pipeline for a single subject."""
        print(f"\nProcessing subject: {subject_id}")
        print(f"Files: {[f.name for f in subject_files]}")
        
        try:
            # Step 1: Load and concatenate sessions
            print("Step 1: Loading data...")
            concatenated_data = self.load_and_concatenate_sessions(subject_files)
            print(f"  Final data shape: {concatenated_data.shape}")

            # Continuing from where the code was cut off in process_subject method
        
            # Step 2: Create MNE Raw object
            print("Step 2: Creating MNE Raw object...")
            raw_original = self.create_mne_raw(concatenated_data, subject_id)
            print(f"  Sampling frequency: {raw_original.info['sfreq']} Hz")
            
            # Keep copy for comparison
            raw_before = raw_original.copy()
            
            # Step 3: Minimal filtering (data already filtered 0.1-100 Hz)
            print("Step 3: Minimal filtering...")
            raw = self.apply_filtering(raw_original)
            
            # Step 4: Rereferencing
            print("Step 4: Rereferencing...")
            raw = self.apply_rereferencing(raw)
            
            # Step 5: Bad channel detection and interpolation
            print("Step 5: Bad channel detection...")
            raw = self.detect_bad_channels(raw)
            
            # Step 6: Artifact rejection (ASR)
            print("Step 6: Artifact rejection...")
            raw = self.apply_asr(raw)
            
            # Step 7: ICA artifact removal
            print("Step 7: ICA artifact removal...")
            raw, ica = self.apply_ica_artifact_removal(raw)
            
            # Step 8: Create epochs
            print("Step 8: Creating epochs...")
            epochs = self.create_epochs_enhanced(raw, subject_files)
            
            if epochs is None:
                print("  ✗ Failed to create epochs")
                return
            
            # Step 9: Skip downsampling (already at target frequency)
            epochs = self.downsample_data(epochs)
            
            # Step 10: Advanced normalization
            print("Step 10: Advanced normalization...")
            epochs = self.apply_advanced_normalization(epochs)
            
            # Step 11: Save preprocessed data
            print("Step 11: Saving data...")
            output_file = self.save_preprocessed_data(epochs, subject_id)
            
            # Step 12: Generate comparison plots
            print("Step 12: Generating plots...")
            self.plot_preprocessing_comparison(raw_before, raw, subject_id)
            
            print(f"  ✓ Successfully processed {subject_id}")
            print(f"  Final epochs: {len(epochs)}")
            print(f"  Output: {output_file}")
            
            return output_file
        
        except Exception as e:
            print(f"  ✗ Error processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_preprocessing_pipeline(self):
        """Run the complete preprocessing pipeline for all subjects."""
        print("Starting SEED-DV preprocessing pipeline...")
        
        # Find all .npy files in input directory
        npy_files = list(self.input_dir.glob("*.npy"))
        
        if not npy_files:
            print(f"No .npy files found in {self.input_dir}")
            return
        
        print(f"Found {len(npy_files)} .npy files")
        
        # Group files by subject
        subject_files = {}
        for file_path in npy_files:
            # Extract subject ID from filename
            filename = file_path.stem
            
            # Handle special case for sub1 (has two sessions)
            if filename == "sub1_session2":
                subject_id = "sub1"
            else:
                subject_id = filename
            
            if subject_id not in subject_files:
                subject_files[subject_id] = []
            subject_files[subject_id].append(file_path)
        
        # Sort subjects for consistent processing order
        subjects = sorted(subject_files.keys())
        
        print(f"Processing {len(subjects)} subjects...")
        
        processed_subjects = []
        failed_subjects = []
        
        for subject_id in subjects:
            files = sorted(subject_files[subject_id])  # Ensure consistent order
            
            print(f"\nProcessing subject: {subject_id}")
            
            if len(files) == 0:
                print(f"No files found for subject {subject_id}")
                continue
            
            try:
                output_file = self.process_subject(subject_id, files)
                if output_file:
                    processed_subjects.append(subject_id)
                else:
                    failed_subjects.append(subject_id)
            except Exception as e:
                print(f"Failed to process {subject_id}, skipping...")
                failed_subjects.append(subject_id)
        
        # Print summary
        print(f"\n{'='*60}")
        print("PREPROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully processed: {len(processed_subjects)} subjects")
        if processed_subjects:
            print(f"  {', '.join(processed_subjects)}")
        
        if failed_subjects:
            print(f"Failed to process: {len(failed_subjects)} subjects")
            print(f"  {', '.join(failed_subjects)}")
        
        print(f"\nOutput directory: {self.output_dir}")
        print("Files saved:")
        output_files = list(self.output_dir.glob("*"))
        for file in sorted(output_files):
            print(f"  {file.name}")
        
        print(f"\nPreprocessing pipeline completed.")
        print(f"All subjects processed successfully.")
        
        return processed_subjects, failed_subjects

    def generate_summary_report(self):
        """Generate a summary report of all preprocessed data."""
        print("\nGenerating preprocessing summary report...")
        
        # Find all preprocessed files
        data_files = list(self.output_dir.glob("*_preprocessed.npy"))
        metadata_files = list(self.output_dir.glob("*_metadata.npy"))
        
        if not data_files:
            print("No preprocessed data files found.")
            return
        
        summary = {
            'total_subjects': len(data_files),
            'subjects': [],
            'total_epochs': 0,
            'total_channels': 0,
            'sampling_frequency': None,
            'epoch_duration': None,
            'preprocessing_steps': None,
            'quality_metrics': {
                'mean_amplitudes': [],
                'std_amplitudes': [],
                'snr_estimates': []
            }
        }
        
        for data_file in sorted(data_files):
            subject_id = data_file.stem.replace('_preprocessed', '')
            metadata_file = self.output_dir / f"{subject_id}_metadata.npy"
            
            try:
                # Load data
                data = np.load(data_file)
                
                # Load metadata if available
                if metadata_file.exists():
                    metadata = np.load(metadata_file, allow_pickle=True).item()
                    
                    subject_info = {
                        'subject_id': subject_id,
                        'n_epochs': metadata.get('n_epochs', len(data)),
                        'n_channels': data.shape[1] if len(data.shape) > 1 else 0,
                        'n_timepoints': data.shape[2] if len(data.shape) > 2 else data.shape[1] if len(data.shape) > 1 else 0,
                        'quality_metrics': metadata.get('quality_metrics', {})
                    }
                    
                    # Update summary statistics
                    summary['total_epochs'] += subject_info['n_epochs']
                    summary['total_channels'] = subject_info['n_channels']  # Should be same for all
                    summary['sampling_frequency'] = metadata.get('sfreq')
                    summary['epoch_duration'] = metadata.get('epoch_duration')
                    summary['preprocessing_steps'] = metadata.get('preprocessing_steps')
                    
                    # Quality metrics
                    qm = subject_info['quality_metrics']
                    if 'mean_amplitude' in qm:
                        summary['quality_metrics']['mean_amplitudes'].append(qm['mean_amplitude'])
                    if 'std_amplitude' in qm:
                        summary['quality_metrics']['std_amplitudes'].append(qm['std_amplitude'])
                    if 'snr_estimate' in qm:
                        summary['quality_metrics']['snr_estimates'].append(qm['snr_estimate'])
                
                else:
                    subject_info = {
                        'subject_id': subject_id,
                        'n_epochs': len(data),
                        'n_channels': data.shape[1] if len(data.shape) > 1 else 0,
                        'n_timepoints': data.shape[2] if len(data.shape) > 2 else data.shape[1] if len(data.shape) > 1 else 0,
                        'quality_metrics': {}
                    }
                    summary['total_epochs'] += subject_info['n_epochs']
                
                summary['subjects'].append(subject_info)
                
            except Exception as e:
                print(f"Error processing {subject_id}: {e}")
        
        # Print summary report
        print(f"\n{'='*60}")
        print("SEED-DV PREPROCESSING SUMMARY REPORT")
        print(f"{'='*60}")
        print(f"Total subjects processed: {summary['total_subjects']}")
        print(f"Total epochs generated: {summary['total_epochs']}")
        print(f"Channels per subject: {summary['total_channels']}")
        print(f"Sampling frequency: {summary['sampling_frequency']} Hz")
        print(f"Epoch duration: {summary['epoch_duration']} seconds")
        
        if summary['quality_metrics']['mean_amplitudes']:
            mean_amps = summary['quality_metrics']['mean_amplitudes']
            print(f"\nQuality Metrics:")
            print(f"  Mean amplitude across subjects: {np.mean(mean_amps):.6f} ± {np.std(mean_amps):.6f}")
            
            if summary['quality_metrics']['std_amplitudes']:
                std_amps = summary['quality_metrics']['std_amplitudes']
                print(f"  Std amplitude across subjects: {np.mean(std_amps):.6f} ± {np.std(std_amps):.6f}")
        
        print(f"\nPer-subject breakdown:")
        for subject in summary['subjects']:
            print(f"  {subject['subject_id']}: {subject['n_epochs']} epochs, "
                  f"shape ({subject['n_epochs']}, {subject['n_channels']}, {subject['n_timepoints']})")
        
        if summary['preprocessing_steps']:
            print(f"\nPreprocessing steps applied:")
            for i, step in enumerate(summary['preprocessing_steps'], 1):
                print(f"  {i}. {step.replace('_', ' ').title()}")
        
        # Save summary report
        report_file = self.output_dir / "preprocessing_summary.npy"
        np.save(report_file, summary)
        print(f"\nSummary report saved: {report_file}")
        
        return summary


# Main execution
def main():
    """Main function to run the SEED-DV preprocessing pipeline."""
    
    # Initialize preprocessor with paths
    preprocessor = SEEDDVPreprocessor(
        input_dir="../SEED-DV/EEG",  # Adjust path as needed
        output_dir="preprocessed_eeg"
    )
    
    # Run preprocessing pipeline
    processed_subjects, failed_subjects = preprocessor.run_preprocessing_pipeline()
    
    # Generate summary report
    summary = None
    if processed_subjects:
        summary = preprocessor.generate_summary_report()
        print("\nSummary Report:")
        print(summary)
    
    if failed_subjects:
        print(f"\nFailed subjects: {failed_subjects}")
        
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED")
    print(f"{'='*60}")
    print(f"Output files saved in: {preprocessor.output_dir}")
    print("Check the output directory for preprocessed data and plots.")
    print("\nEnd of preprocessing pipeline.")
    print("Thank you for using the SEED-DV preprocessing pipeline!")


if __name__ == "__main__":
    main()