#!/usr/bin/env python3
"""
SEED-DV EEG Preprocessed Data Quality Assessment
===============================================
Comprehensive quantitative evaluation of preprocessed EEG data
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats, signal
from scipy.stats import normaltest, kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

class EEGQualityAssessment:
    def __init__(self, data_dir="preprocessed_eeg", sampling_rate=250):
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.results = {}
        self.summary_stats = {}
        
    def load_all_data(self):
        """Load all preprocessed data files"""
        data_files = list(self.data_dir.glob("*_preprocessed.npy"))
        if not data_files:
            raise FileNotFoundError(f"No preprocessed files found in {self.data_dir}")
        
        self.data = {}
        self.subjects = []
        
        print(f"Loading {len(data_files)} preprocessed files...")
        for file_path in sorted(data_files):
            subject_id = file_path.stem.replace('_preprocessed', '')
            try:
                data = np.load(file_path)
                self.data[subject_id] = data
                self.subjects.append(subject_id)
                print(f"  Loaded {subject_id}: {data.shape}")
            except Exception as e:
                print(f"  Error loading {subject_id}: {e}")
        
        print(f"Successfully loaded {len(self.subjects)} subjects")
        
    def compute_signal_quality_metrics(self):
        """Compute comprehensive signal quality metrics"""
        print("\n" + "="*50)
        print("SIGNAL QUALITY ASSESSMENT")
        print("="*50)
        
        metrics = {
            'subject': [],
            'n_epochs': [],
            'n_channels': [],
            'n_timepoints': [],
            'mean_amplitude': [],
            'std_amplitude': [],
            'snr_db': [],
            'signal_range': [],
            'zero_variance_channels': [],
            'high_amplitude_epochs': [],
            'normality_p_value': [],
            'skewness': [],
            'kurtosis_val': [],
            'spectral_entropy': [],
            'alpha_power': [],
            'beta_power': [],
            'gamma_power': [],
            'theta_power': [],
            'delta_power': []
        }
        
        for subject in self.subjects:
            data = self.data[subject]  # Shape: (epochs, channels, timepoints)
            
            # Basic shape metrics
            n_epochs, n_channels, n_timepoints = data.shape
            
            # Amplitude statistics
            mean_amp = np.mean(data)
            std_amp = np.std(data)
            signal_range = np.max(data) - np.min(data)
            
            # Signal-to-noise ratio (approximate)
            signal_power = np.mean(data**2)
            noise_power = np.var(data - signal.savgol_filter(data.flatten(), 51, 3).reshape(data.shape))
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # Data quality checks
            zero_var_channels = np.sum(np.std(data, axis=(0, 2)) < 1e-10)
            high_amp_epochs = np.sum(np.max(np.abs(data), axis=(1, 2)) > 5 * std_amp)
            
            # Statistical properties
            flat_data = data.flatten()
            _, normality_p = normaltest(flat_data[:10000])  # Sample for speed
            skewness_val = skew(flat_data)
            kurtosis_val = kurtosis(flat_data)
            
            # Spectral analysis
            freqs, psd = signal.welch(data, fs=self.sampling_rate, axis=2)
            
            # Frequency band power
            delta_band = (freqs >= 1) & (freqs <= 4)
            theta_band = (freqs >= 4) & (freqs <= 8)
            alpha_band = (freqs >= 8) & (freqs <= 13)
            beta_band = (freqs >= 13) & (freqs <= 30)
            gamma_band = (freqs >= 30) & (freqs <= 80)
            
            delta_power = np.mean(psd[:, :, delta_band])
            theta_power = np.mean(psd[:, :, theta_band])
            alpha_power = np.mean(psd[:, :, alpha_band])
            beta_power = np.mean(psd[:, :, beta_band])
            gamma_power = np.mean(psd[:, :, gamma_band])
            
            # Spectral entropy
            psd_norm = psd / (np.sum(psd, axis=2, keepdims=True) + 1e-10)
            spectral_entropy = -np.mean(np.sum(psd_norm * np.log(psd_norm + 1e-10), axis=2))
            
            # Store metrics
            metrics['subject'].append(subject)
            metrics['n_epochs'].append(n_epochs)
            metrics['n_channels'].append(n_channels)
            metrics['n_timepoints'].append(n_timepoints)
            metrics['mean_amplitude'].append(mean_amp)
            metrics['std_amplitude'].append(std_amp)
            metrics['snr_db'].append(snr_db)
            metrics['signal_range'].append(signal_range)
            metrics['zero_variance_channels'].append(zero_var_channels)
            metrics['high_amplitude_epochs'].append(high_amp_epochs)
            metrics['normality_p_value'].append(normality_p)
            metrics['skewness'].append(skewness_val)
            metrics['kurtosis_val'].append(kurtosis_val)
            metrics['spectral_entropy'].append(spectral_entropy)
            metrics['alpha_power'].append(alpha_power)
            metrics['beta_power'].append(beta_power)
            metrics['gamma_power'].append(gamma_power)
            metrics['theta_power'].append(theta_power)
            metrics['delta_power'].append(delta_power)
        
        self.quality_df = pd.DataFrame(metrics)
        return self.quality_df
    
    def assess_data_consistency(self):
        """Check consistency across subjects"""
        print("\n" + "="*50)
        print("DATA CONSISTENCY ASSESSMENT")
        print("="*50)
        
        shapes = [self.data[subject].shape for subject in self.subjects]
        unique_shapes = list(set(shapes))
        
        print(f"Unique data shapes found: {unique_shapes}")
        
        if len(unique_shapes) == 1:
            print("✓ All subjects have consistent data shapes")
        else:
            print("⚠ Inconsistent data shapes detected!")
            for subject, shape in zip(self.subjects, shapes):
                print(f"  {subject}: {shape}")
        
        # Check sampling rate consistency
        expected_duration = 10.0  # seconds per epoch
        actual_durations = [shape[2] / self.sampling_rate for shape in shapes]
        
        print(f"\nExpected epoch duration: {expected_duration}s")
        print(f"Actual epoch durations: {np.mean(actual_durations):.2f}±{np.std(actual_durations):.3f}s")
        
        if np.std(actual_durations) < 0.01:
            print("✓ Consistent epoch durations")
        else:
            print("⚠ Inconsistent epoch durations detected!")
    
    def detect_artifacts_and_outliers(self):
        """Detect remaining artifacts and outliers"""
        print("\n" + "="*50)
        print("ARTIFACT & OUTLIER DETECTION")
        print("="*50)
        
        artifact_summary = {
            'subject': [],
            'high_amplitude_ratio': [],
            'high_gradient_ratio': [],
            'flatline_ratio': [],
            'outlier_epochs': []
        }
        
        for subject in self.subjects:
            data = self.data[subject]
            n_epochs, n_channels, n_timepoints = data.shape
            
            # High amplitude detection (>200μV equivalent in normalized data)
            high_amp_mask = np.abs(data) > 5 * np.std(data)
            high_amp_ratio = np.mean(high_amp_mask)
            
            # High gradient detection (sudden jumps)
            gradients = np.abs(np.diff(data, axis=2))
            high_grad_mask = gradients > 10 * np.std(gradients)
            high_grad_ratio = np.mean(high_grad_mask)
            
            # Flatline detection
            variance_per_epoch = np.var(data, axis=2)
            flatline_mask = variance_per_epoch < 0.01 * np.mean(variance_per_epoch)
            flatline_ratio = np.mean(flatline_mask)
            
            # Outlier epoch detection (using z-score)
            epoch_features = np.mean(np.abs(data), axis=(1, 2))
            z_scores = np.abs(stats.zscore(epoch_features))
            outlier_epochs = np.sum(z_scores > 3)
            
            artifact_summary['subject'].append(subject)
            artifact_summary['high_amplitude_ratio'].append(high_amp_ratio)
            artifact_summary['high_gradient_ratio'].append(high_grad_ratio)
            artifact_summary['flatline_ratio'].append(flatline_ratio)
            artifact_summary['outlier_epochs'].append(outlier_epochs)
        
        self.artifact_df = pd.DataFrame(artifact_summary)
        
        # Print summary
        print(f"Average high amplitude ratio: {np.mean(artifact_summary['high_amplitude_ratio']):.4f}")
        print(f"Average high gradient ratio: {np.mean(artifact_summary['high_gradient_ratio']):.4f}")
        print(f"Average flatline ratio: {np.mean(artifact_summary['flatline_ratio']):.4f}")
        print(f"Total outlier epochs: {np.sum(artifact_summary['outlier_epochs'])}")
        
        return self.artifact_df
    
    def generate_quality_report(self):
        """Generate comprehensive quality assessment report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE QUALITY ASSESSMENT REPORT")
        print("="*60)
        
        # Overall statistics
        total_epochs = self.quality_df['n_epochs'].sum()
        total_channels = self.quality_df['n_channels'].iloc[0]
        
        print(f"Dataset Overview:")
        print(f"  Total subjects: {len(self.subjects)}")
        print(f"  Total epochs: {total_epochs}")
        print(f"  Channels per subject: {total_channels}")
        print(f"  Total timepoints per epoch: {self.quality_df['n_timepoints'].iloc[0]}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        
        # Signal quality metrics
        print(f"\nSignal Quality Metrics:")
        print(f"  Mean SNR: {self.quality_df['snr_db'].mean():.2f}±{self.quality_df['snr_db'].std():.2f} dB")
        print(f"  Mean amplitude: {self.quality_df['mean_amplitude'].mean():.4f}±{self.quality_df['mean_amplitude'].std():.4f}")
        print(f"  Mean std: {self.quality_df['std_amplitude'].mean():.4f}±{self.quality_df['std_amplitude'].std():.4f}")
        
        # Normalization check
        print(f"\nNormalization Assessment:")
        overall_mean = np.mean([self.quality_df['mean_amplitude'].mean()])
        overall_std = np.mean([self.quality_df['std_amplitude'].mean()])
        print(f"  Overall mean: {overall_mean:.6f} (should be ~0)")
        print(f"  Overall std: {overall_std:.6f} (should be ~1)")
        
        if abs(overall_mean) < 0.1 and abs(overall_std - 1) < 0.2:
            print("  ✓ Data appears properly normalized")
        else:
            print("  ⚠ Data normalization may need review")
        
        # Frequency band analysis
        print(f"\nFrequency Band Power Analysis:")
        print(f"  Delta (1-4 Hz): {self.quality_df['delta_power'].mean():.4f}±{self.quality_df['delta_power'].std():.4f}")
        print(f"  Theta (4-8 Hz): {self.quality_df['theta_power'].mean():.4f}±{self.quality_df['theta_power'].std():.4f}")
        print(f"  Alpha (8-13 Hz): {self.quality_df['alpha_power'].mean():.4f}±{self.quality_df['alpha_power'].std():.4f}")
        print(f"  Beta (13-30 Hz): {self.quality_df['beta_power'].mean():.4f}±{self.quality_df['beta_power'].std():.4f}")
        print(f"  Gamma (30-80 Hz): {self.quality_df['gamma_power'].mean():.4f}±{self.quality_df['gamma_power'].std():.4f}")
        
        # Quality flags
        print(f"\nQuality Flags:")
        zero_var_subjects = self.quality_df[self.quality_df['zero_variance_channels'] > 0]
        if len(zero_var_subjects) > 0:
            print(f"  ⚠ {len(zero_var_subjects)} subjects have zero-variance channels")
        else:
            print("  ✓ No zero-variance channels detected")
        
        high_amp_subjects = self.quality_df[self.quality_df['high_amplitude_epochs'] > 0]
        if len(high_amp_subjects) > 0:
            print(f"  ⚠ {len(high_amp_subjects)} subjects have high-amplitude epochs")
        else:
            print("  ✓ No high-amplitude epochs detected")
        
        # Readiness assessment
        print(f"\nDeep Learning Readiness Assessment:")
        readiness_score = 0
        total_checks = 5
        
        # Check 1: Data consistency
        shapes = [self.data[subject].shape for subject in self.subjects]
        if len(set(shapes)) == 1:
            readiness_score += 1
            print("  ✓ Consistent data shapes across subjects")
        else:
            print("  ⚠ Inconsistent data shapes")
        
        # Check 2: Proper normalization
        if abs(overall_mean) < 0.1 and abs(overall_std - 1) < 0.2:
            readiness_score += 1
            print("  ✓ Proper data normalization")
        else:
            print("  ⚠ Data normalization issues")
        
        # Check 3: No zero variance channels
        if self.quality_df['zero_variance_channels'].sum() == 0:
            readiness_score += 1
            print("  ✓ No zero-variance channels")
        else:
            print("  ⚠ Zero-variance channels present")
        
        # Check 4: Reasonable SNR
        mean_snr = self.quality_df['snr_db'].mean()
        if mean_snr > 10:
            readiness_score += 1
            print(f"  ✓ Good signal-to-noise ratio ({mean_snr:.1f} dB)")
        else:
            print(f"  ⚠ Low signal-to-noise ratio ({mean_snr:.1f} dB)")
        
        # Check 5: Artifact levels
        mean_artifact_ratio = self.artifact_df['high_amplitude_ratio'].mean()
        if mean_artifact_ratio < 0.1:
            readiness_score += 1
            print(f"  ✓ Low artifact levels ({mean_artifact_ratio:.3f})")
        else:
            print(f"  ⚠ High artifact levels ({mean_artifact_ratio:.3f})")
        
        print(f"\nReadiness Score: {readiness_score}/{total_checks}")
        if readiness_score >= 4:
            print("✓ Data is ready for deep learning model training!")
        elif readiness_score >= 3:
            print("⚠ Data is mostly ready but consider addressing flagged issues")
        else:
            print("⚠ Data needs significant improvement before model training")
    
    def create_visualization_dashboard(self):
        """Create comprehensive visualization dashboard"""
        print("\n" + "="*50)
        print("GENERATING VISUALIZATION DASHBOARD")
        print("="*50)
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Data shape consistency
        plt.subplot(3, 4, 1)
        epochs_per_subject = self.quality_df['n_epochs'].values
        plt.bar(range(len(epochs_per_subject)), epochs_per_subject)
        plt.title('Epochs per Subject')
        plt.xlabel('Subject Index')
        plt.ylabel('Number of Epochs')
        
        # 2. SNR distribution
        plt.subplot(3, 4, 2)
        plt.hist(self.quality_df['snr_db'], bins=15, alpha=0.7)
        plt.axvline(self.quality_df['snr_db'].mean(), color='red', linestyle='--', label='Mean')
        plt.title('SNR Distribution')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Count')
        plt.legend()
        
        # 3. Amplitude statistics
        plt.subplot(3, 4, 3)
        plt.scatter(self.quality_df['mean_amplitude'], self.quality_df['std_amplitude'])
        plt.xlabel('Mean Amplitude')
        plt.ylabel('Std Amplitude')
        plt.title('Amplitude Statistics')
        
        # 4. Frequency band powers
        plt.subplot(3, 4, 4)
        band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        band_powers = [
            self.quality_df['delta_power'].mean(),
            self.quality_df['theta_power'].mean(),
            self.quality_df['alpha_power'].mean(),
            self.quality_df['beta_power'].mean(),
            self.quality_df['gamma_power'].mean()
        ]
        plt.bar(band_names, band_powers)
        plt.title('Average Band Powers')
        plt.ylabel('Power')
        plt.xticks(rotation=45)
        
        # 5. Statistical properties
        plt.subplot(3, 4, 5)
        plt.scatter(self.quality_df['skewness'], self.quality_df['kurtosis_val'])
        plt.xlabel('Skewness')
        plt.ylabel('Kurtosis')
        plt.title('Statistical Properties')
        
        # 6. Artifact ratios
        plt.subplot(3, 4, 6)
        artifact_types = ['High Amp', 'High Grad', 'Flatline']
        artifact_ratios = [
            self.artifact_df['high_amplitude_ratio'].mean(),
            self.artifact_df['high_gradient_ratio'].mean(),
            self.artifact_df['flatline_ratio'].mean()
        ]
        plt.bar(artifact_types, artifact_ratios)
        plt.title('Average Artifact Ratios')
        plt.ylabel('Ratio')
        plt.xticks(rotation=45)
        
        # 7. Outlier epochs
        plt.subplot(3, 4, 7)
        plt.bar(range(len(self.artifact_df)), self.artifact_df['outlier_epochs'])
        plt.title('Outlier Epochs per Subject')
        plt.xlabel('Subject Index')
        plt.ylabel('Outlier Epochs')
        
        # 8. Spectral entropy
        plt.subplot(3, 4, 8)
        plt.hist(self.quality_df['spectral_entropy'], bins=15, alpha=0.7)
        plt.title('Spectral Entropy Distribution')
        plt.xlabel('Spectral Entropy')
        plt.ylabel('Count')
        
        # 9. Quality metrics heatmap
        plt.subplot(3, 4, 9)
        quality_metrics = self.quality_df[['snr_db', 'spectral_entropy', 'alpha_power', 'beta_power']].corr()
        sns.heatmap(quality_metrics, annot=True, cmap='coolwarm', center=0)
        plt.title('Quality Metrics Correlation')
        
        # 10. Sample time series
        plt.subplot(3, 4, 10)
        sample_subject = self.subjects[0]
        sample_data = self.data[sample_subject]
        time_axis = np.arange(sample_data.shape[2]) / self.sampling_rate
        plt.plot(time_axis, sample_data[0, 0, :])  # First epoch, first channel
        plt.title(f'Sample Time Series ({sample_subject})')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # 11. Power spectral density
        plt.subplot(3, 4, 11)
        freqs, psd = signal.welch(sample_data[0, 0, :], fs=self.sampling_rate)
        plt.semilogy(freqs, psd)
        plt.title('Sample PSD')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.xlim(0, 80)
        
        # 12. Summary statistics table
        plt.subplot(3, 4, 12)
        plt.axis('off')
        summary_text = f"""
        Dataset Summary:
        Subjects: {len(self.subjects)}
        Total Epochs: {self.quality_df['n_epochs'].sum()}
        Channels: {self.quality_df['n_channels'].iloc[0]}
        
        Quality Metrics:
        Mean SNR: {self.quality_df['snr_db'].mean():.1f} dB
        Mean Entropy: {self.quality_df['spectral_entropy'].mean():.2f}
        
        Artifact Levels:
        High Amp: {self.artifact_df['high_amplitude_ratio'].mean():.3f}
        Outliers: {self.artifact_df['outlier_epochs'].sum()}
        """
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        plt.title('Summary Statistics')
        
        plt.tight_layout()
        
        # Save dashboard
        output_file = self.data_dir / "quality_assessment_dashboard.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to: {output_file}")
        plt.show()
    
    def save_results(self):
        """Save all assessment results"""
        # Save quality metrics
        quality_file = self.data_dir / "quality_metrics.csv"
        self.quality_df.to_csv(quality_file, index=False)
        print(f"Quality metrics saved to: {quality_file}")
        
        # Save artifact metrics
        artifact_file = self.data_dir / "artifact_metrics.csv"
        self.artifact_df.to_csv(artifact_file, index=False)
        print(f"Artifact metrics saved to: {artifact_file}")
        
        # Save summary report
        report_file = self.data_dir / "quality_assessment_report.txt"
        with open(report_file, 'w') as f:
            f.write("SEED-DV EEG Data Quality Assessment Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
            
            f.write("Dataset Overview:\n")
            f.write(f"  Total subjects: {len(self.subjects)}\n")
            f.write(f"  Total epochs: {self.quality_df['n_epochs'].sum()}\n")
            f.write(f"  Channels: {self.quality_df['n_channels'].iloc[0]}\n")
            f.write(f"  Sampling rate: {self.sampling_rate} Hz\n\n")
            
            f.write("Quality Summary:\n")
            f.write(f"  Mean SNR: {self.quality_df['snr_db'].mean():.2f}±{self.quality_df['snr_db'].std():.2f} dB\n")
            f.write(f"  Zero variance channels: {self.quality_df['zero_variance_channels'].sum()}\n")
            f.write(f"  High amplitude epochs: {self.quality_df['high_amplitude_epochs'].sum()}\n")
            f.write(f"  Outlier epochs: {self.artifact_df['outlier_epochs'].sum()}\n")
        
        print(f"Assessment report saved to: {report_file}")
    
    def run_full_assessment(self):
        """Run complete quality assessment pipeline"""
        print("SEED-DV EEG Data Quality Assessment")
        print("=" * 60)
        
        # Load data
        self.load_all_data()
        
        # Run assessments
        self.compute_signal_quality_metrics()
        self.assess_data_consistency()
        self.detect_artifacts_and_outliers()
        
        # Generate report and visualizations
        self.generate_quality_report()
        self.create_visualization_dashboard()
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 60)
        print("QUALITY ASSESSMENT COMPLETE")
        print("=" * 60)
        
        return {
            'quality_metrics': self.quality_df,
            'artifact_metrics': self.artifact_df,
            'subjects': self.subjects
        }

def main():
    # Run quality assessment
    assessor = EEGQualityAssessment(data_dir="preprocessed_eeg", sampling_rate=250)
    results = assessor.run_full_assessment()
    
    return results

if __name__ == "__main__":
    results = main()
