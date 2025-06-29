#!/usr/bin/env python3
"""
SEED-DV EEG Dataset Initial Data Inspection Script
Phase 1: Data structure and integrity verification

This script performs initial inspection of raw SEED-DV EEG data files
without any preprocessing steps.
"""

import os
import numpy as np
from pathlib import Path

def find_eeg_files(root_dir="../SEED-DV/EEG/"):
    """
    Recursively locate all .npy EEG files within the specified directory.
    
    Args:
        root_dir (str): Root directory to search for EEG files
        
    Returns:
        list: List of Path objects for all .npy files found
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Warning: Directory {root_dir} does not exist.")
        return []
    
    npy_files = list(root_path.rglob("*.npy"))
    return sorted(npy_files)

def inspect_eeg_file(file_path):
    """
    Load and inspect a single EEG .npy file.
    
    Args:
        file_path (Path): Path to the .npy file to inspect
    """
    print(f"\n{'='*60}")
    print(f"INSPECTING: {file_path}")
    print(f"{'='*60}")
    
    try:
        # Load the .npy file
        eeg_data = np.load(file_path)
        
        print(f"Full file path: {file_path.absolute()}")
        print(f"NumPy array shape: {eeg_data.shape}")
        print(f"NumPy array data type: {eeg_data.dtype}")
        
        # Display a small data snippet
        print(f"\nData snippet (first 5 time points for first 3 channels):")
        if len(eeg_data.shape) == 2:
            if eeg_data.shape[0] < eeg_data.shape[1]:  # Likely (channels, timepoints)
                snippet = eeg_data[:3, :5] if eeg_data.shape[0] >= 3 else eeg_data[:, :5]
                print(f"Shape interpretation: (channels={eeg_data.shape[0]}, timepoints={eeg_data.shape[1]})")
            else:  # Likely (timepoints, channels)
                snippet = eeg_data[:5, :3] if eeg_data.shape[1] >= 3 else eeg_data[:5, :]
                print(f"Shape interpretation: (timepoints={eeg_data.shape[0]}, channels={eeg_data.shape[1]})")
            print(snippet)
        else:
            print(f"Unexpected array dimensionality: {len(eeg_data.shape)}D")
            print(f"First few elements: {eeg_data.flat[:10]}")
        
        # Data range inspection
        print(f"\nData amplitude range:")
        print(f"  Minimum: {np.min(eeg_data):.6f}")
        print(f"  Maximum: {np.max(eeg_data):.6f}")
        print(f"  Mean: {np.mean(eeg_data):.6f}")
        print(f"  Standard deviation: {np.std(eeg_data):.6f}")
        
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")

def print_dataset_specifications():
    """
    Print expected SEED-DV dataset specifications from the research report.
    """
    print(f"\n{'='*60}")
    print("SEED-DV DATASET EXPECTED SPECIFICATIONS")
    print(f"{'='*60}")
    
    print("EEG Recording Parameters:")
    print("  - Sampling Rate: 1000 Hz")
    print("  - Number of EEG Channels: 62 (10-10 system, active AgCl electrodes)")
    print("  - EEG System: ESI NeuroScan System")
    print("  - Auxiliary Data: EOG and ECG recorded simultaneously")
    print("  - Eye-tracking: Tobii Pro Fusion at 250 Hz")
    
    print("\nVideo Stimulus Parameters:")
    print("  - Resolution: 1920x1080")
    print("  - Frame Rate: 24 fps")
    print("  - Total Videos: 7 (10 minutes each)")
    print("  - Concepts per Video: 40 random concepts")
    print("  - Concept Block Structure: 3-second Chinese cue + 5 x 2-second video clips")
    print("  - Total Concept Block Duration: 13 seconds")
    
    print("\nAuxiliary Data Identification:")
    print("  - EOG and ECG signals are recorded simultaneously with EEG")
    print("  - These may appear as:")
    print("    * Additional channels beyond the 62 EEG channels in the same .npy file")
    print("    * Separate .npy files with naming conventions (e.g., sub1_EOG.npy, sub1_ECG.npy)")
    print("    * Specific channel indices within the main array (check dataset documentation)")
    
    print("\nMultiple Session Handling:")
    print("  Two common approaches for handling multiple sessions per subject:")
    print("  1. CONCATENATION (for continuous processing):")
    print("     - Combine all sessions into a single continuous stream")
    print("     - Suitable when no significant baseline shifts or setup changes occur")
    print("     - Allows unified application of filters and artifact correction")
    print("  2. SEPARATE PROCESSING (for robust handling):")
    print("     - Preprocess each session independently")
    print("     - More robust to inter-session variability (impedance changes, subject state)")
    print("     - Combine data after individual session preprocessing")
    print("     - Requires careful management of session-specific metadata alignment")

def main():
    """
    Main function to execute the data inspection workflow.
    """
    print("SEED-DV EEG Dataset Initial Data Inspection")
    print("Phase 1: Data Structure and Integrity Verification")
    
    # Find all .npy files
    eeg_files = find_eeg_files()
    
    if not eeg_files:
        print("No .npy files found in SEED-DV/EEG/ directory.")
        print("Please ensure the dataset is properly organized and the path is correct.")
        return
    
    print(f"\nFound {len(eeg_files)} .npy files:")
    for i, file_path in enumerate(eeg_files[:10]):  # Show first 10 files
        print(f"  {i+1}. {file_path}")
    if len(eeg_files) > 10:
        print(f"  ... and {len(eeg_files) - 10} more files")
    
    # Inspect the first file
    print(f"\nInspecting the first file for detailed analysis...")
    inspect_eeg_file(eeg_files[0])
    
    # Print dataset specifications
    print_dataset_specifications()
    
    print(f"\n{'='*60}")
    print("INSPECTION COMPLETE")
    print(f"{'='*60}")
    print("Next steps:")
    print("1. Verify that the data shape matches expected specifications")
    print("2. Check if auxiliary EOG/ECG channels are present")
    print("3. Ensure data amplitude ranges are reasonable for EEG (typically μV range)")
    print("4. Proceed to Phase 2: Full preprocessing pipeline implementation")

if __name__ == "__main__":
    main()
