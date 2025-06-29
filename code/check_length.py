import numpy as np
import os

def inspect_npy_data_type(file_path):
    """
    Loads an .npy file, inspects its shape, data type, and value range
    to help determine if it contains raw EEG voltage or extracted features.

    Args:
        file_path (str): The full path to one of your .npy files (e.g., 'data/sub10.npy').
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"\n--- Inspecting: {file_path} ---")

    try:
        data = np.load(file_path)

        print(f"Loaded data shape: {data.shape}")
        print(f"Data type (dtype): {data.dtype}")

        # Flatten the data for statistical analysis across all values,
        # but keep it structured for viewing a slice
        flattened_data = data.flatten()

        print(f"\nStatistical Summary (across all values):")
        print(f"  Min value: {flattened_data.min()}")
        print(f"  Max value: {flattened_data.max()}")
        print(f"  Mean value: {flattened_data.mean()}")
        print(f"  Standard Deviation: {flattened_data.std()}")

        # Display a small slice of the data to see actual values
        # Show first 5 samples from first 2 channels of the first segment
        if data.ndim == 3: # Assuming (7, 62, 104000) structure
            print(f"\nSample Data Slice (first segment, first 2 channels, first 5 time points):")
            print(data[0, :2, :5])
        elif data.ndim == 2: # If it's already flattened into 2D like (channels, time)
            print(f"\nSample Data Slice (first 2 channels, first 5 time points):")
            print(data[:2, :5])
        else:
            print("\nCannot display slice: Data is not 2D or 3D as expected for EEG.")


    except Exception as e:
        print(f"An error occurred while loading or inspecting the file: {e}")

# --- CRUCIAL ACTION ITEM: RUN THIS CODE ---
# Replace 'path/to/your/data_folder/sub10.npy' with the actual path to one of your .npy files
# It's best to use a file you know works, like 'sub10.npy'
example_file_path = '../SEED-DV/EEG/sub10.npy'
inspect_npy_data_type(example_file_path)