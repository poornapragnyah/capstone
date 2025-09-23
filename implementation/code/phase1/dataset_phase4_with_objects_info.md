### ## EEG Data

* **Dataset Name**: `eeg`
* **Shape**: `(Number of Samples, 62, 400)`
* **Breakdown**:
    * **Number of Samples**: The total number of EEG recordings in the dataset (e.g., 28000).
    * **62**: The number of **EEG channels** (electrodes) used for recording.
    * **400**: The number of **time-series data points** captured for each channel in a single sample.

---

### ## Tokenized Captions

* **Dataset Name**: `input_ids`
* **Shape**: `(Number of Samples, 64)`
* **Breakdown**:
    * **Number of Samples**: The total number of samples, corresponding to the EEG data.
    * **64**: The **maximum sequence length** (`max_length`) of the tokenized text captions. Each caption is padded or truncated to this fixed length.

---

### ## Metadata

* **Dataset Name**: `metadata`
* **Shape**: `(Number of Samples, 3 + Number of Unique Objects)`
* **Breakdown**: This vector is a concatenation of two parts:
    1.  **Fixed Features (First 3 elements)**:
        * `Index 0`: **Scene ID** (An integer representing the scene category).
        * `Index 1`: **Color ID** (An integer for the dominant color).
        * `Index 2`: **Motion Score** (A float value for optical flow).
    2.  **Object Vector (Remaining elements)**:
        * This is a **multi-hot encoded vector** representing the presence of objects.
        * Its length is equal to the total number of unique objects found across all metadata files (e.g., 1013).
        * A `1.0` at a specific position indicates the presence of the corresponding object in that sample, while a `0.0` indicates its absence.

For example, a metadata vector with a shape of `(1016,)` means it contains the **3** fixed features plus a vector for **1013** unique objects.