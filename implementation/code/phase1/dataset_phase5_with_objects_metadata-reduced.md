# Dataset Output Specification

This document details the structure and shape of the data returned by the `EEGTextMetaDataset` loader. Each item retrieved from the dataset is a tuple containing three components: EEG data, tokenized text, and a metadata vector.

---

### ## EEG Data

* **Tensor Name**: `eeg_tensor`
* **Shape**: `(62, 400)`
* **Data Type**: `torch.float32`
* **Breakdown**:
    * **62**: The number of **EEG channels** (electrodes) used for recording.
    * **400**: The number of **time-series data points** captured for each channel in a single sample.

---

### ## Tokenized Captions

* **Data Type**: Dictionary containing `input_ids`, `token_type_ids`, and `attention_mask`.
* **Key Tensor**: `input_ids`
* **Shape**: `(64)`
* **Data Type**: `torch.int64`
* **Breakdown**:
    * **64**: The **maximum sequence length** (`max_length`). Each text caption is tokenized and then padded or truncated to this fixed length for batch processing.

---

### ## Metadata

* **Tensor Name**: `metadata_tensor`
* **Shape**: `(2 + Number of Unique Objects)`
* **Data Type**: `torch.float32`
* **Breakdown**: This vector is a concatenation of two parts:
    1.  **Scalar Features (First 2 elements)**:
        * `Index 0`: **Scene ID** (An integer representing the scene category).
        * `Index 1`: **Color ID** (An integer for the dominant color).
    2.  **Object Vector (Remaining elements, from Index 2 onwards)**:
        * This is a **multi-hot encoded vector** representing the presence of objects.
        * Its length is equal to the total number of unique objects in the vocabulary (e.g., 61 based on your output).
        * A `1.0` at a specific position indicates the presence of the corresponding object, while a `0.0` indicates its absence.

For example, a metadata vector with a shape of `(63,)` means it contains **2** scalar features plus a multi-hot vector for **61** unique objects.