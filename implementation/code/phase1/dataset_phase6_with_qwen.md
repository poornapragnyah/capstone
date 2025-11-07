# Dataset Output Specification

This document details the structure and shape of the data returned by the `EEGTextMetaDataset` loader. Each item retrieved from the dataset is a tuple containing three components: EEG data, tokenized text, and a metadata vector.

---

## EEG Data

* **Tensor Name**: `eeg_tensor`
* **Shape**: `(62, 400)`
* **Data Type**: `torch.float32`
* **Breakdown**:
    * **62**: Number of EEG channels.
    * **400**: Time-series data points per channel.

---

## Tokenized Captions

* **Tensor Name**: `input_ids`
* **Shape**: `(64)`
* **Data Type**: `torch.int64`
* **Breakdown**:
    * **64**: Maximum sequence length (`max_length`).  
      Each caption is tokenized, padded, or truncated to this length using a BERT tokenizer.

---

## Metadata

* **Tensor Name**: `metadata_tensor`
* **Shape**: `(1 + Number of Unique Objects)`
* **Data Type**: `torch.float32`
* **Breakdown**:
    1. **Scalar Feature (Index 0)**:
        * `Index 0`: **Color ID** — integer representing the dominant color in the image or scene.
          The mapping is loaded from the external file `color_id_to_name_qwen.json` located in the data directory.
          If a sample is missing a valid color entry, `"black"` is used as the fallback.
    2. **Object Vector (From Index 1 onwards)**:
        * Multi-hot encoded vector representing the presence of objects listed in `semantic_features.objects`.
        * The object vocabulary is loaded from the external file `object_id_to_name_qwen.json` in the data directory.
        * Length equals the total number of unique objects defined in that file.
        * A value of `1.0` indicates presence of the corresponding object, while `0.0` indicates absence.