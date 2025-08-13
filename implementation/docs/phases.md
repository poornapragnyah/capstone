### **Phase 0: Project Setup and Data Loading**

This foundational phase is about preparing your environment and ensuring you can correctly load and preprocess all the necessary data from the SEED-DV dataset.

* **Goal:** Create a reproducible environment and a robust PyTorch `Dataset` and `DataLoader` for the SEED-DV data.
* **Tasks:**
    1.  **Environment Setup:**
        * Create a Python virtual environment (`venv` or `conda`).
        * Install core libraries: `pytorch`, `numpy`, `pandas`.
        * Install specialized libraries: `transformers` (for text models), `torch_geometric` (for GCNs), and `scipy` (to load `.mat` files if needed). Create a `requirements.txt` file.
    2.  **Data Ingestion:**
        * Write a script to parse the SEED-DV dataset. This involves loading the EEG signals, the BLIP-generated captions, and the metadata (color, motion, category).
    3.  **Create a PyTorch `Dataset`:**
        * Implement a custom `torch.utils.data.Dataset` class, let's call it `SEED_DV_Dataset`.
        * The `__getitem__` method of this class should return a single sample: `(eeg_data, tokenized_caption, metadata_labels)`.
        * **EEG Data:** Preprocess the EEG signals into a `[Channels, Time_steps]` tensor (e.g., `[62, 200]`).
        * **Text Data:** Use a tokenizer from the `transformers` library (e.g., `BertTokenizer` or `CLIPTokenizer`) to convert the text captions into `input_ids` and `attention_mask`.
        * **Metadata:** Convert the categorical metadata (color, category) into numerical labels (e.g., 0, 1, 2...). Keep the motion score as a float.
    4.  **Create `DataLoader`s:**
        * Instantiate PyTorch `DataLoader`s for your training, validation, and test sets, using your `SEED_DV_Dataset`. This will handle batching, shuffling, etc.

* **Output/Verification:**
    * **Deliverable:** A Python script `data_loader.py`.
    * **How to Check:** Create a simple test script. Instantiate your `DataLoader`. Pull one batch from it. Print the shapes of the EEG tensor, the tokenized caption tensors, and the metadata tensor. Verify that they are what you expect (e.g., EEG: `[batch_size, 62, 200]`, text input_ids: `[batch_size, seq_len]`, metadata: `[batch_size, 3]`).

---

### **Phase 1: Implement the STGT Encoder**

This is the most novel part of the architecture. We will build it piece by piece.

* **Goal:** Create a `nn.Module` that takes a batch of raw EEG signals and outputs a sequence of spatio-temporally aware embeddings.
* **Tasks:**
    1.  **Graph Construction:** Create the adjacency matrix $A$ for the 62 EEG channels. Start with a static, pre-defined matrix based on physical distance, which you can save as a file. Load it as a `torch.tensor`.
    2.  **Spatial GCN Module:**
        * Using `torch_geometric`, implement a GCN layer (`torch_geometric.nn.GCNConv`).
        * Wrap this in a `nn.Module` that applies the GCN to the graph features at a single time step.
    3.  **Temporal Transformer Module:**
        * Use PyTorch's built-in `nn.TransformerEncoder` and `nn.TransformerEncoderLayer`.
    4.  **Combine into `STGTEncoder`:**
        * Create the main `STGTEncoder` `nn.Module`.
        * Its `forward` pass will:
            a.  Take a batch of EEG data `x` with shape `[batch_size, channels, time_steps]`.
            b.  Loop through the `time_steps` dimension, applying the GCN layer at each step to the `[batch_size, channels, features]` data.
            c.  Stack the outputs from each time step to form a sequence tensor of shape `[batch_size, time_steps, embedding_dim]`.
            d.  Feed this sequence into the `nn.TransformerEncoder`.

* **Output/Verification:**
    * **Deliverable:** A Python file `stgt_encoder.py` containing the `STGTEncoder` class.
    * **How to Check:** Instantiate the encoder. Create a random dummy EEG tensor with the correct input shape (e.g., `torch.randn(32, 62, 200)`). Pass it through the encoder. Assert that the output has the expected shape (e.g., `[32, 200, embedding_dim]`) and that no errors occur. This is a crucial "smoke test".

---

### **Phase 2: Implement Contrastive Alignment (Training Task 1)**

Now, let's bridge the modality gap and set up the primary training objective.

* **Goal:** Create the modules and loss function to align EEG and text embeddings.
* **Tasks:**
    1.  **Load Pre-trained Text Encoder:** From the `transformers` library, load a frozen text encoder (e.g., `CLIPTextModel.from_pretrained(...)`). Set `requires_grad=False` for all its parameters.
    2.  **Implement Projection Heads:** Create two simple `nn.Sequential` MLPs. One will map the final STGT encoder output to the multimodal embedding dimension, and the other will map the text encoder's output to the same dimension.
    3.  **Implement InfoNCE Loss:** Write a Python function `info_nce_loss(eeg_embeds, text_embeds, temperature)`. This function will calculate the cosine similarity matrix between all EEG and text embeddings in the batch and compute the loss as described by the formula:
        $$L_{i,j} = -\log \frac{\exp(\text{sim}(e_{eeg,i}, e_{text,j})/\tau)}{\sum_{k=1}^{N} \exp(\text{sim}(e_{eeg,i}, e_{text,k})/\tau)}$$
        Remember this loss is computed symmetrically (aligning text to EEG as well) for best results.

* **Output/Verification:**
    * **Deliverable:** A script `contrastive_module.py`.
    * **How to Check:** Create two dummy tensors of shape `[batch_size, embedding_dim]` to represent EEG and text embeddings. Pass them to your loss function. Verify it returns a single scalar tensor. For a more robust check, create a batch where `eeg_embeds[i]` and `text_embeds[i]` are identical. The loss should be very low.

---

### **Phase 3: Implement the MTR Head (Training Task 2)**

This phase adds the regularization component.

* **Goal:** Create the auxiliary classification/regression head and its corresponding loss calculation.
* **Tasks:**
    1.  **Build MLP Head:** Create a `nn.Module` called `MTRHead`. This will likely be a simple MLP (`nn.Linear` -> `nn.ReLU` -> `nn.Linear`) that takes the STGT encoder's output embedding as input. The final layer should have a number of outputs corresponding to your tasks (e.g., num_colors + num_categories + 1 for motion).
    2.  **Loss Functions:** Instantiate the necessary loss functions from PyTorch: `nn.CrossEntropyLoss` for color and category classification, and `nn.MSELoss` for motion regression.
    3.  **Combined Auxiliary Loss Function:** Write a helper function that takes the `MTRHead`'s predictions and the ground-truth metadata labels and computes the three auxiliary losses, returning their weighted sum: $L_{aux} = \lambda_1 L_{color} + \lambda_2 L_{motion} + \lambda_3 L_{category}$.

* **Output/Verification:**
    * **Deliverable:** A script `mtr_head.py`.
    * **How to Check:** Pass a dummy embedding from the STGT encoder to the `MTRHead`. Check that the output shape is correct. Then, pass this output and some dummy ground-truth labels to your combined auxiliary loss function. Verify that it returns a single scalar loss value.

---

### **Phase 4: Integrate and Train the Full Encoder Model**

Here we assemble the pieces and train the core representation learning part of STAR-Net. The decoder is not involved yet.

* **Goal:** Train the `STGTEncoder` using the combined contrastive and MTR losses.
* **Tasks:**
    1.  **Create Full STAR-Net Encoder Model:** Create a master `nn.Module` that contains the `STGTEncoder`, the text encoder, the projection heads, and the `MTRHead`.
    2.  **Write the Training Loop:**
        * Iterate through your `DataLoader`.
        * In each step:
            a.  Pass EEG data through `STGTEncoder`.
            b.  Pass captions through the frozen text encoder.
            c.  Map both outputs through their respective projection heads.
            d.  Calculate the contrastive loss $L_{contrastive}$.
            e.  Pass the EEG embedding through the `MTRHead`.
            f.  Calculate the combined auxiliary loss $L_{aux}$.
            g.  Calculate the total loss: $L_{total} = L_{contrastive} + L_{aux}$.
            h.  Perform backpropagation: `loss.backward()`.
            i.  Update weights: `optimizer.step()`.
    3.  **Implement Validation:** After each epoch, run the model on the validation set. Track all loss components (`L_{total}`, `L_{contrastive}`, `L_{aux}`) and also the accuracy for the auxiliary classification tasks.

* **Output/Verification:**
    * **Deliverable:** A `train.py` script.
    * **How to Check:** Run the training script for a few epochs on a small subset of the data. Use a tool like TensorBoard or `wandb` to plot the losses. You should see all loss values trending downwards. The validation accuracy for the auxiliary tasks (e.g., color prediction) should be increasing, proving the encoder is learning meaningful features. **Saving the weights of the best performing encoder is the final deliverable of this phase.**

---

### **Phase 5: Implement the Text Decoder for Inference**

With a trained encoder that produces semantically rich embeddings, we can now generate text.

* **Goal:** Use the trained EEG encoder to generate descriptive text with a pre-trained decoder model.
* **Tasks:**
    1.  **Load a Pre-trained Decoder:** Choose and load an autoregressive model like `BartForConditionalGeneration` or `T5ForConditionalGeneration` from `transformers`.
    2.  **Connect Encoder to Decoder:** The key step is feeding the learned EEG embedding to the decoder. For models like BART or T5, you can pass the final EEG embedding tensor as the `encoder_hidden_states` argument to the model's `generate` method. The decoder will use cross-attention to condition its generation on your EEG representation.
    3.  **Write Inference Script:** Create a script that:
        a.  Loads the saved weights for your trained `STGTEncoder`.
        b.  Takes a single EEG sample from the test set.
        c.  Passes it through the encoder to get the embedding.
        d.  Calls the decoder's `.generate()` method, providing the EEG embedding.
        e.  Decodes the output token IDs back into a human-readable sentence.

* **Output/Verification:**
    * **Deliverable:** An `inference.py` script.
    * **How to Check:** Run the script with a few different EEG samples from your test set. The script should successfully produce textual output without errors. The quality of the text will vary, but the primary check is that the end-to-end generation pipeline works. For example, for an EEG of a "car driving", an output like "a vehicle is moving" would be a huge success at this stage.

---

### **Phase 6: Full Evaluation and Benchmarking**

The final phase is to systematically measure the quality of the generated text against the ground truth.

* **Goal:** Establish the SOTA benchmark by evaluating the generated descriptions using a suite of NLP metrics.
* **Tasks:**
    1.  **Generate Predictions:** Loop through your entire test set and use the inference script from Phase 5 to generate a description for every EEG sample. Save these predictions to a file.
    2.  **Implement Metrics:**
        * Use a library like `huggingface/evaluate` to easily compute metrics.
        * Calculate n-gram overlap metrics: BLEU, ROUGE, METEOR.
        * Calculate semantic metrics: CIDEr, SPICE, and importantly, BERTScore, which is excellent for semantic similarity.
    3.  **Report Results:** Create a script that loads the predicted captions and the reference captions and prints a clean table of all the metric scores.

* **Output/Verification:**
    * **Deliverable:** An `evaluate.py` script and a results file (e.g., `results.json` or `results.txt`).
    * **How to Check:** The script runs successfully and produces numerical scores for all metrics. These scores are your final benchmark results, which you can compare against other models or report in your research.
