- 1) STGT Encoder for embeddings
    
    ### Phase 1: Implement the Graph Convolutional Network (GCN) Layer
    
    The goal of this phase is to build the **spatial filtering component** of your encoder. You'll create a single, reusable layer that can be stacked to process the raw EEG signal at each time step.
    
    - **1.1: Define the GCN Layer Class**
        - **Action:** Create a new Python class, for example, `GCNLayer`, that inherits from `torch.nn.Module`.
        - **Why:** This makes your layer a modular component that can be easily integrated into a larger network and trained by PyTorch's automatic differentiation.
    - **1.2: Define Learnable Weights (W)**
        - **Action:** Inside your `GCNLayer`'s `__init__` method, define the learnable weight matrix W. The simplest way to do this in PyTorch is to use `torch.nn.Linear`. The size of this linear layer will be `(input_features, output_features)`. You will need to decide on the number of output features, which is a hyperparameter (e.g., 64, 128, etc.).
        - **Why:** This `nn.Linear` layer contains the W matrix that the network will learn during training. It transforms the raw signal into a new, more abstract feature space.
    - **1.3: Implement the Forward Pass**
        - **Action:** Implement the `forward` method for your `GCNLayer` class. This method will take two inputs: the EEG features (X) and your pre-computed adjacency matrix (A^). The core operation will be `torch.mm(A_hat, X)` followed by `self.linear` (which applies the W matrix). Finally, apply a non-linear activation function, like `torch.relu`.
        - **Why:** This `forward` method executes the GCN formula, combining the spatial information from A^ with the data from X and transforming it using the learnable weights W.
    - **Expected Outcome:** You will have a working `GCNLayer` class. You should be able to instantiate it and pass a dummy `(62, 1)` tensor and your `(62, 62)` adjacency matrix to it, receiving a new tensor of shape `(62, output_features)`.
    
    ---
    
    ### Phase 2: Implement the Temporal Transformer Component
    
    The goal of this phase is to build the **temporal processing component** of your encoder, which will model the sequence of spatially-aware features from the GCN.
    
    - **2.1: Define Transformer Configuration**
        - **Action:** Decide on the key hyperparameters for your Transformer Encoder: `d_model` (the feature dimension from the GCN), `nhead` (number of attention heads), `num_layers` (number of Transformer blocks), and `dim_feedforward`. These values will be passed to PyTorch's built-in Transformer classes.
        - **Why:** These parameters control the capacity and complexity of your Transformer, dictating how it processes the temporal sequence.
    - **2.2: Instantiate the Transformer**
        - **Action:** Create a new class, for example, `TemporalTransformer`, that holds an instance of `torch.nn.TransformerEncoderLayer` and `torch.nn.TransformerEncoder`. You'll use these to build your component.
        - **Why:** These classes provide a highly optimized and standard way to implement a Transformer, saving you from writing all the complex self-attention logic from scratch.
    - **2.3: Implement the Forward Pass**
        - **Action:** The `forward` method for this class will take a sequence of features as input, shaped `(n_time_steps, batch_size, d_model)`. It will then pass this sequence through the `TransformerEncoder`.
        - **Why:** This method performs the self-attention on the sequence of GCN outputs, learning which temporal patterns are most important for the final embedding.
    - **Expected Outcome:** A working `TemporalTransformer` class that can take a sequence of feature vectors and produce an output sequence where each vector has been updated with information from all other time steps.
    
    ---
    
    ### Phase 3: Assemble the Full STGT Encoder
    
    This is where you combine the components from Phases 1 and 2 into a single, comprehensive module. This final module will be the main feature extractor for your entire project.
    
    - **3.1: Define the Main Encoder Class**
        - **Action:** Create a class named `STGTEncoder` that contains instances of your `GCNLayer` and `TemporalTransformer` classes. You'll likely need to stack a few GCN layers to get a deeper spatial model.
        - **Why:** This top-level class orchestrates the entire process, making the model easy to train and use.
    - **3.2: Implement the Full Forward Pass**
        - **Action:** This is the most complex part. The `forward` method will take the raw EEG data (`(batch_size, n_channels, n_time_steps)`) and the static `A_hat`.
        1. **Process Spatially:** You will need to write a loop that iterates through each of the `n_time_steps`. In each loop iteration, extract the `(batch_size, n_channels, 1)` data for that time step and pass it through your GCN layers.
        2. **Collect Sequence:** Store the output from each GCN pass into a list.
        3. **Reshape for Transformer:** Concatenate the list of GCN outputs to form a single tensor of shape `(n_time_steps, batch_size, feature_dimension)`. This is the standard input shape for a Transformer.
        4. **Process Temporally:** Pass this sequence into your `TemporalTransformer`.
        5. **Create Final Embedding:** Take the output of the Transformer and perform an aggregation step (e.g., global average pooling across the time steps) to get a single, final embedding vector of shape `(batch_size, final_embedding_size)`.
        - **Why:** This `forward` method is the complete pipeline, first applying spatial modeling and then temporal modeling to produce a single, unified embedding that represents the entire EEG epoch.
    - **Expected Outcome:** A fully functional `STGTEncoder` module. You should be able to feed it a batch of raw EEG data and receive a single embedding for each example in the batch. This final embedding is what will be used for both your contrastive loss and your MTR head.
    
    ## Phase 1 in detail:
    
    ### Phase 1: Implement the GCN Layer
    
    ### **1.1: Define the GCN Layer Class**
    
    This is about setting up the basic structure of your GCN module in Python.
    
    - **1.1.1: Create a New File**
        - **Action:** Create a new Python file named `gcn_layer.py`.
        - **Why:** This keeps your code organized and modular.
        - **Expected Outcome:** You have an empty `.py` file ready for your code.
    - **1.1.2: Import Necessary Libraries**
        - **Action:** At the top of your file, add the necessary imports: `import torch` and `import torch.nn as nn`.
        - **Why:** These are the fundamental libraries you'll need for building any neural network module in PyTorch.
        - **Expected Outcome:** Your file has the required import statements.
    - **1.1.3: Define the Class Structure**
        - **Action:** Define a new class `GCNLayer(nn.Module):`
        - **Why:** Inheriting from `nn.Module` is the standard way to create a PyTorch layer. It gives your class access to crucial functionalities like tracking learnable parameters and the `forward` method.
        - **Expected Outcome:** You have a basic class definition ready to be filled with code.
    
    ### **1.2: Define Learnable Weights (W)**
    
    This is where you'll add the components that the model will learn during training.
    
    - **1.2.1: Implement the Constructor**
        - **Action:** Define the `__init__` method for your `GCNLayer` class. This method will take `in_features` and `out_features` as arguments.
        - **Why:** The constructor is where you define all the sub-modules and parameters that your layer will use.
        - **Expected Outcome:** Your class has an `__init__` method that accepts the input and output feature dimensions.
    - **1.2.2: Instantiate the Linear Layer**
        - **Action:** Inside `__init__`, create an instance of `nn.Linear` and assign it to a class attribute, like `self.linear`. The layer's input size will be `in_features` and its output size will be `out_features`. For a GCN, you should also set `bias=False`.
        - **Why:** This `nn.Linear` object effectively creates and manages your learnable weight matrix W. It also handles the matrix multiplication (`X @ W`) for you.
        - **Expected Outcome:** Your `GCNLayer` object has a `self.linear` attribute which is an `nn.Linear` layer.
    
    ### **1.3: Implement the Forward Pass**
    
    This is the most critical part, where the GCN operation is actually performed.
    
    - **1.3.1: Define the Forward Method Signature**
        - **Action:** Define the `forward` method for your class. It should accept `x` (your input features, a tensor of shape `(n_channels, in_features)`) and `adj_matrix` (your A^ matrix, a tensor of shape `(n_channels, n_channels)`).
        - **Why:** This method defines the forward propagation logic of your layer.
        - **Expected Outcome:** You have a `forward` method signature ready to be implemented.
    - **1.3.2: Perform the Matrix Multiplication**
        - **Action:** The first step inside `forward` is to perform the `$\hat{A}X$` operation. You can do this using `torch.mm(adj_matrix, x)`. Assign the result to a new variable.
        - **Why:** This step aggregates the features from neighboring nodes (channels) according to the connectivity defined by your adjacency matrix.
        - **Expected Outcome:** You have a variable that holds the spatially aggregated features.
    - **1.3.3: Apply the Linear Transformation**
        - **Action:** Pass the result from the previous step through your linear layer: `self.linear(...)`.
        - **Why:** This step applies the learnable weight matrix W to the aggregated features, transforming them into the new feature space.
        - **Expected Outcome:** You have a variable that holds the transformed, aggregated features.
    - **1.3.4: Apply the Activation Function**
        - **Action:** Wrap the result from the previous step in a non-linear activation function, like `torch.relu(...)`.
        - **Why:** Non-linear activation functions are essential for neural networks to learn complex patterns and prevent the network from collapsing into a simple linear model.
        - **Expected Outcome:** The final output of the `forward` method is the new feature matrix, ready to be passed to the next layer.
    - **1.3.5: Test and Verify**
        - **Action:** Outside of your class definition, write a small script to instantiate your `GCNLayer`, create a dummy input tensor and your `A_hat` matrix, and call the `forward` method. Print the shape of the output tensor to confirm it matches `(62, out_features)`.
        - **Why:** This is a crucial sanity check to ensure your layer is working as expected before you build the rest of the network.
        - **Expected Outcome:** You have confirmed your `GCNLayer` is correctly processing input and producing the expected output shape.
        
        ## Phase 2 in detail
        
        ### What we’re doing in Phase 2: Temporal Transformer
        
        - You already have a **GCN (Graph Convolution Network)** that looks at the spatial relationships between EEG channels at each moment in time.
        - But brain activity is not just about “where” things happen, it’s also about **how things change over time**.
        - That’s why we need the **Temporal Transformer** → it looks at **patterns across time steps** (like 400 time slices of EEG per trial).
        
        Think of it like this:
        
        👉 GCN = “taking snapshots of brain activity at each moment.”
        
        👉 Transformer = “watching the movie made from those snapshots, and figuring out how scenes connect.”
        
        ---
        
        ### Step 2.1: Define Transformer Hyperparameters
        
        Hyperparameters are just **knobs/settings** you set before training.
        
        1. **d_model** → the size of the feature vector for each time step.
            - After GCN, you have `(batch_size, n_time_steps, n_channels, out_features)`.
            - You need to squish `(n_channels * out_features)` into one vector per time step.
            - Example: if you have 62 channels and each has 64 features, then one time step = `62*64 = 3968` numbers. So `d_model=3968`.
        2. **nhead** → the number of "attention heads."
            - Think of attention heads as different “perspectives” for looking at relationships in time.
            - Must divide `d_model` evenly. Example: if `d_model=3968`, you could choose 8 heads (3968 ÷ 8 = 496).
        3. **num_encoder_layers** → how many Transformer layers you stack.
            - More layers = smarter model but slower. Start with 2–3.
        4. **dim_feedforward** → how big the hidden layer is inside each Transformer block.
            - Usually 2x or 4x bigger than d_model. Example: if d_model=3968, then 4x = ~16k.
        5. **dropout** → a trick to avoid overfitting (randomly “ignores” some neurons while training).
            - Usually 0.1.
        
        ✅ So these numbers are just **design decisions** that control how powerful your Transformer will be.
        
        ---
        
        ### Step 2.2: Build the Transformer in PyTorch
        
        Instead of coding the Transformer from scratch, PyTorch gives us **ready-made blocks**:
        
        - `TransformerEncoderLayer` = one Transformer block (self-attention + feedforward).
        - `TransformerEncoder` = stack multiple blocks.
        
        So your `TemporalTransformer` class is just “wrapping” these PyTorch building blocks into one neat module.
        
        ---
        
        ### Step 2.3: Forward Pass (How data flows inside)
        
        When data goes into the Transformer, the shape matters a lot:
        
        - PyTorch expects input as:
            
            **(sequence_length, batch_size, d_model)**
            
            → sequence_length = number of time steps (400 for EEG)
            
            → batch_size = number of samples in one batch
            
            → d_model = feature size per time step (e.g., 3968)
            
        - But your GCN gives:
            
            **(batch_size, n_time_steps, n_channels, out_features)**
            
        
        So we must:
        
        1. Flatten `(n_channels * out_features)` → one long vector.
            
            → shape becomes `(batch_size, n_time_steps, d_model)`.
            
        2. Rearrange dimensions (`permute`) so it becomes `(n_time_steps, batch_size, d_model)`.
        3. Pass through Transformer → output has same shape.
        
        ---
        
        ### Step 2.4: Test it with Dummy Data
        
        Before plugging into your full pipeline, you “dry-run” it:
        
        1. Pretend you have 400 time steps, batch size of 4, d_model=3968.
        2. Make a random tensor with that shape.
        3. Pass it into your Transformer.
        4. Check if the output shape is correct.
        
        If input shape = `(400, 4, 3968)`, output shape should also be `(400, 4, 3968)`.
        
        That means Transformer is working fine.
        
        ---
        
        ✅ **End Result of Phase 2:**
        
        You now have a module that can take EEG features across time and let the Transformer figure out which time steps influence each other (kind of like seeing “cause-effect patterns” in the brain movie).
        
        ## Phase 3 in detail
        
        ## Phase 3: Assemble the Full STGT Encoder
        
        The **STGTEncoder** is the heart of your EEG processing pipeline.
        
        Its job: **turn raw EEG (62 channels × 400 time steps) into a compact spatio-temporal embedding**.
        
        It does this in **two stages**:
        
        1. **GCN Layer** → extracts **spatial dependencies** (relations across EEG channels).
        2. **Transformer** → extracts **temporal dependencies** (how patterns evolve over time).
        
        Finally, it condenses the whole sequence into a **single embedding vector**.
        
        ---
        
        ### 🔹 Step 3.1: Define the STGTEncoder class
        
        - **Action**: Create `stgt_encoder.py` and define a `STGTEncoder` class.
        - **Why**: This becomes a *self-contained EEG feature extractor* you can plug into bigger models like STAR-Net.
        
        Key points inside `__init__`:
        
        - `GCNLayer`: learns spatial brain connectivity.
        - `TemporalTransformer`: learns sequence dynamics.
        - `transformer_d_model`: must equal `(in_channels × gcn_out_features)` because the GCN outputs `(channels × features)` that get flattened for the transformer.
        
        ---
        
        ### 🔹 Step 3.2: Implement the `forward` method
        
        This is the **data flow** definition.
        
        ### Input:
        
        - `eeg_batch_data`: `(batch_size, in_channels, num_time_steps)` → e.g. `(32, 62, 400)`
        - `adj_matrix`: `(in_channels, in_channels)` → e.g. `(62, 62)`
        
        ### Workflow:
        
        1. **Loop through time**
            - Extract EEG at time `t`: `(batch_size, in_channels)`
            - Reshape to `(batch_size, in_channels, 1)` → because GCN expects input features.
            - Run through `GCNLayer` → output `(batch_size, in_channels, gcn_out_features)`
        2. **Stack GCN outputs** across all time steps
            - Result: `(batch_size, num_time_steps, in_channels, gcn_out_features)`
        3. **Flatten for Transformer**
            - Merge `(in_channels × gcn_out_features)` → `(transformer_d_model)`
            - Shape: `(batch_size, num_time_steps, transformer_d_model)`
        4. **Permute for Transformer input**
            - Transformers in PyTorch expect `(seq_len, batch_size, d_model)`
            - So → `(num_time_steps, batch_size, transformer_d_model)`
        5. **Feed into Transformer**
            - Output is the same shape `(num_time_steps, batch_size, transformer_d_model)`
        6. **Aggregate into final embedding**
            - Mean across `num_time_steps`
            - Final shape: `(batch_size, transformer_d_model)`
        
        ---
        
        ### 🔹 Step 3.3: Test the Encoder
        
        - Load EEG data + adjacency matrix.
        - Instantiate `STGTEncoder` with correct hyperparameters.
        - Select a dummy batch.
        - Run forward pass.
        - Verify **output shape = (batch_size, transformer_d_model)**.
        
        If you see the correct shape, it means **the whole spatial + temporal pipeline is wired correctly**.
        
        ---
        
        ✅ **Expected outcome**:
        
        When you run:
        
        ```bash
        python stgt_encoder.py
        
        ```
        
        You’ll see:
        
        - EEG + adjacency matrix loaded.
        - Forward pass runs without error.
        - Final embedding shape `(batch_size, transformer_d_model)` is printed.
        
        This confirms that your **STGTEncoder is ready** as a reusable EEG feature extractor
