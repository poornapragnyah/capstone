# ==================================================================================
# STANDALONE CPU INFERENCE SCRIPT
#
# This script is designed to be run in parallel with the training script
# to check the progress of the latest saved model.
#
# IT IS HARD-CODED TO RUN ON THE CPU to avoid interfering with training.
# ==================================================================================

import torch
import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from statsmodels.tsa.stattools import grangercausalitytests
from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops
from scipy.sparse import coo_matrix
import time
import math
import random
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import json
import evaluate as hf_evaluate  # Use alias to avoid conflict
from tqdm.auto import tqdm

# ==================================================================================
# ALL PATHS AND CONSTANTS (Copied from your script)
# ==================================================================================
H5_FILE_PATH = "/home/poorna/data/eeg_dataset_with_qwen.h5"
LOCAL_MODEL_PATH = "/home/poorna/models/bert-base-uncased"
TRAIN_PCT, VAL_PCT = 0.8, 0.1
BATCH_SIZE = 16 # Batch size doesn't matter much here, but good for consistency

NUM_COLORS = 12
NUM_OBJECTS = 90

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
PAD_ID = tokenizer.pad_token_id
SOS_ID = tokenizer.cls_token_id
EOS_ID = tokenizer.sep_token_id
TEXT_VOCAB_SIZE = tokenizer.vocab_size

# --- CRITICAL: FORCE CPU ---
device = torch.device("cpu")
print(f"Using device: {device} (This is intentional for safe inference)")
print(f"Metadata config: {NUM_COLORS} colors, {NUM_OBJECTS} objects (NO categories)")

# ==================================================================================
# GRANGER CAUSALITY (Unchanged)
# ==================================================================================
def create_granger_causality_matrix(eeg_batch):
    eeg_sample = eeg_batch[0].cpu().numpy().T
    num_channels = eeg_sample.shape[1]
    causality_matrix = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                continue
            ts_i = eeg_sample[:, i]
            ts_j = eeg_sample[:, j]
            min_len = 20
            if len(ts_i) < min_len or len(ts_j) < min_len:
                causality_matrix[i, j] = 0.0
                continue
            data = np.vstack([ts_j, ts_i]).T
            try:
                current_maxlag = min(5, len(data)//2 - 2)
                if current_maxlag < 1:
                    causality_matrix[i, j] = 0.0
                    continue
                results = grangercausalitytests(data, maxlag=current_maxlag, verbose=False)
                p_value = results[current_maxlag][0]['ssr_ftest'][1]
                if p_value < 0.05:
                    causality_matrix[i, j] = 1.0
            except Exception as e:
                causality_matrix[i, j] = 0.0

    adj_matrix = coo_matrix(causality_matrix)
    edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix)

    if edge_attr is None:
        edge_attr = torch.tensor([], dtype=torch.float)
    elif edge_attr.ndim == 0:
        edge_attr = edge_attr.unsqueeze(0)

    return edge_index.to(torch.long), edge_attr.to(torch.float)

# ==================================================================================
# DATASET AND DATALOADER (Unchanged)
# ==================================================================================
class EEGMetaTextH5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = None
        with h5py.File(self.h5_path, 'r') as f:
            self.n_samples = f['eeg'].shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        eeg = torch.from_numpy(self.h5_file['eeg'][idx].astype(np.float32))
        meta = torch.from_numpy(self.h5_file['metadata'][idx].astype(np.float32))
        text = torch.from_numpy(self.h5_file['input_ids'][idx].astype(np.int64))

        return eeg, meta, text

def collate_multimodal_batch(batch):
    eeg_list, meta_list, text_list = [], [], []
    for eeg, meta, txt in batch:
        eeg_list.append(eeg)
        meta_list.append(meta)
        text_list.append(txt)

    eeg_batch = torch.stack(eeg_list, dim=0)
    meta_batch = torch.stack(meta_list, dim=0)
    text_padded = pad_sequence(text_list, batch_first=True, padding_value=PAD_ID)

    return eeg_batch.float(), meta_batch.float(), text_padded

# ==================================================================================
# ALL MODEL CLASSES (Needed for torch.load)
# ==================================================================================

class SpatioTemporalEEGEncoder(nn.Module):
    def __init__(self, num_channels=62, enc_hidden=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_channels = num_channels
        self.gcn1 = GCNConv(num_channels, enc_hidden)
        self.gcn2 = GCNConv(enc_hidden, enc_hidden)
        self.rnn = nn.GRU(enc_hidden, enc_hidden, num_layers,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        print(f"Encoder RNN input size: {enc_hidden}")

    def forward(self, eeg, edge_index, edge_attr):
        batch_size = eeg.shape[0]
        num_timesteps = eeg.shape[2]

        batch_edge_index = edge_index.repeat(1, batch_size)
        batch_edge_attr = edge_attr.repeat(batch_size)
        batch_offset = torch.arange(batch_size, device=eeg.device) * self.num_channels
        batch_edge_index = batch_edge_index + batch_offset.repeat_interleave(edge_index.shape[1])

        eeg_reshaped = eeg.permute(0, 2, 1).reshape(-1, self.num_channels)

        x = F.relu(self.gcn1(eeg_reshaped, batch_edge_index, batch_edge_attr))
        x = self.dropout(x)
        x = F.relu(self.gcn2(x, batch_edge_index, batch_edge_attr))

        temporal_features = x.reshape(batch_size, num_timesteps, -1)
        encoder_outputs, encoder_hidden = self.rnn(temporal_features)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        return encoder_outputs, encoder_hidden

class LuongAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.attn = nn.Linear(enc_dim, dec_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        attn_energies = self.attn(encoder_outputs)
        scores = torch.bmm(decoder_hidden.permute(1, 0, 2), attn_energies.permute(1, 2, 0))
        attn_weights = F.softmax(scores, dim=2)
        context = torch.bmm(attn_weights, encoder_outputs.permute(1, 0, 2))
        return context, attn_weights.squeeze(1)

class MetadataEncoder(nn.Module):
    def __init__(self, num_colors, num_objects, 
                 color_emb_dim=16, object_feature_dim=128):
        super().__init__()
        
        self.color_embedding = nn.Embedding(num_colors, color_emb_dim)
        
        self.object_processor = nn.Sequential(
            nn.Linear(num_objects, 256), # Changed from 61 to num_objects (90)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, object_feature_dim)
        )
        self.output_dim = color_emb_dim + object_feature_dim
        print(f"MetadataEncoder output dimension: {self.output_dim} (color:{color_emb_dim} + objects:{object_feature_dim})")

    def forward(self, metadata):
        color_ids = metadata[:, 0].long()
        object_features_raw = metadata[:, 1:] # Changed from [:, 2:] to [:, 1:]
        object_features_raw = object_features_raw.float()

        color_vec = self.color_embedding(color_ids)
        object_vec = self.object_processor(object_features_raw)

        combined_features = torch.cat([color_vec, object_vec], dim=1)
        
        return combined_features

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden, dec_hidden, meta_features_dim, num_layers, pad_id, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.dec_hidden = dec_hidden
        self.num_layers = num_layers
        enc_dim = enc_hidden * 2

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attention = LuongAttention(enc_dim, dec_hidden)

        self.rnn_input_dim = emb_dim + enc_dim + meta_features_dim + enc_dim
        print(f"Decoder RNN input dimension: {self.rnn_input_dim}")
        self.rnn = nn.GRU(self.rnn_input_dim, dec_hidden, num_layers, dropout=dropout if num_layers > 1 else 0)

        self.fc_out = nn.Linear(dec_hidden, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.bridge = nn.Linear(enc_dim, dec_hidden)

    def init_hidden(self, encoder_hidden):
        hidden = encoder_hidden.view(self.num_layers, 2, encoder_hidden.size(1), -1)
        last_layer_hidden = hidden[-1]
        encoder_hidden_cat = torch.cat((last_layer_hidden[0], last_layer_hidden[1]), dim=1)
        bridged_hidden = torch.tanh(self.bridge(encoder_hidden_cat))
        decoder_initial_hidden = bridged_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        return decoder_initial_hidden

    def forward(self, token, decoder_hidden, encoder_outputs, meta_features, global_eeg_context):
        token = token.unsqueeze(0)
        embedded = self.dropout(self.embedding(token))
        context, attn_weights = self.attention(decoder_hidden[-1].unsqueeze(0), encoder_outputs)
        
        meta_features_unsqueezed = meta_features.unsqueeze(0)
        global_eeg_context_unsqueezed = global_eeg_context.unsqueeze(0)
        context_permuted = context.permute(1, 0, 2)

        rnn_input = torch.cat((
            embedded,
            context_permuted,
            meta_features_unsqueezed,
            global_eeg_context_unsqueezed
        ), dim=2)

        output, hidden = self.rnn(rnn_input, decoder_hidden)
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, context.squeeze(1)

# This must be the *latest* version of the Seq2Seq class
class Seq2Seq(nn.Module):
    def __init__(self, text_vocab_size, num_colors, num_objects, enc_hidden=256, dec_hidden=256,
                 pad_id=0, dropout=0.2, color_emb_dim=16, object_feature_dim=128, emb_dim=256, dec_layers=2):
        super().__init__()
        self.encoder = SpatioTemporalEEGEncoder(enc_hidden=enc_hidden, dropout=dropout, num_layers=dec_layers)
        
        self.meta_encoder = MetadataEncoder(
            num_colors, 
            num_objects,
            color_emb_dim, 
            object_feature_dim
        )

        meta_features_dim = self.meta_encoder.output_dim
        enc_dim = enc_hidden * 2

        self.decoder = Decoder(text_vocab_size, emb_dim, enc_hidden, dec_hidden,
                                 meta_features_dim, dec_layers, pad_id, dropout)

        self.meta_head = nn.Sequential(
            nn.Linear(enc_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_colors + num_objects)
        )
        self.num_colors = num_colors
        self.num_objects = num_objects

    def forward(self, eeg, metadata, target_text, edge_index, edge_attr, 
                text_teacher_forcing_ratio=0.5, meta_teacher_forcing_ratio=1.0):
        batch_size = eeg.shape[0]
        target_len = target_text.shape[1]
        target_vocab_size = self.decoder.vocab_size

        encoder_outputs, encoder_hidden = self.encoder(eeg, edge_index, edge_attr)
        decoder_hidden = self.decoder.init_hidden(encoder_hidden)

        hidden_reshaped = encoder_hidden.view(self.encoder.rnn.num_layers, 2, batch_size, -1)
        last_layer_hidden = hidden_reshaped[-1]
        global_eeg_context = torch.cat((last_layer_hidden[0], last_layer_hidden[1]), dim=1)

        meta_preds = self.meta_head(global_eeg_context)
        pred_color = meta_preds[:, :self.num_colors]
        pred_object = meta_preds[:, self.num_colors:]

        use_true_meta = random.random() < meta_teacher_forcing_ratio
        
        if use_true_meta:
            meta_features = self.meta_encoder(metadata)
        else:
            with torch.no_grad():
                pred_color_id_vec = pred_color.argmax(dim=-1).float().unsqueeze(1)
                pred_object_vec = (torch.sigmoid(pred_object) > 0.5).float()
                predicted_meta_vector = torch.cat([
                    pred_color_id_vec,
                    pred_object_vec
                ], dim=1)
            
            meta_features = self.meta_encoder(predicted_meta_vector)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(eeg.device)
        decoder_input = target_text[:, 0]

        for t in range(1, target_len):
            output, decoder_hidden, _ = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs,
                meta_features,
                global_eeg_context
            )

            outputs[t] = output
            teacher_force = random.random() < text_teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target_text[:, t] if teacher_force else top1

        return outputs[1:].permute(1, 0, 2), pred_color, pred_object

# ==================================================================================
# MAIN INFERENCE EXECUTION
# ==================================================================================
if __name__ == "__main__":
    # Create dataset and loaders
    dataset = EEGMetaTextH5Dataset(H5_FILE_PATH)
    N = len(dataset)
    n_train = int(N * TRAIN_PCT)
    n_val = int(N * VAL_PCT)
    n_test = N - n_train - n_val
    g = torch.Generator().manual_seed(42) # Use same seed!
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=g)

    # We need the test_loader to get samples
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_multimodal_batch)

    # Create Granger matrix (using a sample from test set)
    print("Creating Granger Causality matrix...")
    try:
        # Get a sample batch from the test loader
        eeg_b, _, _ = next(iter(test_loader))
        granger_edge_index, granger_edge_attr = create_granger_causality_matrix(eeg_b)
        num_channels = eeg_b.shape[1]

        granger_edge_index, granger_edge_attr = add_self_loops(
            granger_edge_index,
            edge_attr=granger_edge_attr,
            num_nodes=num_channels,
            fill_value=1.0
        )

        if granger_edge_attr is None:
            granger_edge_attr = torch.ones(granger_edge_index.shape[1], dtype=torch.float)

        # --- FORCE GRANGER TO CPU ---
        granger_edge_index = granger_edge_index.to(torch.long).to(device)
        granger_edge_attr = granger_edge_attr.to(torch.float32).to(device)
        print(f"Granger matrix created: {granger_edge_index.shape}")

    except Exception as e:
        print(f"Error creating Granger matrix: {e}. Using fallback.")
        num_channels = 62
        edge_index = torch.combinations(torch.arange(num_channels), r=2).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_channels)
        granger_edge_index = edge_index.to(torch.long).to(device)
        granger_edge_attr = torch.ones(granger_edge_index.shape[1], dtype=torch.float32).to(device)

    # Instantiate model
    model = Seq2Seq(
        text_vocab_size=TEXT_VOCAB_SIZE,
        num_colors=NUM_COLORS,
        num_objects=NUM_OBJECTS,
        pad_id=PAD_ID,
        dropout=0.2,
        enc_hidden=256,
        dec_hidden=256,
        emb_dim=256,
        dec_layers=2
    ).to(device) # Send model to CPU

    print(f"Model instantiated on '{device}'.")

    # ==================================================================================
    # REFACTORED INFERENCE (Copied from your script)
    # ==================================================================================
    
    # Load best model
    checkpoint_path = 'eeg-meta-text-qwen-refactored-model.pt'
    try:
        # --- FORCE MODEL TO LOAD ONTO CPU ---
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"\nBest model '{checkpoint_path}' loaded for inference.")
    except FileNotFoundError:
        print(f"Error: Model file '{checkpoint_path}' not found.")
        print("Please wait for the training script to save a checkpoint.")
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()


    # Load object mapping
    OBJECT_MAPPING_FILE = "/home/poorna/data/object_id_to_name_qwen.json"
    try:
        with open(OBJECT_MAPPING_FILE, 'r') as f:
            object_mapping = json.load(f)
        print(f"Object mapping loaded: {len(object_mapping)} objects")
    except FileNotFoundError:
        print(f"Warning: '{OBJECT_MAPPING_FILE}' not found.")
        object_mapping = {}

    @torch.no_grad()
    def generate_end_to_end(model, eeg_signal, edge_index, edge_attr,
                            sample_idx,
                            k=5,
                            penalty_alpha=0.3,
                            context_beta=0.7,
                            max_len=100):
        
        model.eval()
        eeg_signal = eeg_signal.unsqueeze(0).to(device) # Send data to CPU

        # 1. Encode EEG
        encoder_outputs, encoder_hidden = model.encoder(eeg_signal, edge_index, edge_attr)
        
        # 2. Get global EEG context
        hidden_reshaped = encoder_hidden.view(model.encoder.rnn.num_layers, 2, 1, -1)
        last_layer_hidden = hidden_reshaped[-1]
        global_eeg_context = torch.cat((last_layer_hidden[0], last_layer_hidden[1]), dim=1)

        # 3. Predict metadata from EEG
        meta_preds_logits = model.meta_head(global_eeg_context)

        pred_color_logits = meta_preds_logits[:, :model.num_colors]
        pred_object_logits = meta_preds_logits[:, model.num_colors:]

        # For decoder input
        pred_color_id_vec = pred_color_logits.argmax(dim=-1).float().unsqueeze(1)
        pred_object_vec = (torch.sigmoid(pred_object_logits) > 0.5).float()

        predicted_meta_vector = torch.cat([
            pred_color_id_vec,
            pred_object_vec
        ], dim=1)
        
        # For printing
        pred_color_for_print = pred_color_id_vec.item()
        
        object_probs = torch.sigmoid(pred_object_logits)
        pred_object_ids_list = (object_probs > 0.5).nonzero(as_tuple=True)[1].tolist()
        
        # 5. Encode the predicted metadata
        predicted_meta_features = model.meta_encoder(predicted_meta_vector)

        # 6. Initialize decoder
        decoder_hidden = model.decoder.init_hidden(encoder_hidden)

        if sample_idx < 2:
            print(f"\n--- [Sample {sample_idx+1}] Generation Start (End-to-End) ---")

        # 7. Generation loop
        generated_ids = torch.tensor([SOS_ID], device=device)
        for step in range(max_len):
            input_token = generated_ids[-1].unsqueeze(0)

            prediction, new_hidden, attention_context = model.decoder(
                input_token,
                decoder_hidden,
                encoder_outputs,
                predicted_meta_features,
                global_eeg_context
            )
            decoder_hidden = new_hidden
            
            model_log_probs = F.log_softmax(prediction, dim=-1).squeeze(0)
            topk_model_log_probs, topk_ids = torch.topk(model_log_probs, k)
            
            current_seq_len = generated_ids.shape[0]
            prev_token_embeddings = F.normalize(model.decoder.embedding(generated_ids), dim=-1)
            candidate_token_embeddings = F.normalize(model.decoder.embedding(topk_ids), dim=-1)
            
            sim_matrix = torch.matmul(candidate_token_embeddings, prev_token_embeddings.t())
            degeneration_penalty = torch.zeros(k, device=device)
            if current_seq_len > 1:
                degeneration_penalty, _ = torch.max(sim_matrix, dim=-1)
                
            current_decoder_state = F.normalize(decoder_hidden[-1].squeeze(), dim=-1)
            context_agreement_score = torch.matmul(candidate_token_embeddings, current_decoder_state)
            
            final_score = topk_model_log_probs + context_beta * context_agreement_score - penalty_alpha * degeneration_penalty
            
            best_next_token_idx = torch.argmax(final_score)
            next_token_id = topk_ids[best_next_token_idx]

            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)])
            if next_token_id.item() == EOS_ID:
                if sample_idx < 5:
                    print("  [EOS Reached]")
                break
                
        if generated_ids.numel() > 1:
            predicted_text_ids = generated_ids[1:-1] if generated_ids[-1].item() == EOS_ID else generated_ids[1:]
            predicted_text = tokenizer.decode(predicted_text_ids.tolist(), skip_special_tokens=True)
        else:
            predicted_text = ""
        
        if sample_idx < 5:
            print(f"--- [Sample {sample_idx+1}] Generation End ---")

        return predicted_text, pred_color_for_print, pred_object_ids_list

    # Run inference
    NUM_SAMPLES = 20
    print(f"\n--- Running TRUE END-TO-END Inference on {NUM_SAMPLES} Samples ---")

    predictions = []
    references = []

    for i in range(NUM_SAMPLES):
        eeg_sample, meta_sample, true_text_ids = test_ds[i]

        # Extract true metadata
        true_color_id = int(meta_sample[0].item())
        true_object_vector = meta_sample[1:]
        true_object_ids_tensors = true_object_vector.nonzero(as_tuple=True)[0]
        true_object_names = [object_mapping.get(str(id_item), f"ID:{id_item}") 
                             for id_item in true_object_ids_tensors.tolist()]
        if not true_object_names:
            true_object_names = ["None"]

        predicted_text, pred_color, pred_object_ids = generate_end_to_end(
            model,
            eeg_sample,
            granger_edge_index,
            granger_edge_attr,
            sample_idx=i,
            k=5,
            penalty_alpha=0.3,
            context_beta=0.7
        )

        # Decode true text
        true_text_ids_list = true_text_ids.long().tolist()
        true_text = tokenizer.decode(true_text_ids_list, skip_special_tokens=True)

        predictions.append(predicted_text)
        references.append(true_text)

        # Convert predicted object IDs to names
        pred_object_names = [object_mapping.get(str(oid), f"ID:{oid}") 
                             for oid in pred_object_ids]
        if not pred_object_names:
            pred_object_names = ["None"]

        # Print summary
        print(f"\n--- Sample {i+1}/{NUM_SAMPLES} Summary (Index: {i}) ---")
        print(f"GROUND TRUTH TEXT: {true_text}")
        print(f"MODEL PREDICTION TEXT: {predicted_text}")
        print("\nMETADATA PREDICTION (FROM EEG):")
        print(f"  Color ID:      Truth={true_color_id}, Predicted={pred_color}")
        print(f"  Object(s):     Truth={', '.join(true_object_names)}, Predicted={', '.join(pred_object_names)}")
        print("-" * 50)

    # Evaluation metrics
    print("\n--- Evaluation Metrics (True End-to-End Model) ---")
    try:
        bleu_metric = hf_evaluate.load('bleu')
        bleu_results = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
        print(f"BLEU Score: {bleu_results['bleu']:.4f}")

        rouge_metric = hf_evaluate.load('rouge')
        rouge_results = rouge_metric.compute(predictions=predictions, references=references)
        print(f"ROUGE-1 Score: {rouge_results['rouge1']:.4f}")
        print(f"ROUGE-2 Score: {rouge_results['rouge2']:.4f}")
        print(f"ROUGE-L Score: {rouge_results['rougeL']:.4f}")
    except Exception as e:
        print(f"Could not calculate evaluation metrics: {e}")
