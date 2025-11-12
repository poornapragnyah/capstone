# ==================================================================================
# STANDALONE INFERENCE SCRIPT FOR TRANSFORMER (V5) — FULLY FIXED
# ==================================================================================

import torch
import torch.nn as nn
import h5py
import numpy as np
import copy
import math
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from statsmodels.tsa.stattools import grangercausalitytests
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import GCNConv
from scipy.sparse import coo_matrix
import torch.nn.functional as F
import json
import evaluate as hf_evaluate
import os

# ==================================================================================
# --- CONSTANTS & SETUP ---
# ==================================================================================
H5_FILE_PATH = "/home/poorna/data/eeg_dataset_with_qwen.h5"
LOCAL_MODEL_PATH = "/home/poorna/models/bert-base-uncased"
MODEL_SAVE_PATH = 'eeg-meta-text-transformer-v5-model.pt'
TRAIN_PCT, VAL_PCT = 0.8, 0.1
BATCH_SIZE = 16

NUM_COLORS = 12
NUM_OBJECTS = 90

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
PAD_ID = tokenizer.pad_token_id
SOS_ID = tokenizer.cls_token_id
EOS_ID = tokenizer.sep_token_id
TEXT_VOCAB_SIZE = tokenizer.vocab_size
D_MODEL = 256

device = torch.device("cpu")
print(f"Using device: {device} (Inference Mode)")

NUM_LAYERS = 4
NUM_HEADS = 8
D_FF = 1024
DROPOUT = 0.1

# ==================================================================================
# --- MODEL DEFINITIONS ---
# ==================================================================================

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=DROPOUT):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = get_clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        # mask: (L, L) -> (1, 1, L, L)
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(1)  # FIXED: Add batch and head dims
        batch_size = query.size(0)
        
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, d_meta):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.meta_gate = nn.Sequential(nn.Linear(d_meta, d_model), nn.Sigmoid())
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask, meta_features):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        gate = self.meta_gate(meta_features).unsqueeze(1)
        x = x * gate 
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x

class MetadataEncoder(nn.Module):
    def __init__(self, num_colors, num_objects, color_emb_dim=16, object_feature_dim=128):
        super().__init__()
        self.color_embedding = nn.Embedding(num_colors, color_emb_dim)
        self.object_processor = nn.Sequential(
            nn.Linear(num_objects, 256), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(256, object_feature_dim)
        )
        self.output_dim = color_emb_dim + object_feature_dim
        print(f"MetadataEncoder output dimension: {self.output_dim}")

    def forward(self, metadata):
        color_ids = metadata[:, 0].long()
        object_features_raw = metadata[:, 1:].float()
        color_vec = self.color_embedding(color_ids)
        object_vec = self.object_processor(object_features_raw)
        return torch.cat([color_vec, object_vec], dim=1)

class SpatioTemporalEEGEncoderTF(nn.Module):
    def __init__(self, num_channels=62, d_model=D_MODEL, num_layers=NUM_LAYERS,
                 num_heads=NUM_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.gcn1 = GCNConv(num_channels, d_model)
        self.gcn2 = GCNConv(d_model, d_model)
        self.spatial_dropout = nn.Dropout(dropout)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
        self.transformer_layers = get_clones(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        print("SpatioTemporalEEGEncoderTF initialized (Fixed spatial GCN mode).")

    def forward(self, eeg, edge_index, edge_attr):
        B, C, T = eeg.shape
        assert C == self.num_channels
        eeg = eeg.permute(0, 2, 1)  # (B, T, C)

        all_timestep_features = []
        for t in range(T):
            x_t = eeg[:, t, :]  # (B, C)
            x_t_out = []
            for b in range(B):
                # Create diagonal matrix: (C, C) from channel values
                diag = torch.diag(x_t[b])
                x = F.relu(self.gcn1(diag, edge_index, edge_attr))
                x = self.spatial_dropout(x)
                x = F.relu(self.gcn2(x, edge_index, edge_attr))
                x_t_out.append(x.mean(dim=0))  # (D,)
            x_t_out = torch.stack(x_t_out, dim=0)  # (B, D)
            all_timestep_features.append(x_t_out)

        x = torch.stack(all_timestep_features, dim=1)  # (B, T, D)
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.pos_encoding(x)

        for layer in self.transformer_layers:
            x = layer(x)

        return self.layer_norm(x)

class DecoderTF(nn.Module):
    def __init__(self, vocab_size, emb_dim, d_model, num_layers, num_heads, d_ff, pad_id, dropout, d_meta):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(emb_dim)
        self.input_projection = nn.Linear(emb_dim, d_model)
        decoder_layer = TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, d_meta)
        self.transformer_layers = get_clones(decoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        print("DecoderTF initialized.")

    def forward(self, target_text_ids, memory, memory_mask, meta_features):
        tgt_embed = self.embedding(target_text_ids)
        x = self.pos_encoding(tgt_embed)
        x = self.input_projection(x)

        tgt_seq_len = target_text_ids.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=x.device), diagonal=1).bool()

        for layer in self.transformer_layers:
            x = layer(x, memory, memory_mask, tgt_mask, meta_features)
        
        x = self.layer_norm(x)
        return self.fc_out(x)

class Seq2SeqTF(nn.Module):
    def __init__(self, text_vocab_size, num_colors, num_objects, d_model=D_MODEL, num_layers=NUM_LAYERS,
                 num_heads=NUM_HEADS, d_ff=D_FF, pad_id=PAD_ID, dropout=DROPOUT):
        super().__init__()
        self.encoder = SpatioTemporalEEGEncoderTF(d_model=d_model, num_layers=num_layers,
                                                  num_heads=num_heads, d_ff=d_ff, dropout=dropout)
        self.meta_encoder = MetadataEncoder(num_colors, num_objects)
        meta_features_dim = self.meta_encoder.output_dim
        self.decoder = DecoderTF(text_vocab_size, d_model, d_model, num_layers, num_heads, d_ff, pad_id, dropout, meta_features_dim)
        self.meta_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.LayerNorm(256), nn.Dropout(0.3),
            nn.Linear(256, num_colors + num_objects)
        )
        self.num_colors = num_colors
        self.num_objects = num_objects
        self.pad_id = pad_id
        print(f"Seq2SeqTF (V5) model initialized.")

    def forward(self, eeg, metadata, target_text, edge_index, edge_attr, meta_teacher_forcing_ratio=1.0):
        eeg_features = self.encoder(eeg, edge_index, edge_attr)
        cls_token_feature = eeg_features[:, 0, :]
        meta_preds = self.meta_head(cls_token_feature)
        use_true_meta = random.random() < meta_teacher_forcing_ratio
        if use_true_meta:
            meta_features = self.meta_encoder(metadata)
        else:
            with torch.no_grad():
                pred_color_id_vec = meta_preds[:, :self.num_colors].argmax(dim=-1).float().unsqueeze(1)
                pred_object_soft = torch.sigmoid(meta_preds[:, self.num_colors:])
                predicted_meta_vector = torch.cat([pred_color_id_vec, pred_object_soft], dim=1)
            meta_features = self.meta_encoder(predicted_meta_vector)
        decoder_memory = eeg_features[:, 1:, :]
        text_logits = self.decoder(target_text[:, :-1], decoder_memory, None, meta_features)
        return text_logits, meta_preds[:, :self.num_colors], meta_preds[:, self.num_colors:]

# ==================================================================================
# --- DATASET & GRANGER ---
# ==================================================================================

class EEGMetaTextH5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = None
        with h5py.File(self.h5_path, 'r') as f:
            self.n_samples = f['eeg'].shape[0]
    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        if self.h5_file is None: self.h5_file = h5py.File(self.h5_path, 'r')
        eeg = torch.from_numpy(self.h5_file['eeg'][idx].astype('float32'))
        meta = torch.from_numpy(self.h5_file['metadata'][idx].astype('float32'))
        text = torch.from_numpy(self.h5_file['input_ids'][idx].astype('int64'))
        return eeg, meta, text

def collate_multimodal_batch(batch):
    eeg_list, meta_list, text_list = [], [], []
    for eeg, meta, txt in batch:
        eeg_list.append(eeg); meta_list.append(meta); text_list.append(txt)
    return torch.stack(eeg_list), torch.stack(meta_list), pad_sequence(text_list, batch_first=True, padding_value=PAD_ID)

def create_granger_causality_matrix(eeg_batch):
    eeg_sample = eeg_batch[0].cpu().numpy().T
    num_channels = eeg_sample.shape[1]
    causality_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j: continue
            ts_i, ts_j = eeg_sample[:, i], eeg_sample[:, j]
            data = np.vstack([ts_j, ts_i]).T
            try:
                results = grangercausalitytests(data, maxlag=5, verbose=False)
                p_value = results[5][0]['ssr_ftest'][1]
                if p_value < 0.05: causality_matrix[i, j] = 1.0
            except: pass
    adj_matrix = coo_matrix(causality_matrix)
    edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix)
    if edge_attr is None: edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float)
    return edge_index.to(torch.long), edge_attr.to(torch.float)

# ==================================================================================
# --- AUTOREGRESSIVE INFERENCE ---
# ==================================================================================

@torch.no_grad()
def generate_transformer_greedy(model, eeg_signal, edge_index, edge_attr, sample_idx, max_len=100):
    model.eval()
    eeg_signal = eeg_signal.unsqueeze(0).to(device)
    eeg_features = model.encoder(eeg_signal, edge_index, edge_attr)
    cls_token_feature = eeg_features[:, 0, :]
    decoder_memory = eeg_features[:, 1:, :]

    meta_preds_logits = model.meta_head(cls_token_feature)
    pred_color_logits = meta_preds_logits[:, :model.num_colors]
    pred_object_logits = meta_preds_logits[:, model.num_colors:]
    pred_color_id_vec = pred_color_logits.argmax(dim=-1).float().unsqueeze(1)
    pred_object_soft = torch.sigmoid(pred_object_logits)
    predicted_meta_vector = torch.cat([pred_color_id_vec, pred_object_soft], dim=1)
    meta_features = model.meta_encoder(predicted_meta_vector)
    pred_color_for_print = pred_color_id_vec.item()
    pred_object_ids_list = (pred_object_soft > 0.5).nonzero(as_tuple=True)[1].tolist()

    generated_ids = torch.tensor([SOS_ID], device=device).unsqueeze(0)

    if sample_idx < 2:
        print(f"\n--- [Sample {sample_idx+1}] Generation Start (Transformer Greedy) ---")

    for _ in range(max_len):
        text_logits = model.decoder(generated_ids, decoder_memory, None, meta_features)
        next_token_id = text_logits[:, -1, :].argmax(1).item()
        generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
        if next_token_id == EOS_ID:
            if sample_idx < 5: print("  [EOS Reached]")
            break

    predicted_text_ids = generated_ids[0, 1:].tolist()
    if predicted_text_ids and predicted_text_ids[-1] == EOS_ID:
        predicted_text_ids = predicted_text_ids[:-1]
    predicted_text = tokenizer.decode(predicted_text_ids, skip_special_tokens=True)

    if sample_idx < 5:
        print(f"--- [Sample {sample_idx+1}] Generation End ---")

    return predicted_text, pred_color_for_print, pred_object_ids_list

# ==================================================================================
# --- MAIN ---
# ==================================================================================
if __name__ == "__main__":
    g = torch.Generator().manual_seed(42)
    dataset = EEGMetaTextH5Dataset(H5_FILE_PATH)
    N = len(dataset)
    n_train = int(N * TRAIN_PCT)
    n_val = int(N * VAL_PCT)
    n_test = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=g)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_multimodal_batch)

    print("Creating Granger Causality matrix for inference...")
    try:
        eeg_b, _, _ = next(iter(test_loader))
        granger_edge_index, granger_edge_attr = create_granger_causality_matrix(eeg_b)
        granger_edge_index = granger_edge_index.to(device)
        granger_edge_attr = granger_edge_attr.to(device)
        print(f"Static Granger Graph ready on {device}. Edges: {granger_edge_index.shape[1]}")
    except Exception as e:
        print(f"Error: {e}. Using fallback.")
        num_channels = 62
        edge_index = torch.combinations(torch.arange(num_channels), r=2).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        granger_edge_index = edge_index.to(device)
        granger_edge_attr = torch.ones(granger_edge_index.shape[1], dtype=torch.float32).to(device)

    model = Seq2SeqTF(
        text_vocab_size=TEXT_VOCAB_SIZE,
        num_colors=NUM_COLORS, num_objects=NUM_OBJECTS,
        d_model=D_MODEL, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, d_ff=D_FF, pad_id=PAD_ID, dropout=DROPOUT
    ).to(device)

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"\nSuccessfully loaded best model from '{MODEL_SAVE_PATH}'")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        exit()

    try:
        with open("/home/poorna/data/object_id_to_name_qwen.json", 'r') as f:
            object_mapping = json.load(f)
        print(f"Object mapping loaded: {len(object_mapping)} objects")
    except: object_mapping = {}

    NUM_SAMPLES = 20
    print(f"\n--- Running TRUE END-TO-END Inference on {NUM_SAMPLES} Test Samples ---")

    predictions, references = [], []
    for i in range(NUM_SAMPLES):
        eeg_sample, meta_sample, true_text_ids = test_ds[i]
        true_color_id = int(meta_sample[0].item())
        true_object_names = [object_mapping.get(str(int(id_item)), f"ID:{int(id_item)}") 
                             for id_item in meta_sample[1:].nonzero(as_tuple=True)[0].tolist()] or ["None"]

        predicted_text, pred_color, pred_object_ids = generate_transformer_greedy(
            model, eeg_sample, granger_edge_index, granger_edge_attr, i
        )

        true_text = tokenizer.decode(true_text_ids.tolist(), skip_special_tokens=True)
        predictions.append(predicted_text)
        references.append(true_text)

        pred_object_names = [object_mapping.get(str(oid), f"ID:{oid}") for oid in pred_object_ids] or ["None"]

        print(f"\n--- Sample {i+1}/{NUM_SAMPLES} Summary ---")
        print(f"GROUND TRUTH: {true_text}")
        print(f"PREDICTION:   {predicted_text}")
        print(f"Color: Truth={true_color_id}, Pred={pred_color}")
        print(f"Objects: Truth={', '.join(true_object_names)}, Pred={', '.join(pred_object_names)}")
        print("-" * 50)

    print("\n--- Evaluation Metrics ---")
    try:
        bleu = hf_evaluate.load('bleu').compute(predictions=predictions, references=[[r] for r in references])['bleu']
        rouge = hf_evaluate.load('rouge').compute(predictions=predictions, references=references)
        print(f"BLEU: {bleu:.4f}")
        print(f"ROUGE-1: {rouge['rouge1']:.4f}, ROUGE-2: {rouge['rouge2']:.4f}, ROUGE-L: {rouge['rougeL']:.4f}")
    except Exception as e:
        print(f"Metrics failed: {e}")
