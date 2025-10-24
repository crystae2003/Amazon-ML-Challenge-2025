import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import pandas as pd
import numpy as np
import faiss
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
# <-- 1. IMPORT THE COSINE SCHEDULER
from transformers import get_cosine_schedule_with_warmup
from sklearn.preprocessing import MinMaxScaler

# DDP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- Mixed Precision Imports ---
from torch.cuda.amp import GradScaler, autocast

# --- Configuration ---
K_NEIGHBORS = 5
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
TRAIN_CSV_PATH = '/home/user3/amazon/train_split_final.csv'
VAL_CSV_PATH = '/home/user3/amazon/val_split_final.csv'

CHECKPOINT_PATH = '/home/user3/amazon/best_finetuned_model_state_dict.pth'
CHECKPOINT_SAVE_PATH = '/home/user3/amazon/best_with_image_KNN.pth'

# --- PyTorch/MLP Configuration ---
BATCH_SIZE = 32
LEARNING_RATE_TRANSFORMER = 1e-5
LEARNING_RATE_MLP = 1e-4
EPOCHS = 10
DROPOUT_RATE = 0.3

# --- Helper Functions (smape, faiss, etc.) ---
def smape_loss_numpy(y_pred, y_true, epsilon=1e-8):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    loss = numerator / (denominator + epsilon)
    return np.mean(loss) * 100

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def find_neighbor_indices(index, query_embeddings, k, is_training=False):
    search_k = k + 1 if is_training else k
    _, indices = index.search(query_embeddings, k=search_k)
    if is_training:
        return indices[:, 1:]
    return indices

# --- Dataset & Collate Function ---
class PriceDataset(Dataset):
    def __init__(self, query_df, ref_df, neighbor_indices):
        self.query_df = query_df.reset_index(drop=True)
        self.ref_df = ref_df.reset_index(drop=True)
        self.neighbor_indices = neighbor_indices

    def __len__(self):
        return len(self.query_df)

    def __getitem__(self, idx):
        query_row = self.query_df.iloc[idx]
        query_text = query_row['desc_for_llm']
        query_features = torch.tensor([
            query_row['Count'], query_row['oz'], query_row['fl_oz']
        ], dtype=torch.float32)
        target_price = torch.tensor([query_row['price']], dtype=torch.float32)
        neighbor_idxs = self.neighbor_indices[idx]
        neighbor_rows = self.ref_df.iloc[neighbor_idxs]
        neighbor_texts = neighbor_rows['desc_for_llm'].tolist()
        neighbor_features = torch.tensor(
            neighbor_rows[['Count', 'oz', 'fl_oz', 'price']].values,
            dtype=torch.float32
        )
        return {
            'query_text': query_text, 'query_features': query_features,
            'neighbor_texts': neighbor_texts, 'neighbor_features': neighbor_features,
            'target': target_price
        }

def create_collate_fn(tokenizer):
    def custom_collate_fn(batch):
        query_texts = [item['query_text'] for item in batch]
        all_neighbor_texts = [neighbor for item in batch for neighbor in item['neighbor_texts']]
        query_tokens = tokenizer(query_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        neighbor_tokens = tokenizer(all_neighbor_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        return {
            'query_tokens': query_tokens,
            'neighbor_tokens': neighbor_tokens,
            'query_features': torch.stack([item['query_features'] for item in batch]),
            'neighbor_features': torch.stack([item['neighbor_features'] for item in batch]),
            'targets': torch.stack([item['target'] for item in batch])
        }
    return custom_collate_fn

# --- Model Architecture ---
class EndToEndPricePredictor(nn.Module):
    def __init__(self, model_name, mlp_dropout=0.3):
        super().__init__()
        self.sbert = SentenceTransformer(model_name)
        self.sbert.max_seq_length = 64
        embedding_dim = self.sbert.get_sentence_embedding_dimension()
        mlp_input_size = (embedding_dim + 3) + K_NEIGHBORS * (embedding_dim + 4)
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(512, 1)
        )

    def forward(self, batch):
        device = batch['query_features'].device
        query_tokens = {k: v.to(device) for k, v in batch['query_tokens'].items()}
        neighbor_tokens = {k: v.to(device) for k, v in batch['neighbor_tokens'].items()}
        query_embs = self.sbert(query_tokens)['sentence_embedding']
        neighbor_embs = self.sbert(neighbor_tokens)['sentence_embedding']
        batch_size = batch['query_features'].shape[0]
        embedding_dim = query_embs.shape[1]
        if batch_size == 0: raise ValueError("Empty batch encountered.")
        total_neighbors = neighbor_embs.shape[0]
        k_actual = total_neighbors // batch_size
        neighbor_embs = neighbor_embs.view(batch_size, k_actual, embedding_dim)
        neighbor_feats = batch['neighbor_features'].to(device)
        if k_actual < K_NEIGHBORS:
            padded_embs = torch.zeros((batch_size, K_NEIGHBORS, embedding_dim), device=device, dtype=neighbor_embs.dtype)
            padded_feats = torch.zeros((batch_size, K_NEIGHBORS, neighbor_feats.shape[2]), device=device, dtype=neighbor_feats.dtype)
            padded_embs[:, :k_actual, :] = neighbor_embs
            padded_feats[:, :k_actual, :] = neighbor_feats
            neighbor_embs, neighbor_feats = padded_embs, padded_feats
        elif k_actual > K_NEIGHBORS:
            neighbor_embs = neighbor_embs[:, :K_NEIGHBORS, :].contiguous()
            neighbor_feats = neighbor_feats[:, :K_NEIGHBORS, :].contiguous()
        query_vec = torch.cat([query_embs, batch['query_features'].to(device)], dim=1)
        neighbor_vecs = torch.cat([neighbor_embs, neighbor_feats], dim=2)
        neighbor_vecs_flat = neighbor_vecs.view(batch_size, -1)
        combined_vec = torch.cat([query_vec, neighbor_vecs_flat], dim=1)
        return self.mlp_head(combined_vec)

# --- DDP Setup and helpers ---
def setup_ddp(rank, world_size):
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def move_batch_to_device(batch, device):
    batch['query_tokens'] = {k: v.to(device) for k, v in batch['query_tokens'].items()}
    batch['neighbor_tokens'] = {k: v.to(device) for k, v in batch['neighbor_tokens'].items()}
    for key in ['query_features', 'neighbor_features', 'targets']:
        batch[key] = batch[key].to(device)
    return batch

def gather_tensors_across_ranks(local_tensor):
    device = local_tensor.device
    world_size = dist.get_world_size()
    local_len = torch.tensor([local_tensor.shape[0]], device=device)
    lengths = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lengths, local_len)
    lengths = [int(x.item()) for x in lengths]
    max_len = max(lengths)
    C = local_tensor.shape[1] if local_tensor.ndim == 2 else 1
    padded = torch.zeros((max_len, C), device=device)
    if local_tensor.shape[0] > 0:
        padded[:local_tensor.shape[0], ...] = local_tensor
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    if dist.get_rank() == 0:
        parts = [gathered[r][:ln, ...].cpu() for r, ln in enumerate(lengths) if ln > 0]
        return torch.cat(parts, dim=0) if parts else torch.empty((0, C))
    return None

# --- Main worker ---
def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        _ = SentenceTransformer(MODEL_NAME)
    dist.barrier()

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    val_df = pd.read_csv(VAL_CSV_PATH)
    train_neighbor_indices = np.load('train_neighbor_indices.npy')
    val_neighbor_indices = np.load('val_neighbor_indices.npy')
    
    train_df['Count'] = np.log1p(train_df['Count'])
    val_df['Count'] = np.log1p(val_df['Count'])
    scaler = MinMaxScaler()
    scaler.fit(train_df[['Count']])
    train_df['Count'] = scaler.transform(train_df[['Count']])
    val_df['Count'] = scaler.transform(val_df[['Count']])
    train_df['oz'] = np.log1p(train_df['oz'])
    val_df['oz'] = np.log1p(val_df['oz'])
    train_df['fl_oz'] = np.log1p(train_df['fl_oz'])
    val_df['fl_oz'] = np.log1p(val_df['fl_oz'])

    model = EndToEndPricePredictor(MODEL_NAME, mlp_dropout=DROPOUT_RATE).to(device)
    tokenizer = model.sbert.tokenizer
    collate_fn = create_collate_fn(tokenizer)
    train_dataset = PriceDataset(train_df, train_df, train_neighbor_indices)
    val_dataset = PriceDataset(val_df, train_df, val_neighbor_indices)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=collate_fn, sampler=val_sampler, pin_memory=True)

    optimizer = torch.optim.AdamW([
        {'params': model.sbert.parameters(), 'lr': LEARNING_RATE_TRANSFORMER},
        {'params': model.mlp_head.parameters(), 'lr': LEARNING_RATE_MLP}
    ])

    # <-- 2. INITIALIZE THE COSINE SCHEDULER
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    
    start_epoch = 0
    best_smape = float('inf')

    # ROBUST CHECKPOINT LOADING LOGIC
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        if rank == 0:
            print(f"--> Loading from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_smape = checkpoint.get('best_smape', float('inf'))
            if rank == 0:
                print(f"--> Resumed training from epoch {start_epoch}. Best SMAPE was {best_smape:.4f}%.")
        else:
            model.load_state_dict(checkpoint)
            if rank == 0:
                print("--> Loaded a weights-only file. Optimizer and scheduler will start from scratch.")
    else:
        if rank == 0:
            print("--> No checkpoint found. Training from scratch.")

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    criterion = nn.MSELoss()
    grad_scaler = GradScaler()

    if rank == 0:
        print(f"\nStarting DDP training on {world_size} GPUs with batch size {BATCH_SIZE} per GPU...")

    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        train_sampler.set_epoch(epoch)
        model.train()
        total_train_loss = 0.0        
        running_loss = 0.0 

        for i, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            target_log = torch.log1p(batch['targets'])
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                output_log = model(batch)
                loss = criterion(output_log, target_log)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            scheduler.step()
            grad_scaler.update()
            current_loss = loss.item()
            total_train_loss += current_loss
            running_loss += current_loss

            if (i + 1) % 100 == 0 and rank == 0:
                avg_last_100_loss = running_loss / 100
                print(f"   Epoch {epoch+1}, Step {i+1}/{len(train_loader)}, Avg Loss (last 100): {avg_last_100_loss:.4f}")
                running_loss = 0.0

        model.eval()
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                with autocast():
                    out_log = model(batch)
                all_val_preds.append(torch.expm1(out_log))
                all_val_targets.append(batch['targets'])

        gathered_preds = gather_tensors_across_ranks(torch.cat(all_val_preds) if all_val_preds else torch.empty((0,1), device=device))
        gathered_targets = gather_tensors_across_ranks(torch.cat(all_val_targets) if all_val_targets else torch.empty((0,1), device=device))

        if rank == 0:
            val_preds_np = gathered_preds.cpu().numpy().flatten()[:len(val_dataset)]
            val_targets_np = gathered_targets.cpu().numpy().flatten()[:len(val_dataset)]
            val_smape = smape_loss_numpy(val_preds_np, val_targets_np)
            avg_train_loss = total_train_loss / max(1, len(train_loader))
            print("-" * 60)
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val SMAPE: {val_smape:.4f}% | Time: {time.time() - start_time:.2f}s")
            print("-" * 60)

            if val_smape < best_smape:
                best_smape = val_smape
                print(f"   -> New best model saved with SMAPE: {best_smape:.4f}%")
                # SAVE THE FULL CHECKPOINT
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_smape': best_smape
                }
                torch.save(checkpoint, CHECKPOINT_SAVE_PATH)
    cleanup_ddp()

if __name__ == "__main__":
    main()