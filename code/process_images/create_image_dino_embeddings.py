# generate_embeddings_ddp.py
import os
import argparse
import pickle
import subprocess
import time
from pathlib import Path
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import pandas as pd

# ------------------------------
# Re-download utility (wget)
# ------------------------------
def redownload_image(sample_id, image_link_map, retry_folder, suffix=".jpg", retries=3, delay=2):
    """
    Re-download image for sample_id into retry_folder. Returns path if success else None.
    """
    if sample_id not in image_link_map:
        return None
    url = image_link_map[sample_id]
    os.makedirs(retry_folder, exist_ok=True)
    target = os.path.join(retry_folder, f"{sample_id}{suffix}")
    for attempt in range(retries):
        try:
            cmd = ["wget", "-q", "--timeout=15", "--tries=1", "-O", target, url]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(target):
                return target
        except Exception:
            pass
        time.sleep(delay)
    return None

# ------------------------------
# Dataset: returns file paths + sample_ids
# ------------------------------
class ImagePathDataset(Dataset):
    def __init__(self, folder_path, exts=('.jpg', '.jpeg', '.png')):
        self.folder_path = folder_path
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(exts)]
        files.sort()  # deterministic order
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        sample_id = os.path.splitext(fname)[0]
        return os.path.join(self.folder_path, fname), sample_id

# ------------------------------
# Collate that returns lists (default collate would try to stack strings)
# ------------------------------
def collate_paths(batch):
    paths, sample_ids = zip(*batch)
    return list(paths), list(sample_ids)

# ------------------------------
# Main worker (per-process)
# ------------------------------
def main_worker(rank, world_size, args):
    use_cuda = torch.cuda.is_available() and world_size > 0
    # If no GPUs available, run single-process on CPU
    if use_cuda:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
        dist.init_process_group(backend=backend, init_method='tcp://127.0.0.1:29500',
                                world_size=world_size, rank=rank)
    else:
        device = torch.device("cpu")
        # no distributed init if CPU-only
        if world_size > 1:
            # fallback to gloo if user forced multi-process CPU (rare)
            dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500',
                                    world_size=world_size, rank=rank)

    # Only rank 0 prints high level logs
    def log(msg):
        if rank == 0:
            print(msg)

    log(f"[rank {rank}] Device: {device}")

    # Load model (each process loads model and moves to its device)
    log(f"[rank {rank}] Loading model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
    model.eval()
    model.to(device)

    # If using CUDA, wrap with DDP
    if use_cuda and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    # image link map for re-downloads (string keys)
    df = pd.read_csv(args.csv_path)
    image_link_map = dict(zip(df['sample_id'].astype(str), df['image_link']))

    # dataset + distributed sampler
    dataset = ImagePathDataset(args.image_folder)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None

    per_proc_batch = max(1, args.batch_size // max(1, world_size))
    dataloader = DataLoader(
        dataset,
        batch_size=per_proc_batch,
        sampler=sampler,
        shuffle=(sampler is None and args.shuffle),
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_paths
    )

    # Embedding loop
    embeddings = OrderedDict()
    tmp_out_dir = args.tmp_out_dir
    os.makedirs(tmp_out_dir, exist_ok=True)
    tmp_out_path = os.path.join(tmp_out_dir, f"embeddings_rank{rank}.pkl")

    if sampler is not None:
        sampler.set_epoch(0)

    log(f"[rank {rank}] Starting processing: {len(dataloader)} batches (batch_size per-proc = {per_proc_batch})")

    for paths_batch, sample_ids_batch in tqdm(dataloader, desc=f"rank{rank}", disable=(rank != 0 or args.hide_progress)):
        # build a list of tensors for this batch (may shrink if corrupted files re-downloaded)
        processed_tensors = []
        valid_sample_ids = []

        for path, sid in zip(paths_batch, sample_ids_batch):
            success = False
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    tensor = preprocess(img)
                    processed_tensors.append(tensor)
                    valid_sample_ids.append(sid)
                    success = True
            except UnidentifiedImageError:
                # try re-download into retry folder, print name and retry count
                print(f"[rank {rank}] Corrupted image: {os.path.basename(path)}; attempting re-download...")
                suffix = Path(path).suffix or ".jpg"
                retry_path = redownload_image(sid, image_link_map, args.retry_folder, suffix=suffix,
                                              retries=args.retry_count, delay=args.retry_delay)
                if retry_path:
                    try:
                        with Image.open(retry_path) as img:
                            img = img.convert("RGB")
                            tensor = preprocess(img)
                            processed_tensors.append(tensor)
                            valid_sample_ids.append(sid)
                            success = True
                            print(f"[rank {rank}] Re-download succeeded: {sid}")
                    except Exception as ee:
                        print(f"[rank {rank}] Failed to open re-downloaded file for {sid}: {ee}")
                else:
                    print(f"[rank {rank}] Re-download failed for {sid} after {args.retry_count} tries; skipping.")
            except Exception as e:
                print(f"[rank {rank}] Unexpected error reading {path}: {e}")

            # if not success: skipped

        if not processed_tensors:
            continue  # nothing to process in this batch

        # create batch tensor and move to device
        batch_tensor = torch.stack(processed_tensors, dim=0).to(device)
        with torch.no_grad():
            batch_emb = model(batch_tensor)
            # If model is DDP wrapped, batch_emb is still a tensor
            batch_emb = batch_emb.cpu().numpy()

        for sid, emb in zip(valid_sample_ids, batch_emb):
            embeddings[str(sid)] = emb

    # Save per-rank embeddings
    with open(tmp_out_path, "wb") as f:
        pickle.dump(embeddings, f)
    log(f"[rank {rank}] Saved {len(embeddings)} embeddings to {tmp_out_path}")

    # synchronize: ensure all ranks wrote their files
    if world_size > 1:
        dist.barrier()

    # Rank 0 merges partials and writes final output
    if (not use_cuda and world_size == 1) or (world_size > 1 and rank == 0):
        # wait for all partials if in distributed mode
        if world_size > 1:
            # allow small polling wait if needed
            for _ in range(60):
                exist_all = all(os.path.exists(os.path.join(tmp_out_dir, f"embeddings_rank{r}.pkl")) for r in range(world_size))
                if exist_all:
                    break
                time.sleep(0.5)

        merged = {}
        for r in range(world_size if world_size > 1 else 1):
            part_path = os.path.join(tmp_out_dir, f"embeddings_rank{r}.pkl")
            if not os.path.exists(part_path):
                print(f"[rank 0] Warning: missing partial file {part_path}, skipping")
                continue
            with open(part_path, "rb") as f:
                part = pickle.load(f)
            # update merged with part, no overwrite expected because partitions are disjoint
            merged.update(part)

        final_out = os.path.join(args.output_dir, args.output_file)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(final_out, "wb") as f:
            pickle.dump(merged, f)
        log(f"[rank 0] Merged embeddings saved to {final_out} (total {len(merged)})")

        # optional: cleanup partials
        for r in range(world_size if world_size > 1 else 1):
            p = os.path.join(tmp_out_dir, f"embeddings_rank{r}.pkl")
            try:
                os.remove(p)
            except Exception:
                pass

    # cleanup dist
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

# ------------------------------
# Launcher
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-folder", type=str, required=True, help="Folder containing downloaded images")
    p.add_argument("--csv-path", type=str, required=True, help="CSV with sample_id,image_link columns")
    p.add_argument("--output-dir", type=str, default="./", help="Directory to save final embeddings")
    p.add_argument("--output-file", type=str, default="embeddings.pkl", help="Name of final embeddings file")
    p.add_argument("--tmp-out-dir", type=str, default="./tmp_embeddings", help="Tmp per-rank outputs")
    p.add_argument("--retry-folder", type=str, default="./redownloaded_images", help="Folder to save re-downloaded files")
    p.add_argument("--batch-size", type=int, default=64, help="Global batch size (will be split across ranks)")
    p.add_argument("--num-workers", type=int, default=4, help="Dataloader workers per process")
    p.add_argument("--world-size", type=int, default=2, help="Number of processes / GPUs to use")
    p.add_argument("--shuffle", action='store_true', help="Shuffle dataset (default: False)")
    p.add_argument("--retry-count", type=int, default=3, help="Number of times to re-download corrupted image")
    p.add_argument("--retry-delay", type=int, default=2, help="Seconds between re-download attempts")
    p.add_argument("--hide-progress", action='store_true', help="Hide tqdm progressbars for non-rank0")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Detect GPU availability; if not available, force world_size=1
    if not torch.cuda.is_available():
        print("CUDA not available — running single-process CPU mode.")
        args.world_size = 1

    # If world_size==1 just call main_worker directly
    if args.world_size == 1:
        main_worker(0, 1, args)
    else:
        # spawn processes on the current node
        mp.spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
