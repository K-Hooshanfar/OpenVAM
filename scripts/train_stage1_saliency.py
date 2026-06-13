import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import torch

import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from utils.losses import SaliencyLoss
from utils.data import MergedTrainDataset, MergedValDataset, merged_collate_fn

parser = argparse.ArgumentParser(description="Train OpenVAMSaliencyNet with a configurable backbone model.")
parser.add_argument(
    "--model",
    type=str,
    default="facebook/dinov3-vitb16-pretrain-lvd1689m",
    help="HuggingFace model name for the DINOv3 backbone.",
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    help="Path to save the best model weights. Defaults to best_<model_slug>.pth",
)
args = parser.parse_args()

HF_MODEL_NAME = args.model
if args.save_path:
    SAVE_PATH = args.save_path
else:
    model_slug = HF_MODEL_NAME.replace("/", "_").replace("-", "_")
    SAVE_PATH = f"best_{model_slug}.pth"

print(f"Using backbone model : {HF_MODEL_NAME}")
print(f"Best model save path : {SAVE_PATH}")

# Dataset names in merged JSONL ids (id format: {dataset_name}_{original_id})
MERGED_DATASET_NAMES = [
    "datasets_UI_256", "CAT2000_256", "MIT1003_256", "OSIE_256",
    "salicon_256", "fiwi_256", "SalEC",
]

def get_dataset_from_id(sample_id):
    if not sample_id:
        return "other"
    for name in MERGED_DATASET_NAMES:
        if sample_id.startswith(name + "_"):
            return name
    return "other"
    
from net.saliency_net import OpenVAMSaliencyNet

# Merged dataset: datasets/merged/merged_train.jsonl, merged_val.jsonl; dirs train/{stimuli,saliency,fixations}, val/{stimuli,saliency,fixations}
MERGED_DIR = "datasets/merged"

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = MergedTrainDataset(merged_dir=MERGED_DIR, jsonl_name="merged_train.jsonl", transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)



val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = MergedValDataset(merged_dir=MERGED_DIR, jsonl_name="merged_val.jsonl", transform=val_transform)
val_loaders = {"val_loader_0": DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=merged_collate_fn)}

# Startup check: report how many samples have non-zero fixation / saliency
if hasattr(train_dataset, "check_fixation_saliency_coverage"):
    n_fix_tr, n_sal_tr, n_tr, ids_miss_tr = train_dataset.check_fixation_saliency_coverage(sample_limit=500)
    print(f"[MergedDataset] Train (first {n_tr}): {n_fix_tr}/{n_tr} with non-zero fixation, {n_sal_tr}/{n_tr} with non-zero saliency")
    if ids_miss_tr and len(ids_miss_tr) <= 20:
        print(f"  Train ids with zero fixation: {ids_miss_tr}")
    elif ids_miss_tr:
        print(f"  Train ids with zero fixation (first 20): {ids_miss_tr[:20]} ...")
if hasattr(val_dataset, "check_fixation_saliency_coverage"):
    n_fix_val, n_sal_val, n_val, ids_miss_val = val_dataset.check_fixation_saliency_coverage(sample_limit=min(1000, len(val_dataset)))
    print(f"[MergedDataset] Val (first {n_val}): {n_fix_val}/{n_val} with non-zero fixation, {n_sal_val}/{n_val} with non-zero saliency")
    if ids_miss_val and len(ids_miss_val) <= 30:
        print(f"  Val ids with zero fixation: {ids_miss_val}")
    elif ids_miss_val:
        print(f"  Val ids with zero fixation (first 30): {ids_miss_val[:30]} ...")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)


# Provide your own Hugging Face access token via the HF_TOKEN (or HUGGING_FACE_HUB_TOKEN)
# environment variable, e.g.  export HF_TOKEN=hf_xxx
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

model = OpenVAMSaliencyNet(
    use_hf=True,
    hf_model_name=HF_MODEL_NAME,
    hf_token=HF_TOKEN,
    backbone="vitb_rn50_384",
    features=256,
    readout="project",
    upsample_output_to_input_res=True,
    out_channels=1,
    freeze_dino=False,
    freeze_vit=False,  
)


# Move the entire model and submodules to GPU
model = model.to(device)

dummy_input = torch.randn(1, 3, 256, 256).to(device)
# Run once to initialize lazy layers
_ = model(dummy_input)

# Print trainable parameters
print("\nTrainable Parameters:")
total_params = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        num = param.numel()
        total_params += num
        # print(f"{name:60} | {num:,}")
print(f"\nTotal Trainable Parameters: {total_params:,}\n")

optimizer = optim.Adam(model.parameters(), lr=7.4e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
loss_fn = SaliencyLoss() 
mse_loss = nn.MSELoss()

cpu_tensors = []

for group in optimizer.param_groups:
    for p in group['params']:
        if p.grad is not None:
            if p.device != device:
                print(f"[ERROR] Param {p.shape} on {p.device}")
            if p.grad.device != device:
                print(f"[ERROR] Grad for param {p.shape} is on {p.grad.device}")
            if p.device.type == 'cpu' or p.grad.device.type == 'cpu':
                cpu_tensors.append(p)

if cpu_tensors:
    print(f"❌ Found {len(cpu_tensors)} param(s) with CPU device mismatch!")
else:
    print("✅ All params and grads are on correct device.")
        
# Training and Validation Loop
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')
num_epochs = 30

# Early stopping setup
early_stop_counter = 0
early_stop_threshold = 4

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    # Training Phase
    model.train()
    metrics = {'loss': [], 'kl': [], 'cc': [], 'sim': [], 'nss': []}

    for batch in tqdm(train_loader, desc="Training"):
        stimuli, smap, fmap, condition = batch['image'].to(device), batch['saliency'].to(device), batch['fixation'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(stimuli)
        
        # Compute losses
        kl = loss_fn(outputs, smap, loss_type='kldiv')
        cc = loss_fn(outputs, smap, loss_type='cc')
        sim = loss_fn(outputs, smap, loss_type='sim')
        nss = loss_fn(outputs, fmap, loss_type='nss')
        loss1 = -2.15*cc + 12*kl - 1.69*sim - 3.17*nss
        loss2 = mse_loss(outputs, smap)
        loss = loss1 + 2.99 * loss2

        loss.backward()
        optimizer.step()

        # Accumulate raw metric values
        metrics['loss'].append(loss.item())
        metrics['kl'].append(kl.item())
        metrics['cc'].append(cc.item())
        metrics['sim'].append(sim.item())
        metrics['nss'].append(nss.item())

    scheduler.step()

    # Calculate mean and std dev for each metric
    for metric in metrics.keys():
        metrics[metric] = (np.mean(metrics[metric]), np.std(metrics[metric]))
    
    # Print training metrics with mean and std dev
    print("Train - " + ", ".join([f"{metric}: {mean:.4f} ± {std:.4f}" for metric, (mean, std) in metrics.items()]))

    # After first epoch: report which samples had to be resized (if any)
    if epoch == 0 and hasattr(train_dataset, "get_resized_report"):
        report = train_dataset.get_resized_report()
        if report:
            print(f"[MergedDataset] Summary: {len(report)} load(s) were resized (see lines above). Unique (id, field) count: {len(report)}")
        else:
            print("[MergedDataset] No resizing needed: all images/saliency/fixation were already 256x256.")
    
    # Validation Phase: per-sample metrics grouped by dataset (from id prefix)
    model.eval()
    per_ds = {}  # dataset_name -> { 'loss': [], 'kl': [], ... }

    for name, loader in val_loaders.items():
        for batch in tqdm(loader, desc=f"Validating {name}"):
            stimuli = batch['image'].to(device)
            smap = batch['saliency'].to(device)
            fmap = batch['fixation'].to(device)
            ids = batch.get('id', [])
            B = stimuli.shape[0]
            if B != len(ids):
                ids = ids + [""] * (B - len(ids))

            with torch.no_grad():
                outputs = model(stimuli)

            for i in range(B):
                ds = get_dataset_from_id(ids[i] if i < len(ids) else "")
                if ds not in per_ds:
                    per_ds[ds] = {'loss': [], 'kl': [], 'cc': [], 'sim': [], 'nss': [], 'auc': []}

                o_i = outputs[i : i + 1]
                s_i = smap[i : i + 1]
                f_i = fmap[i : i + 1]
                has_fix = f_i.sum().item() > 0

                kl = loss_fn(o_i, s_i, loss_type='kldiv').item()
                cc = loss_fn(o_i, s_i, loss_type='cc').item()
                sim = loss_fn(o_i, s_i, loss_type='sim').item()
                per_ds[ds]['kl'].append(kl)
                per_ds[ds]['cc'].append(cc)
                per_ds[ds]['sim'].append(sim)

                nss_val = 0.0
                if has_fix:
                    nss_val = loss_fn(o_i, f_i, loss_type='nss').item()
                    nss_val = nss_val if not np.isnan(nss_val) else 0.0
                    per_ds[ds]['nss'].append(nss_val)
                    auc_val = loss_fn(o_i, f_i, loss_type='auc').item()
                    per_ds[ds]['auc'].append(auc_val if not np.isnan(auc_val) else None)

                loss1 = -2.15 * cc + 12 * kl - 1.69 * sim - 3.17 * nss_val
                loss2 = mse_loss(o_i, s_i).item()
                per_ds[ds]['loss'].append(loss1 + 2.99 * loss2)

        # Overall val metrics (aggregate all datasets)
        all_kl, all_cc, all_sim, all_nss, all_auc, all_loss = [], [], [], [], [], []
        for ds, m in per_ds.items():
            all_kl.extend(m['kl'])
            all_cc.extend(m['cc'])
            all_sim.extend(m['sim'])
            all_loss.extend(m['loss'])
            all_nss.extend([x for x in m['nss'] if x is not None and not np.isnan(x)])
            all_auc.extend([x for x in m['auc'] if x is not None and not np.isnan(x)])
        overall = {
            'loss': (np.mean(all_loss), np.std(all_loss)),
            'kl': (np.mean(all_kl), np.std(all_kl)),
            'cc': (np.mean(all_cc), np.std(all_cc)),
            'sim': (np.mean(all_sim), np.std(all_sim)),
            'nss': (np.mean(all_nss), np.std(all_nss)) if all_nss else (float('nan'), float('nan')),
            'auc': (np.mean(all_auc), np.std(all_auc)) if all_auc else (float('nan'), float('nan')),
        }
        metrics_str = ", ".join([f"{k}: {v[0]:.4f} ± {v[1]:.4f}" for k, v in overall.items()])
        print(f"{name} - Val (overall): {metrics_str}")

        # Per-dataset metrics
        print("Val per dataset:")
        for ds in sorted(per_ds.keys()):
            m = per_ds[ds]
            n = len(m['loss'])
            nss_vals = [x for x in m['nss'] if x is not None and not np.isnan(x)]
            auc_vals = [x for x in m['auc'] if x is not None and not np.isnan(x)]
            n_nss = len(nss_vals)
            line = (f"  {ds}: n={n} | loss={np.mean(m['loss']):.4f} kl={np.mean(m['kl']):.4f} "
                    f"cc={np.mean(m['cc']):.4f} sim={np.mean(m['sim']):.4f}")
            if n_nss:
                line += f" nss={np.mean(nss_vals):.4f} auc={np.mean(auc_vals):.4f}" if auc_vals else f" nss={np.mean(nss_vals):.4f} auc=—"
            else:
                line += " nss=— auc=— (no fixations)"
            print(line)

        # Keep val_metrics for total_val_loss (use overall kl)
        val_metrics = {name: {'kl': overall['kl']}}

    # After validation phase
    total_val_loss = sum(val_metrics[n]['kl'][0] for n in val_loaders.keys())

    print(f"Epoch {epoch+1}: Total Val Loss across all datasets: {total_val_loss:.4f}")

    # Check for best model
    if total_val_loss < best_loss:
        print(f"New best model found at epoch {epoch+1}!")
        best_loss = total_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, SAVE_PATH)
        early_stop_counter = 0  # Reset counter after improvement
    else:
        early_stop_counter += 1
        print(f"No improvement in Total Val Loss for {early_stop_counter} epoch(s).")

    # Early stopping check
    if early_stop_counter >= early_stop_threshold:
        print("Early stopping triggered.")
        break