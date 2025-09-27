import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import warnings

from themis_model import get_Themis
from datasets import get_dataset, DGM4_Dataset, dgm4_load_annotations_file

warnings.filterwarnings(action="ignore")

# --- Funzione per calcolare EER ---
def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer

# --- Funzione di validazione ---
def validate(themis, dataloader_val, criterion, device):
    themis.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels, texts, _ in tqdm(dataloader_val, desc="Validating"):
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            outputs = themis(images, texts)
            loss = criterion(outputs.float(), labels.float().unsqueeze(1))
            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader_val)
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, binary_preds)
    auc = roc_auc_score(all_labels, all_preds)
    eer = compute_eer(all_labels, all_preds)

    print(f"Val Loss: {epoch_loss:.4f} || Accuracy: {acc:.4f} || AUC: {auc:.4f} || EER: {eer:.4f}")
    return epoch_loss, acc, auc, eer

# --- Funzione di training per un epoch ---
def train_epoch(themis, dataloader, criterion, optimizer, device):
    themis.train()
    running_loss = 0.0

    for images, labels, texts, _ in tqdm(dataloader, desc="Training"):
        images, texts, labels = images.to(device), texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = themis(images, texts)
        loss = criterion(outputs.float(), labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(dataloader)

# --- Training su sottoinsieme (stepwise) ---
def train_on_subset(themis, subset, batch_size, criterion, optimizer, device):
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    return train_epoch(themis, loader, criterion, optimizer, device)

# --- MAIN ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name_llm", type=str, required=True)
    parser.add_argument("--name_img_embed", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--merge_tokens", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--lora_r", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, required=True)
    parser.add_argument("--use_lora", type=bool, required=True)
    parser.add_argument("--number_of_epochs", type=int, default=10)
    parser.add_argument("--n_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--stepwise_samples", type=int, default=None,
                        help="Se specificato, esegue training stepwise su blocchi di questo numero di samples")
    args = parser.parse_args()

    device = args.device
    merge_tokens = args.merge_tokens if args.merge_tokens != 0 else None

    # --- Modello ---
    themis, tokenizer, processor = get_Themis(
        name_llm=args.name_llm,
        name_img_embed=args.name_img_embed,
        use_lora=args.use_lora,
        is_pythia="pythia" in args.name_llm,
        lora_alpha=args.lora_alpha,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        merge_tokens=merge_tokens,
    )
    themis.to(device)

    # --- Caricamento checkpoint ---
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        missing_keys, unexpected_keys = themis.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    # --- Dataset ---
    dataset_name = "DGM4"
    experiment_name = "DGM4"
    dataset_train_full = get_dataset(
        DGM4_Dataset, dgm4_load_annotations_file, args.n_tokens, processor, tokenizer,
        f"{dataset_name}/train.tsv", f"{dataset_name}/train"
    )
    dataset_val = get_dataset(
        DGM4_Dataset, dgm4_load_annotations_file, args.n_tokens, processor, tokenizer,
        f"{dataset_name}/val.tsv", f"{dataset_name}/val"
    )
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                generator=torch.Generator(device='cuda'))

    # --- Setup training ---
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(themis.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.number_of_epochs)

    # --- Liste metriche ---
    train_loss, val_loss = [], []
    accuracy, aucs, eers = [], [], []

    save_dir = f"{dataset_name}/{experiment_name}/{args.name_llm}_{args.name_img_embed.split('/')[0]}"
    os.makedirs(save_dir, exist_ok=True)
    log_path = f"{save_dir}/metrics_log.csv"

    # --- Loop principale ---
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Train_Loss", "Val_Loss", "Accuracy", "AUC", "EER"])

        if args.stepwise_samples is None:
            print(f">> Allenamento normale per {args.number_of_epochs} epoche.")
            dataloader_train = DataLoader(dataset_train_full, batch_size=args.batch_size, shuffle=True,
                                          generator=torch.Generator(device='cuda'))
            for epoch in range(args.number_of_epochs):
                print(f"\nEpoch {epoch + 1}/{args.number_of_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")
                epoch_train_loss = train_epoch(themis, dataloader_train, criterion, optimizer, device)
                train_loss.append(epoch_train_loss)

                val_epoch_loss, acc, auc_val, eer_val = validate(themis, dataloader_val, criterion, device)
                val_loss.append(val_epoch_loss)
                accuracy.append(acc)
                aucs.append(auc_val)
                eers.append(eer_val)

                writer.writerow([epoch + 1, epoch_train_loss, val_epoch_loss, acc, auc_val, eer_val])
                f.flush()
                scheduler.step()
        else:
            print(f">> Allenamento stepwise con incrementi di {args.stepwise_samples} samples.")
            total_samples = len(dataset_train_full)
            indices = list(range(total_samples))
            step_count = 1

            for step_count, end_idx in enumerate(range(args.stepwise_samples, total_samples + args.stepwise_samples, args.stepwise_samples),1):
                end_idx = min(end_idx, total_samples)
                subset_indices = indices[0:end_idx]
                dataset_subset = Subset(dataset_train_full, subset_indices)

                print(f"\n[STEPWISE] Training su campioni 0-{end_idx - 1} (tot: {len(dataset_subset)})")
                epoch_train_loss = train_on_subset(themis, dataset_subset, args.batch_size, criterion, optimizer, device)
                train_loss.append(epoch_train_loss)

                val_epoch_loss, acc, auc_val, eer_val = validate(themis, dataloader_val, criterion, device)
                val_loss.append(val_epoch_loss)
                accuracy.append(acc)
                aucs.append(auc_val)
                eers.append(eer_val)

                writer.writerow([step_count, epoch_train_loss, val_epoch_loss, acc, auc_val, eer_val])
                f.flush()
                scheduler.step()
                step_count += 1

    # --- Plot finale ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.plot(accuracy, label='Accuracy')
    plt.plot(aucs, label='AUC')
    plt.plot(eers, label='EER')
    plt.xlabel("Training steps")
    plt.ylabel("Metric value")
    plt.title("Themis's performance during training steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_plot.png")
    plt.show()