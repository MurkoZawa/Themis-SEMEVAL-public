import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import json 
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from tqdm import tqdm
from themis_model_6L import get_Themis
import re
import warnings
warnings.filterwarnings(action="ignore")

from datasets_6L import get_dataset, Fakeddit_Dataset, fakeddit_load_annotations_file

if __name__ == "__main__":
    # Read arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_llm", type=str)
    parser.add_argument("--name_img_embed", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--merge_tokens", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--use_lora", type=bool)
    parser.add_argument("--number_of_epochs", type=int, default=10)
    parser.add_argument("--n_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")

    name_llm = parser.parse_args().name_llm
    name_img_embed = parser.parse_args().name_img_embed
    batch_size = parser.parse_args().batch_size
    merge_tokens = parser.parse_args().merge_tokens
    if merge_tokens == 0:
        merge_tokens = None
    lora_alpha = parser.parse_args().lora_alpha
    lora_r = parser.parse_args().lora_r
    lora_dropout = parser.parse_args().lora_dropout
    use_lora = parser.parse_args().use_lora
    epochs = parser.parse_args().number_of_epochs
    n_tokens = parser.parse_args().n_tokens
    device = parser.parse_args().device

    # Load Themis, tokenizer, and processor
    themis, tokenizer, processor = get_Themis(
        name_llm=name_llm,
        name_img_embed=name_img_embed,
        use_lora=use_lora,
        is_pythia=True if "pythia" in name_llm else False,
        lora_alpha=lora_alpha,
        lora_r=lora_r,
        lora_dropout=lora_dropout,
        merge_tokens=merge_tokens
    )
    themis.to(device)
    
    dataset_name = "Fakeddit"
    experiment_name = "Fakeddit"
    dataset_train = get_dataset(Fakeddit_Dataset, fakeddit_load_annotations_file, n_tokens, processor, tokenizer, 
                f"{dataset_name}/train.tsv",
                f"{dataset_name}/train")
   
    dataset_val = get_dataset(Fakeddit_Dataset, fakeddit_load_annotations_file, n_tokens, processor, tokenizer, 
                f"{dataset_name}/val.tsv",
                f"{dataset_name}/val")
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, generator=torch.Generator(device='cuda'))

    # Multi-class loss function
    loss = nn.CrossEntropyLoss()
    lr = 0.0001
    optimizer = optim.AdamW(themis.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    warmup_epochs = 0
    warmup_lr = 1e-6
    warmup_optimizer = optim.AdamW(themis.parameters(), lr=warmup_lr)

    def validate(themis, dataloader_val, loss, running_loss, best_f1=0):
        accumulated_labels = []
        accumulated_preds = []
        with torch.no_grad():
            for images, labels, texts in tqdm(dataloader_val):
                images = images.to(device)
                texts = texts.to(device)
                labels = labels.to(device)
                outputs = themis(images, texts)
                loss_val = loss(outputs, labels)
                running_loss += loss_val.item()
                accumulated_labels.extend(labels.to("cpu").numpy())
                accumulated_preds.extend(outputs.argmax(dim=1).to("cpu").numpy())  # Use argmax for multi-class
                
            epoch_loss = running_loss / len(dataloader_val)
            val_loss.append(epoch_loss)
            print(f"Validation loss: {epoch_loss}")
            acc = accuracy_score(accumulated_labels, accumulated_preds)
            prec = precision_score(accumulated_labels, accumulated_preds, average='weighted')
            rec = recall_score(accumulated_labels, accumulated_preds, average='weighted')
            f1 = f1_score(accumulated_labels, accumulated_preds, average='weighted')
            accuracy.append(acc)
            print(f"Accuracy: {acc} || Precision: {prec} || Recall: {rec} || F1: {f1}")
            if f1 > best_f1:
                best_f1 = f1
                path_out = f"{dataset_name}/{experiment_name}/"+name_llm+"_"+name_img_embed+"_"+str(merge_tokens)+"_"+str(lora_alpha)+"_"+str(lora_r)+"_"+str(lora_dropout)+"_"+str(use_lora)+str(epochs)+"_best.pt"
                if not os.path.exists(os.path.dirname(path_out)):
                    os.makedirs(os.path.dirname(path_out))
                torch.save(themis.state_dict(), path_out)
        return best_f1 

    def train_epoch(themis, dataloader_train, loss, optimizer, epoch):
        print(f"Epoch {epoch+1}/{epochs} current lr: {optimizer.param_groups[0]['lr']}")
        print("-" * 10)
        running_loss = 0.0
        
        for images, labels, texts in tqdm(dataloader_train):
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = themis(images, texts)
            loss_val = loss(outputs, labels)
            loss_val.backward()
            optimizer.step()
            running_loss += loss_val.item()
        return running_loss

    themis.train()
    best_f1 = 0
    train_loss = []
    val_loss = []
    accuracy = []
    for epoch in range(epochs + warmup_epochs):
        running_loss = train_epoch(themis, dataloader_train, loss, optimizer, epoch)
        epoch_loss = running_loss / len(dataloader_train)
        train_loss.append(epoch_loss)
        print(f"Training loss: {epoch_loss}")
        print("Evaluating...")
        scheduler.step()
        running_loss = 0.0
        best_f1 = validate(themis, dataloader_val, loss, running_loss, best_f1=best_f1)

    import matplotlib.pyplot as plt

    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.plot(accuracy, label='acc val')
    plt.xticks(range(epochs))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc='upper left')
    plt.savefig(f"{dataset_name}/{experiment_name}/"+name_llm+"_"+name_img_embed.split('/')[0]+"/loss.png")
    plt.show()
