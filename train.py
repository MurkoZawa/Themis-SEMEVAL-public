import torch
import torch.nn as nn
import torch.optim as optim

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import json 
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tqdm import tqdm
from themis_model import get_Themis
import re

if __name__ == "__main__":
    #read arguments
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

    def clean_text(text):
        if text == "" or text == None:
            return "Blank"
        # Remove escape sequences and replace "\\n" with a space
        cleaned_text = re.sub(r'\\n', ' ', text)
        # Remove any other special characters or patterns as needed
        cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', cleaned_text)
        return cleaned_text


    class EVALITA_Dataset(Dataset):
        def __init__(self, annotations, img_dir, preprocessor=None, tokenizer=None):
            #file = open(annotations_file, "r", encoding="utf-8")
            #annotations = json.load(file)

            #convert to pandas dataframe
            #if the label is "propagandistic" then 1 else 0
            img_labels = []
            for annotation in annotations:
                img_labels.append([annotation["id"] + ".jpg", annotation["2_way_label"], annotation["clean_title"]])
            self.img_labels = pd.DataFrame(img_labels, columns=["id", "2_way_label", "clean_title"])
            self.img_dir = img_dir
            self.imgs_path = self.img_labels.iloc[:, 0]
            self.texts = self.img_labels.iloc[:, 2]
            self.texts = [clean_text(text) for text in self.texts]

            self.preprocessor = preprocessor
            self.tokenizer = tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
            print(len(self.img_labels))
            print(len(self.imgs_path))
            print(len(self.texts))
        def __len__(self):
            return len(self.img_labels)
        
        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.imgs_path[idx])
            image = Image.open(img_path).convert("RGB")
            label = self.img_labels.iloc[idx, 1]
            text = self.texts[idx]
            if self.tokenizer:
                if text == "" or text == None:
                    text = "Blank"
                text = self.tokenizer(text, 
                                      return_tensors="pt",
                                      padding='max_length',
                                      truncation=True,
                                      return_attention_mask=False,
                                      max_length=n_tokens)
            if self.preprocessor:
                image = self.preprocessor(images=image, return_tensors="pt")
            
            return image, label, text
        

    class EVALITA_Test_Dataset(Dataset):
        def __init__(self, annotations, img_dir, preprocessor=None, tokenizer=None):
            #file = open(annotations_file, "r", encoding="utf-8")
            #annotations = json.load(file)

            #convert to pandas dataframe
            #if the label is "propagandistic" then 1 else 0
            img_labels = []
            for annotation in annotations:
                    img_labels.append([annotation["id"] + ".jpg", annotation["2_way_label"], annotation["clean_title"]])
            img_labels = pd.DataFrame(img_labels, columns=["id", "2_way_label", "clean_title"])
            self.img_labels = img_labels
            self.img_dir = img_dir
            self.imgs_path = self.img_labels.iloc[:, 0] 
            self.texts = self.img_labels.iloc[:, 2]
            self.ids = self.img_labels.iloc[:, 1]
            self.texts = [clean_text(text) for text in self.texts]
            self.preprocessor = preprocessor
            self.tokenizer = tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
            print(len(self.ids))
            print(len(self.imgs_path))
            print(len(self.texts))
        def __len__(self):
            return len(self.img_labels)
        
        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.imgs_path[idx])
            image = Image.open(img_path).convert("RGB")
            id = self.ids[idx]
            text = self.texts[idx]
            if self.tokenizer:
                if text == "" or text == None:
                    text = "Blank"
                text = self.tokenizer(text, 
                                      return_tensors="pt",
                                      padding='max_length',
                                      truncation=True,
                                      return_attention_mask=False,
                                      max_length=n_tokens)
            if self.preprocessor:
                image = self.preprocessor(images=image, return_tensors="pt")
            
            return image, id, text
        
    
    import json
    import pandas as pd

    # Caricamento di Themis, tokenizer, processor
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
    themis.to("cuda")

    # Funzione per caricare i dati da un file TSV
    def load_tsv_file(file_path):
        try:
            df = pd.read_csv(file_path, sep='\t')
            annotations = df.to_dict(orient='records')
        except Exception as e:
            print(f"Errore nel caricare il file TSV: {e}")
            annotations = []
        return annotations

    # Caricamento dei dataset
    annotations_train = load_tsv_file("Fakeddit/annotations/subtask2b/train.tsv")
    dataset_train = EVALITA_Dataset(
        annotations=annotations_train,
        img_dir="Fakeddit/images/2b/train",
        preprocessor=processor,
        tokenizer=tokenizer
    )

    annotations_val = load_tsv_file("Fakeddit/annotations/subtask2b/val.tsv")
    dataset_val = EVALITA_Dataset(
        annotations=annotations_val,
        img_dir="Fakeddit/images/2b/val",
        preprocessor=processor,
        tokenizer=tokenizer
    )

    annotations_test = load_tsv_file("Fakeddit/annotations/subtask2b/dev_unlabeled.tsv")
    dataset_test = EVALITA_Test_Dataset(
        annotations=annotations_test,
        img_dir="Fakeddit/images/2b/dev",
        preprocessor=processor,
        tokenizer=tokenizer
    )


    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,generator=torch.Generator(device='cuda'))
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,generator=torch.Generator(device='cuda'))
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,generator=torch.Generator(device='cuda'))

    loss = nn.BCELoss()
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
                images = images.to("cuda")
                texts = texts.to("cuda")
                labels = labels.to("cuda")
                outputs = themis(images, texts)
                loss_val = loss(outputs.float(), labels.float().unsqueeze(1))
                running_loss += loss_val.item()
                accumulated_labels.extend(labels.to("cpu").numpy())
                accumulated_preds.extend(outputs.to("cpu").detach().numpy())
            
            print(f'Len labels: {len(accumulated_labels)}')
            print(f'Len pred: {len(accumulated_preds)}')
            torch.cuda.empty_cache()
            epoch_loss = running_loss / len(dataloader_val)
            print(f"Validation loss: {epoch_loss}")
            accumulated_preds = [1 if i > 0.5 else 0 for i in accumulated_preds]
            acc = accuracy_score(accumulated_labels, accumulated_preds)
            prec = precision_score(accumulated_labels, accumulated_preds)
            rec = recall_score(accumulated_labels, accumulated_preds)
            f1 = f1_score(accumulated_labels, accumulated_preds)
            print(f"Accuracy: {acc} || Precision: {prec} || Recall: {rec} || F1: {f1}")
            if f1 > best_f1:
                best_f1 = f1
                path_out = "Fakeddit/outputsv2/"+name_llm+"_"+name_img_embed+"_"+str(merge_tokens)+"_"+str(lora_alpha)+"_"+str(lora_r)+"_"+str(lora_dropout)+"_"+str(use_lora)+str(epochs)+"_best.pt"
                if not os.path.exists(os.path.dirname(path_out)):
                    os.makedirs(os.path.dirname(path_out))
                torch.save(themis.state_dict(), path_out)
        return best_f1 
    def train_epoch(themis, dataloader_train, loss, optimizer, epoch):
        print(f"Epoch {epoch+1}/{epochs} current lr: {optimizer.param_groups[0]['lr']}")
        print("-" * 10)
        running_loss = 0.0
        
        for images, labels, texts in tqdm(dataloader_train):
            images = images.to("cuda")
            texts = texts.to("cuda")
            labels = labels.to("cuda")
            optimizer.zero_grad()
            outputs = themis(images, texts)
            loss_val = loss(outputs.float(), labels.float().unsqueeze(1))
            loss_val.backward()
            optimizer.step()
            running_loss += loss_val.item()
        return running_loss 
    torch.cuda.empty_cache()
    #torch.nn.utils.clip_grad_norm_(themis.parameters(), 1.0)
    themis.train()
    best_f1 = 0
    for epoch in range(epochs+warmup_epochs):
        opt = optimizer
        running_loss=train_epoch(themis, dataloader_train, loss, opt, epoch)
        epoch_loss = running_loss / len(dataloader_train)
        print(f"Training loss: {epoch_loss}")
        print("Evaluating...")
        scheduler.step()
        running_loss = 0.0
        best_f1=validate(themis, dataloader_val, loss,running_loss, best_f1=best_f1)
        
        torch.cuda.empty_cache()




