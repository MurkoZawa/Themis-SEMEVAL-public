from torch.utils.data import Dataset, Subset
import pandas as pd
import re
import json 
import torch
from PIL import Image
import os
import numpy as np

def clean_text(text):
    if text == "" or text is None:
        return "Blank"
    if not isinstance(text, str):  
        return "Blank"
    # Remove escape sequences and replace "\\n" with a space
    cleaned_text = re.sub(r'\\n', ' ', text)
    # Remove any other special characters or patterns as needed
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', cleaned_text)
    return cleaned_text

class MFD_Dataset(Dataset):
        def __init__(self, annotations, img_dir, n_tokens, preprocessor=None, tokenizer=None):
            img_labels = []
            for annotation in annotations:
                img_labels.append([annotation["Media"].strip("[]'"), annotation["Label"], annotation["Text"]])
            self.img_labels = pd.DataFrame(img_labels, columns=["Media", "Label", "Text"])
            self.img_dir = img_dir
            self.imgs_path = self.img_labels.iloc[:, 0]
            self.texts = self.img_labels.iloc[:, 2]
            self.texts = [clean_text(text) if text else "No text available" for text in self.texts]
            self.n_tokens = n_tokens

            self.preprocessor = preprocessor
            self.tokenizer = tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                                      max_length=self.n_tokens)
            if self.preprocessor:
                image = self.preprocessor(images=image, return_tensors="pt")
            
            return image, label, text, img_path


class DGM4_Dataset(Dataset):
        def __init__(self, annotations, img_dir, n_tokens, preprocessor=None, tokenizer=None):
            img_labels = []
            for annotation in annotations:
                img_labels.append([str(annotation["image"]), annotation["fake_cls"], annotation["text"]])
            self.img_labels = pd.DataFrame(img_labels, columns=["image", "fake_cls", "text"])
            self.img_dir = img_dir
            self.imgs_path = self.img_labels.iloc[:, 0]
            self.texts = self.img_labels.iloc[:, 2]
            self.texts = [clean_text(text) for text in self.texts]
            self.n_tokens = n_tokens

            self.preprocessor = preprocessor
            self.tokenizer = tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                                      max_length=self.n_tokens)
            if self.preprocessor:
                image = self.preprocessor(images=image, return_tensors="pt")
            
            return image, label, text, img_path
            
class DGM4_text_Dataset(Dataset):
    def __init__(self, annotations, n_tokens, tokenizer=None):
        img_labels = []
        for annotation in annotations:
            img_labels.append([annotation["fake_cls"], annotation["text"]])
        self.img_labels = pd.DataFrame(img_labels, columns=["fake_cls", "text"])
        self.texts = self.img_labels.iloc[:, 1]
        self.texts = [clean_text(text) for text in self.texts]
        self.n_tokens = n_tokens

        self.tokenizer = tokenizer
        if self.tokenizer:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(len(self.img_labels))
        print(len(self.texts))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 0]
        text = self.texts[idx]
        if self.tokenizer:
            if text == "" or text is None:
                text = "Blank"
            text = self.tokenizer(
                text, 
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                return_attention_mask=False,
                max_length=self.n_tokens
            )
        return label, text

class Fakeddit_Dataset(Dataset):
        def __init__(self, annotations, img_dir, n_tokens, preprocessor=None, tokenizer=None):
            img_labels = []
            for annotation in annotations:
                img_labels.append([annotation["id"] + ".jpg", annotation["2_way_label"], annotation["clean_title"]])
            self.img_labels = pd.DataFrame(img_labels, columns=["id", "2_way_label", "clean_title"])
            self.img_dir = img_dir
            self.imgs_path = self.img_labels.iloc[:, 0]
            self.texts = self.img_labels.iloc[:, 2]
            self.texts = [clean_text(text) for text in self.texts]
            self.n_tokens = n_tokens

            self.preprocessor = preprocessor
            self.tokenizer = tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                                      max_length=self.n_tokens)
            if self.preprocessor:
                image = self.preprocessor(images=image, return_tensors="pt")
            
            return image, label, text, img_path

class Recovery_Dataset(Dataset):
        def __init__(self, annotations, img_dir, n_tokens, preprocessor=None, tokenizer=None):
            img_labels = []
            for annotation in annotations:
                img_labels.append([str(annotation['id']) + '.jpg', annotation["label"], annotation["title"] + annotation['text']])
            self.img_labels = pd.DataFrame(img_labels, columns=["img", "label", "text"])
            self.img_dir = img_dir
            self.imgs_path = self.img_labels.iloc[:, 0]
            self.texts = self.img_labels.iloc[:, 2]
            self.texts = [clean_text(text) for text in self.texts]
            self.n_tokens = n_tokens

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
                                      max_length=self.n_tokens)
            if self.preprocessor:
                image = self.preprocessor(images=image, return_tensors="pt")
            
            return image, label, text, img_path
        
        
class Memes_Dataset(Dataset):
    def __init__(self, annotations, img_dir, n_tokens, preprocessor=None, tokenizer=None):
        img_labels = []
        for annotation in annotations:
            img_labels.append([annotation["image"], annotation["label"], annotation["text"]])
        self.img_labels = pd.DataFrame(img_labels, columns=["image", "label", "text"])
        self.img_dir = img_dir
        self.imgs_path = self.img_labels.iloc[:, 0]
        self.texts = self.img_labels.iloc[:, 2]
        self.texts = [clean_text(text) for text in self.texts]
        self.n_tokens = n_tokens

        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            if text == "" or text is None:
                text = "Blank"
            text = self.tokenizer(text, 
                                  return_tensors="pt",
                                  padding='max_length',
                                  truncation=True,
                                  return_attention_mask=False,
                                  max_length=self.n_tokens)
        if self.preprocessor:
            image = self.preprocessor(images=image, return_tensors="pt")
        
        return image, label, text

   
# Carica le annotazioni da file_path per MFD 
def mfd_load_annotations_file(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        annotations = df.to_dict(orient='records')
    except Exception as e:
        print(f"Errore nel caricare il file: {e}")
        annotations = []
    return annotations

# Carica le annotazioni da file_path per DGM4   
def dgm4_load_annotations_file(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        annotations = df.to_dict(orient='records')
    except Exception as e:
        print(f"Errore nel caricare il file: {e}")
        annotations = []
    return annotations
   
# Carica le annotazioni da file_path per Fakeddit    
def fakeddit_load_annotations_file(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        annotations = df.to_dict(orient='records')
    except Exception as e:
        print(f"Errore nel caricare il file: {e}")
        annotations = []
    return annotations

# Carica le annotazione da file_path per Recovery
def recovery_load_annotations_file(file_path):
    try:
        df = pd.read_csv(file_path, sep=',')
        annotations = df.to_dict(orient='records')
    except Exception as e:
        print(f"Errore nel caricare il file: {e}")
        annotations = []
    return annotations

# Funzione per caricare le annotazioni dai file JSON per Memes
def memes_load_annotations_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    except Exception as e:
        print(f"Errore nel caricare il file: {e}")
        annotations = []
    return annotations

# Restituire un dataset
def get_dataset(Dataset, load_annotations_func, n_tokens, processor, tokenizer, ann_path, img_dir):
    # Caricamento delle annotazioni dal file
    annotations = load_annotations_func(ann_path)

    # Creazione del dataset
    dataset = Dataset(
        annotations=annotations,
        img_dir=img_dir,
        n_tokens=n_tokens,
        preprocessor=processor,
        tokenizer=tokenizer
    )
    return dataset
    
# Restituire un dataset per il testo
def get_dataset_text(Dataset, load_annotations_func, n_tokens, tokenizer, ann_path):
    # Caricamento delle annotazioni dal file
    annotations = load_annotations_func(ann_path)

    # Creazione del dataset (senza immagini, solo testo)
    dataset = Dataset(
        annotations=annotations,
        n_tokens=n_tokens,
        tokenizer=tokenizer
    )
    return dataset

